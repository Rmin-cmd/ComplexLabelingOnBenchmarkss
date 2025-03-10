import os
import hydra
from omegaconf import OmegaConf,DictConfig
import torch
import optuna
from optuna import Trial
from sklearn.model_selection import KFold
import torch.optim as optim
from Models.models import *
# from Models.networks import *
# from Models.networks_new import *
from Models.FCCN_model import networks as net
# from Models.FCCN_model import model as net_new
import torch.nn as nn
from train_test import TrainTestPipeline
from train_test_data import TrainTestData
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from utils.utils import Metrics, show_confusion
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path='config', config_name='config.yaml', version_base=None)
def main(cfg: DictConfig):

    dataset = TrainTestData(dataset_name=cfg.datasets.name, color_model=cfg.training.color_model)

    print(cfg.datasets.name)

    train_dataset, test_dataset, classnames = dataset.initial_load_dataset(base_path=cfg.training.base_path, download=True)

    metrics = Metrics(num_class=len(classnames))

    if cfg.training.Tuning:

        optuna_path = os.path.join(os.getcwd(), 'optuna_results')

        try:

            os.mkdir(optuna_path)

            RESULT_OPTUNA_PATH = os.path.join(optuna_path, f'nested_cv_{cfg.datasets.name}.sqlite3')

        except FileExistsError:

            RESULT_OPTUNA_PATH = os.path.join(optuna_path, f'nested_cv_{cfg.datasets.name}.sqlite3')

        except Exception as e:

            raise RuntimeError('Specified Path is not correct.')

        kf = KFold(n_splits=cfg.training.n_folds, shuffle=True, random_state=cfg.training.random_state)

        study = optuna.create_study(
            study_name=f'nested_cv_tuning_{cfg.datasets.name}',
            directions=["maximize"],
            storage=f'sqlite:///{RESULT_OPTUNA_PATH}',
            sampler=optuna.samplers.TPESampler(),
            load_if_exists=True
        )

        def objective(trial: Trial):

            learning_rate = trial.suggest_float('learning_rate', low=1e-4, high=1e-2)

            l2 = trial.suggest_categorical('l2', choices=[0.01, 1e-5])

            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

            drop_out = trial.suggest_categorical('drop_out', [0.0, 0.2, 0.4, 0.5])

            beta_loss = trial.suggest_float('beta', low=0.2, high=0.8)

            temperature = trial.suggest_float('temperature', low=0.2, high=1.0)

            mean_loss, mean_f1_score = [], []

            # for datasets where the input image is in size 32 * 32 use flag 1

            for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):

                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
                val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

                # model = ComplexNet(dropout=cfg.datasets.drop_out, output_neurons=len(classnames)).to(device)
                # model = ComplexCifarNet(dropout=drop_out, output_neurons=len(classnames)).to(device)
                model = net.CDS_E(len(classnames), dropout=drop_out).to(device)

                criterion = nn.CrossEntropyLoss()

                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

                train_test_pip = TrainTestPipeline(model, cfg.datasets.name, criterion, beta=beta_loss, temperature=temperature,
                                                   hsv_ihsv_flag=cfg.training.color_model)

                best_loss, best_f1, early_stopping = torch.inf, 0, 0

                for epoch in tqdm(range(cfg.training.epochs)):

                    _, _, _ = train_test_pip.train(train_loader, optimizer)

                    loss, predicted_labels, ground_truth = train_test_pip.test(test_loader=val_loader)

                    out_metrics = metrics.compute_metrics(torch.tensor([predicted_labels]).to(device),
                                                          torch.tensor([ground_truth]).to(device))

                    if loss < best_loss and out_metrics[3] > best_f1:
                        early_stopping = 0
                        best_loss, best_f1 = loss, out_metrics[3]
                    else:
                        early_stopping += 1
                    if early_stopping > 5:

                        # torch.save(model.state_dict(), log_path + '\\model_latest_fold'+str(fold)+'.t7')
                        break

                    scheduler.step(out_metrics[3])

                mean_loss.append(best_loss.item())
                mean_f1_score.append(best_f1)

            return np.mean(mean_f1_score)

        study.optimize(
            func=objective,
            n_trials=cfg.training.n_trials,
            show_progress_bar=True,
            catch=[AttributeError, ValueError]
        )

        config = OmegaConf.create({
            "datasets": {
                "name": cfg.datasets.name,
                **study.best_params
            }
        })

        config_yaml = "# @package _global_\n" + OmegaConf.to_yaml(config)

        config_dataset_save = os.path.join('config', 'datasets', f'{cfg.datasets.name}.yaml')

        with open(config_dataset_save, 'w') as f:
            f.write(config_yaml)

    else:

        # model = ComplexNet(dropout=cfg.datasets.drop_out, output_neurons=len(classnames)).to(device)

        # FCCN model
        model = net.CDS_E(len(classnames), dropout=cfg.datasets.drop_out).to(device)

        # model = net_new.CDS_E(dset_type='iHSV', outsize=len(classnames)).to(device)

        # model = ComplexCifarNet(dropout=cfg.datasets.drop_out, output_neurons=len(classnames)).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg.datasets.learning_rate, weight_decay=cfg.datasets.l2)

        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        criterion = nn.CrossEntropyLoss()

        train_dataset, valid_dataset = random_split(train_dataset,
                                                    [int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))])

        train_loader = DataLoader(train_dataset, batch_size=cfg.datasets.batch_size, shuffle=True)
        valid_loader = DataLoader(test_dataset, batch_size=cfg.datasets.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.datasets.batch_size, shuffle=False)

        writer = SummaryWriter(
            log_dir=os.path.join('Tensorboard_results', f'runs_new_loss_tmp_{cfg.datasets.name}'))

        train_test_pip = TrainTestPipeline(model, cfg.datasets.name, criterion, beta=cfg.datasets.beta, temperature=cfg.datasets.temperature,
                                           hsv_ihsv_flag=cfg.training.color_model)

        save_path = os.path.join(os.getcwd(), 'saved_models')

        try:

            os.mkdir(save_path)

            MODEL_SAVE_PATH = os.path.join(save_path, f'best_model_for_new_loss_{cfg.datasets.name}.pth')

        except FileExistsError:

            MODEL_SAVE_PATH = os.path.join(save_path, f'best_model_for_new_loss_{cfg.datasets.name}.pth')


        if cfg.training.Training:

            conf_mat_epochs = []

            early_stopping = 0

            best_acc, best_loss = 0, np.inf

            for epoch in tqdm(range(cfg.training.epochs)):

                train_loss, train_accuracy, model = train_test_pip.train(train_loader, optimizer)

                loss_valid, predicted_labels, ground_truth = train_test_pip.test(test_loader=valid_loader)

                out_metrics = metrics.compute_metrics(torch.tensor([predicted_labels]).to(device),
                                                      torch.tensor([ground_truth]).to(device))

                # scheduler.step(out_metrics[3])

                outstrtrain = '\n epoch:%d, Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                              (epoch, loss_valid, out_metrics[0], out_metrics[1], out_metrics[2],
                               out_metrics[3])

                print(outstrtrain)

                pred_np, label_np = np.array(torch.tensor(predicted_labels).tolist()), np.array(torch.tensor(ground_truth).tolist())

                conf_mat_epochs.append(confusion_matrix(label_np, pred_np))

                writer.add_scalars('Loss', {'Train': train_loss,
                                            'Validation': loss_valid}, epoch)

                writer.add_scalars("Accuracy", {'Train': train_accuracy,
                                                'Valid': out_metrics[0]}, epoch)

                writer.add_scalar("recall/val", out_metrics[1], epoch)
                writer.add_scalar("precision/val", out_metrics[2], epoch)
                writer.add_scalar("F1 Score", out_metrics[3], epoch)

                if loss_valid < best_loss and out_metrics[0] > best_acc:
                    early_stopping = 0
                    best_acc = out_metrics[0]
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                else:

                    early_stopping += 1

                if early_stopping > 5:
                    # pass
                    # torch.save(model.state_dict(), log_path + '\\model_latest_fold'+str(fold)+'.t7')
                    break

            fig = show_confusion(np.mean(conf_mat_epochs, axis=0), class_names=classnames, show=False)

            writer.add_figure("Confusion Matrix", fig)

        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        print(model)

        loss_valid, predicted_labels, ground_truth = train_test_pip.test(test_loader=test_loader, model=model)

        out_metrics = metrics.compute_metrics(torch.tensor([predicted_labels]).to(device),
                                              torch.tensor([ground_truth]).to(device))

        outstrtrain = 'Test Results\n Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                      (loss_valid, out_metrics[0], out_metrics[1], out_metrics[2],
                       out_metrics[3])

        print(outstrtrain)


if __name__ == '__main__':
    main()




