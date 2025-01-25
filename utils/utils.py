import torchmetrics as tcm
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Metrics:

    def __init__(self, num_class):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.accuracy = tcm.Accuracy(task='multiclass', num_classes=num_class).to(device)

        self.recall = tcm.Recall(task='multiclass', num_classes=num_class, average='macro').to(device)

        self.precision = tcm.Precision(task='multiclass', num_classes=num_class, average='macro').to(device)

        self.F1_score = tcm.F1Score(task='multiclass', num_classes=num_class, average='macro').to(device)

    def compute_metrics(self, preds, target):

        # _, preds = preds.max(-1)

        # _, target = target.max(-1)

        return [self.accuracy(preds, target).item(), self.recall(preds, target).item(),
               self.precision(preds, target).item(), self.F1_score(preds, target).item()]


def show_confusion(conf_mat, class_names, show=True):

    fig = plt.figure()

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    if show:
        plt.show()
    else:
        return fig


def save_best_params_to_hydra_config(best_params, fold_idx, dataset_name):
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)  # Ensure the directory exists
    config_file = config_dir / f"{dataset_name}_best_params.yaml"

    # Load existing config if it exists
    if config_file.exists():
        with open(config_file, "r") as f:
            existing_config = yaml.safe_load(f) or {}
    else:
        existing_config = {}

    # Update the configuration with the new best parameters
    existing_config[f"fold_{fold_idx}"] = best_params

    # Save the updated config to the YAML file
    with open(config_file, "w") as f:
        yaml.dump(existing_config, f)