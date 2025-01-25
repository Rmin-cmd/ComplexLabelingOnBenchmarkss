import torch
from utils.utils_label import label_encoding
from utils.utils_loss import loss_function
from utils.utils_color import ColorModelConverter
import complextorch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainTestPipeline:

    def __init__(self, model, criterion, beta=0.5, temperature=1, hsv_ihsv_flag=True):
        self.model = model
        self.criterion = criterion
        self.hsv_ihsv_flag = hsv_ihsv_flag
        self.converter = ColorModelConverter(device)
        self.beta, self.temperature = beta, temperature

    def train(self, train_loader:DataLoader, optimizer):
        self.model.train()
        train_loss = 0.0
        total_correct, total_targets = 0, 0
        for inputs, targets in train_loader:
            label = label_encoding(targets).to(device)

            if self.hsv_ihsv_flag:
                inputs, target = self.converter.convert_hsv_ihsv(inputs, targets)
            else:
                inputs, target = complextorch.CVTensor(inputs, i=torch.zeros_like(inputs)).to(device), targets.to(
                    device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss, predicted_label, preds = loss_function(self.criterion, outputs, label, target, beta=self.beta,
                                                         temperature=self.temperature)
            total_correct += torch.sum(predicted_label == target.data).item()
            total_targets += targets.size(0)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss/len(train_loader), total_correct/total_targets

    def test(self, test_loader:DataLoader):

        self.model.eval()
        test_loss = 0
        pred_, label_ = [], []

        with torch.no_grad():

            for inputs, targets in test_loader:
                label = label_encoding(targets).to(device)
                if self.hsv_ihsv_flag:
                    inputs, target = self.converter.convert_hsv_ihsv(inputs, targets)
                else:
                    inputs, target = complextorch.CVTensor(inputs, i=torch.zeros_like(inputs)).to(device), targets.to(device)
                # inputs, target = complextorch.CVTensor(inputs, i=inputs).to(device), targets.to(device)
                outputs = self.model(inputs)
                loss, predicted_label, preds = loss_function(self.criterion, outputs, label, target, beta=self.beta,
                                                             temperature= self.temperature)
                test_loss += loss
                # total_correct += torch.sum(predicted_label == target.data).item()

                pred_ += preds
                label_ += targets.tolist()

            return test_loss/len(test_loader), pred_, label_
