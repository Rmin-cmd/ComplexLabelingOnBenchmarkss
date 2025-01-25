import complextorch.nn as compnn
import torch.nn as nn
from utils.batch_norm import ComplexBatchNorm2d


class ComplexNet(nn.Module):
    def __init__(self, flag=1, dropout=0.5):
        super(ComplexNet, self).__init__()
        self.conv1 = compnn.CVConv2d(3, 10, kernel_size=(5, 5), stride=(1, 1))
        self.relu1 = compnn.CVCardiod()
        self.bn1 = ComplexBatchNorm2d(10)
        self.conv2 = compnn.CVConv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
        self.bn2 = ComplexBatchNorm2d(20)
        if flag:
            self.fc1 = compnn.CVLinear(5 * 5 * 20, 50)
        else:
            self.fc1 = compnn.CVLinear(21 * 21 * 20, 50)
        self.dp1 = compnn.CVDropout(p=0.5)
        # self.fc1 = compnn.CVLinear(21 * 21 * 20, 50)
        # self.fc2 = compnn.CVLinear(50, 10)
        self.fc2 = compnn.CVLinear(50, 10)
        self.smx = compnn.PhaseSoftMax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.dp1(x)
        mxpool1 = compnn.CVAdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))
        x = mxpool1(x)
        # x = self.bn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu1(x)
        # x = self.dp1(x)
        mxpool1 = compnn.CVAdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))
        x = mxpool1(x)
        x = x.view(-1, 20 * 5 * 5)
        # x = x.view(-1, 20 * 21 * 21)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = x.abs()
        # out = self.smx(x)
        return x
