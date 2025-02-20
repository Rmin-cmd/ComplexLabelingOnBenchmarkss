import complextorch.nn as compnn
import torch.nn as nn
from utils.batch_norm import ComplexBatchNorm2d
from utils.LazyCVLinear import LazyCVLinear


class ComplexNet(nn.Module):
    def __init__(self, dropout=0.5, output_neurons=10):
        super(ComplexNet, self).__init__()
        self.conv1 = compnn.CVConv2d(3, 10, kernel_size=(5, 5), stride=(1, 1))
        self.relu1 = compnn.CVCardiod()
        self.bn1 = ComplexBatchNorm2d(10)
        self.conv2 = compnn.CVConv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
        self.bn2 = ComplexBatchNorm2d(20)
        self.fc1 = LazyCVLinear(50)
        self.dp1 = compnn.CVDropout(p=dropout)
        self.fc2 = LazyCVLinear(output_neurons)
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
        # x = x.view(-1, 20 * 5 * 5)
        # x = x.view(-1, 20 * 21 * 21)
        x = self.fc1(x.view(x.shape[0], -1))
        x = self.dp1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = x.abs()
        # out = self.smx(x)
        return x


class ComplexCifarNet(nn.Module):
    def __init__(self, dropout=0.5,  output_neurons=10):
        super(ComplexCifarNet, self).__init__()
        self.conv1 = compnn.CVConv2d(3, 64, kernel_size=(5, 5), padding=2)
        self.conv2 = compnn.CVConv2d(64, 64, kernel_size=(5, 5), padding=2)
        self.conv3 = compnn.CVConv2d(64, 128, kernel_size=(5, 5), padding=2)
        self.fc1 = LazyCVLinear(384)
        self.fc1 = LazyCVLinear(192)
        self.fc1 = LazyCVLinear(output_neurons)
        self.dropout = nn.Dropout(dropout)
        self.activation = compnn.CReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        mxpool1 = compnn.CVAdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))
        x = mxpool1(x)
        x = self.activation(self.conv2(x))
        mxpool2 = compnn.CVAdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))
        x = mxpool2(x)
        x = self.activation(self.conv3(x))
        mxpool3 = compnn.CVAdaptiveAvgPool2d((x.shape[2] // 2, x.shape[3] // 2))
        x = mxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


