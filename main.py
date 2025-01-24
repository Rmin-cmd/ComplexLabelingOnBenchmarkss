import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import complextorch as comptorch
import complextorch.nn as compnn
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import ComplexBatchNorm2d
from torch import Tensor
from utils_color import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

num_epochs = 50
batch_size = 256
learning_rate = 0.00001
num_folds = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
beta = 0.5
ihsv_color_model = True



