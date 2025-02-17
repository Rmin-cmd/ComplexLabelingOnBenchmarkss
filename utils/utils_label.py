import torch
import numpy as np


def label_encoding(labels, dataset='CIFAR-10'):
    label_en = torch.zeros_like(labels).to(torch.complex64)

    if dataset == 'CIFAR-10' or dataset == 'SVHN' or dataset == 'STL-10':
        label_comp = torch.tensor([np.exp(2j * np.pi * (i / 10)) for i in range(10)])

    if dataset == 'CIFAR-100':
        label_comp = torch.tensor([np.exp(2j * np.pi * (i / 100)) for i in range(100)])

    # label_comp = [np.exp(2j * np.pi * (22.5 / 360)), np.exp(2j * np.pi * (270 / 360)), np.exp(2j * np.pi * (57.5 / 360)),
    #               np.exp(2j * np.pi * (135 / 360)), np.exp(2j * np.pi * (153 / 360)), np.exp(2j * np.pi * (117 / 360)),
    #               np.exp(2j * np.pi * (99 / 360)), np.exp(2j * np.pi * (171 / 360)), np.exp(2j * np.pi * (0 / 360)), np.exp(2j * np.pi * (180 / 360))]

    for lbl in torch.unique(labels):

        label_en[labels == lbl] = label_comp[lbl]

    return label_en