import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np


def permute_dataset(dataset):

    superclass_to_fine = [
        [4, 30, 55, 72, 95],  # Aquatic mammals: beaver, dolphin, otter, seal, whale
        [1, 32, 67, 73, 91],  # Fish: aquarium fish, flatfish, ray, shark, trout
        [54, 62, 70, 82, 92],  # Flowers: orchids, poppies, roses, sunflowers, tulips
        [9, 16, 28, 61, 10],  # Food containers: bottles, bowls, cans, cups, plates
        [0, 51, 53, 57, 83],  # Fruit and vegetables: apples, mushrooms, oranges, pears, sweet peppers
        [22, 39, 40, 86, 87],  # Household electrical devices: clock, keyboard, lamp, telephone, television
        [5, 20, 25, 84, 94],  # Household furniture: bed, chair, couch, table, wardrobe
        [6, 7, 14, 18, 24],  # Insects: bee, beetle, butterfly, caterpillar, cockroach
        [3, 42, 43, 88, 97],  # Large carnivores: bear, leopard, lion, tiger, wolf
        [12, 17, 37, 68, 76],  # Large man-made outdoor things: bridge, castle, house, road, skyscraper
        [23, 33, 49, 60, 71],  # Large natural outdoor scenes: cloud, forest, mountain, plain, sea
        [15, 19, 21, 31, 38],  # Large omnivores and herbivores: camel, cattle, chimpanzee, elephant, kangaroo
        [34, 63, 64, 66, 75],  # Medium-sized mammals: fox, porcupine, possum, raccoon, skunk
        [26, 45, 77, 79, 99],  # Non-insect invertebrates: crab, lobster, snail, spider, worm
        [2, 11, 35, 46, 98],  # People: baby, boy, girl, man, woman
        [27, 29, 44, 78, 93],  # Reptiles: crocodile, dinosaur, lizard, snake, turtle
        [36, 50, 65, 74, 80],  # Small mammals: hamster, mouse, rabbit, shrew, squirrel
        [47, 52, 56, 59, 96],  # Trees: maple, oak, palm, pine, willow
        [8, 13, 48, 58, 90],  # Vehicles 1: bicycle, bus, motorcycle, pickup truck, train
        [41, 69, 81, 85, 89]  # Vehicles 2: lawn mower, rocket, streetcar, tank, tractor
    ]

    classes = np.array(superclass_to_fine).flatten().tolist()

    new_label_order = {v: i for i, v in enumerate(classes)}

    permuted_targets = [new_label_order[label] for label in dataset.targets]

    permuted_data = [x for _, x in sorted(zip(permuted_targets, dataset.data), key=lambda pair: pair[0])]

    return permuted_data, permuted_targets


class PermutedDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class TrainTestData:

    def __init__(self, dataset_name: str, color_model: str):

        self.dataset_name = dataset_name

        self.color_model = color_model

    def transform(self):

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))]

        if self.color_model in ['HSV', 'iHSV']:
            transform_list.append(transforms.RGB2HSV())  # Note: RGB2HSV may not exist in PyTorch
        elif self.color_model == 'LAB':
            transform_list.append(transforms.RGB2LAB())

        return transforms.Compose(transform_list)

    def initial_load_dataset(self, base_path, download = False):

        if self.dataset_name == 'CIFAR-10':

            train_dataset = datasets.CIFAR10(root=os.path.join(base_path, 'data'), train=True, download=download,
                                             transform=self.transform)
            test_dataset = datasets.CIFAR10(root=os.path.join(base_path, 'data'), train=False, download=download,
                                            transform=self.transform)

        elif self.dataset_name == 'CIFAR-100':

            train_dataset = datasets.CIFAR100(root=os.path.join(base_path, 'data'), train=True, download=download,
                                              transform=self.transform)
            test_dataset = datasets.CIFAR100(root=os.path.join(base_path, 'data'), train=False, download=download,
                                             transform=self.transform)

        elif self.dataset_name == 'SVHN':

            train_dataset = datasets.SVHN(root=os.path.join(base_path, 'data_svhn'), split='train', download=download,
                                          transform=self.transform)
            test_dataset = datasets.SVHN(root=os.path.join(base_path, 'data_svhn'), split='test', download=download,
                                         transform=self.transform)

        elif self.dataset_name == 'STL-10':

            train_dataset = datasets.STL10(root=os.path.join(base_path, 'data_STL10'), split='train', download=download,
                                           transform=self.transform)
            test_dataset = datasets.STL10(root=os.path.join(base_path, 'data_STL10'), split='test', download=download,
                                          transform=self.transform)

        elif self.dataset_name == 'tiny_imagenet':

            train_dataset = datasets.ImageFolder('tiny-imagenet-200/train', transform=self.transform)
            test_dataset = datasets.ImageFolder('tiny-imagenet-200/test', transform=self.transform)

        elif self.dataset_name == 'permuted_CIFAR-100':

            train_dataset = datasets.CIFAR100(root=os.path.join(base_path, 'data'), train=True, download=download,
                                              transform=self.transform)
            test_dataset = datasets.CIFAR100(root=os.path.join(base_path, 'data'), train=False, download=download,
                                             transform=self.transform)

            permute_train_data, permute_train_targets = permute_dataset(train_dataset)
            permute_test_data, permute_test_targets = permute_dataset(test_dataset)

            train_dataset = PermutedDataset(permute_train_data, permute_train_targets, transform=self.transform)
            test_dataset = PermutedDataset(permute_test_data, permute_test_targets, transform=self.transform)

        elif self.dataset_name == 'clothing':
            pass

        return train_dataset, test_dataset
