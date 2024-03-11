import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from .utils import quantize

def rgb_image_transform(x, num_bins=16):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()

class CIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, *args, exclude_list=[0, 1, 3, 4, 5, 6, 7, 8, 9], **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        if exclude_list == []:
            return
        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
        self.data = self.data[mask]
        self.train_labels = labels[mask].tolist()

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]
    
        
    
def get_standard_transform(num_bins, train=True):
    return transforms.Compose([transforms.ToTensor(),
                            transforms.RandomHorizontalFlip() if train else nn.Identity(),
                            transforms.RandomAutocontrast() if train else nn.Identity(),
                            transforms.Lambda(lambda x: rgb_image_transform(x, num_bins)),])

def get_cifar10_datasets(num_bins:int = 16, root='./datasets/') -> tuple[Dataset, Dataset, Dataset]:
    # create datasets, train and val are the same (apart from transform), test is official cifar10 test set
    train_set = CIFAR10(root=root, train=True, download=True, transform=get_standard_transform(num_bins, train=True))
    val_set = CIFAR10(root=root, train=True, download=True, transform=get_standard_transform(num_bins, train=False))
    test_set = CIFAR10(root=root, train=False, download=True, transform=get_standard_transform(num_bins, train=False))
    return train_set, val_set, test_set

def get_cifar10_dataloaders(batch_size:int = 32, num_bins:int = 16, valid_size=0.01, seed=7) -> tuple[DataLoader, DataLoader, DataLoader]:

    # get datasets
    train_set, val_set, test_set = get_cifar10_datasets(num_bins)

    # train and val are the same by default, split them up using SubsetRandomSampler
    indices = list(range(len(train_set)))
    split = int(valid_size*len(train_set))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # create dataloaders, using samplers
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_set, sampler=valid_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader