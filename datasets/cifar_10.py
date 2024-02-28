import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import quantize

def rgb_image_transform(x, num_bins=256):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()

class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]
    
def get_standard_transform(num_bins, train=True):
    return transforms.Compose([transforms.ToTensor(),
                            transforms.RandomHorizontalFlip() if train else transforms.Identity(),
                            transforms.Lambda(lambda x: rgb_image_transform(x, num_bins)),
                            transforms.Lambda(lambda x: (2*x)-1) ])

def get_cifar10_datasets(num_bins:int = 16, root='./datasets/') -> tuple[Dataset, Dataset, Dataset]:
    train_set = CIFAR10(root=root, train=True, download=True, transform=get_standard_transform(num_bins, train=True))
    val_set = CIFAR10(root=root, train=True, download=True, transform=get_standard_transform(num_bins, train=False))
    test_set = CIFAR10(root=root, train=False, download=True, transform=get_standard_transform(num_bins, train=False))
    return train_set, val_set, test_set

def get_cifar10_dataloaders(batch_size:int = 32, num_bins:int = 16) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = get_cifar10_datasets(num_bins)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader