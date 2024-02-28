import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import quantize

def rgb_image_transform(x, num_bins=256):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()

class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]

def get_cifar10_datasets(num_bins:int = 16) -> tuple[Dataset, Dataset, Dataset]:
    train_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Lambda(lambda x: rgb_image_transform(x, num_bins))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: rgb_image_transform(x, num_bins))])
    
    train_set = CIFAR10(train=True, download=True, transform=train_transform)
    val_set = CIFAR10(train=True, download=True, transform=test_transform)
    test_set = CIFAR10(train=False, download=True, transform=test_transform)

    return train_set, val_set, test_set

def get_cifar10_dataloaders(batch_size:int = 64, num_bins:int = 16) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = get_cifar10_datasets(num_bins)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader