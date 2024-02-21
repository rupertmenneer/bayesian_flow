import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BinaryData(Dataset):
    """
    A dataset class that generates binary variables.

    Args:
        n (int): The number of samples in the dataset.
        k (int): The number of binary variables in each sample.

    Attributes:
        n (int): The number of samples in the dataset.
        k (int): The number of binary variables in each sample.
        data (torch.Tensor): The generated binary data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the binary data at the given index.
        create_binary_data(p): Generates binary data using softmax function and thresholding.

    """

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.data = self.create_binary_data()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def create_binary_data(self, p=0.5):
        x0 = torch.randint(0, 2, size=(self.n,), dtype=torch.bool)
        x1 = ~x0
        return torch.vstack([x0, x1]).to(torch.int64).T

# Simple wrapper dataloader class
class BinaryDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        super(BinaryDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)