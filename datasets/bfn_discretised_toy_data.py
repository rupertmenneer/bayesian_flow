import torch
from torch.utils.data import Dataset
from .utils import quantize

class DiscretisedBimodalData(Dataset):
    """
    A dataset class that generates a bimodal distribution and discretises it into K bins.

    Args:
        n (int): The number of samples in the dataset.
        k (int): The number of data bins.

    Attributes:
        n (int): The number of samples in the dataset.
        k (int): The number of binary variables in each sample.
        data (torch.Tensor): The generated binary data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the binary data at the given index.
        create_binary_data(p): Generates binary data using softmax function and thresholding.

    """

    def __init__(self, n=1024, k=16):
        self.n = n
        self.k = k
        self.k_centers = self.get_k_centers()
        self.k_lower = self.k_centers - (1/self.k)
        self.k_upper = self.k_centers + (1/self.k)
        self.data = self.create_discretised_bimodal()


    def get_k_centers(self):
        k_ = torch.linspace(1, self.k, self.k )
        return ((2 * k_ - 1)/self.k) - 1

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        x = self.data[index]
        # clip data to between -1 and 1
        x = torch.clamp(x, min=-1, max=1)
        return x.unsqueeze(-1)

    def create_discretised_bimodal(self):
        # Create a bimodal distribution and generate samples from it
        mean_0, std_0 = -1, 0.25
        mean_1, std_1 = 1, 0.25
        dist_0 = torch.distributions.normal.Normal(mean_0, std_0)
        dist_1 = torch.distributions.normal.Normal(mean_1, std_1)
        samples_0 = dist_0.sample((self.n // 2,))
        samples_1 = dist_1.sample((self.n // 2,))
        # combine the distributions
        bimodal_data = torch.cat((samples_0, samples_1), dim=0)
        
        # normalize the bimodal data between -1 and 1 using min and max
        min_val = torch.min(bimodal_data)
        max_val = torch.max(bimodal_data)
        normalized_data = (bimodal_data - min_val) / (max_val - min_val) * 2 - 1
        
        # discretise the bimodal distribution into K bins
        discretised_data = quantize(normalized_data, self.k)

        return discretised_data
    
