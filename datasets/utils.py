import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


def idx_to_float(idx: np.ndarray, num_bins: int):
    flt_zero_one = (idx + 0.5) / num_bins
    return (2.0 * flt_zero_one) - 1.0

def float_to_idx(flt: np.ndarray, num_bins: int):
    flt_zero_one = (flt / 2.0) + 0.5
    return torch.clamp(torch.floor(flt_zero_one * num_bins), min=0, max=num_bins - 1).long()

def quantize(flt, num_bins: int):
    return idx_to_float(float_to_idx(flt, num_bins), num_bins)

    
def plot_tensor_images(images, n=8):
    fig, axes = plt.subplots(1, n, figsize=(10, 10))
    for i in range(n):
        image = images[i].squeeze().cpu().detach().numpy()
        image = (image + 1) / 2  # Rescale from (-1, 1) to (0, 1)
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
    plt.show()

def sandwich(x: Tensor):
    return x.reshape(x.size(0), -1, x.size(-1))

def pe_encode(sequence_length: int, embedding_size: int) -> Tensor:
    """Positional encoding as described in original attention is all you need paper"""

    pe = torch.zeros((sequence_length, embedding_size))
    pos = torch.arange(sequence_length).unsqueeze(1)
    pe[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )
    pe[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )

    return pe


def pe_encode_float(x: Tensor, max_freq: float, embedding_size: int) -> Tensor:
    pe = torch.zeros(list(x.shape) + [embedding_size], device=x.device)
    pos = (((x + 1) / 2) * max_freq).unsqueeze(-1)
    pe[..., 0::2] = torch.sin(
        pos
        / torch.pow(10000, torch.arange(0, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    pe[..., 1::2] = torch.cos(
        pos
        / torch.pow(10000, torch.arange(1, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    return pe