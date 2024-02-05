import torch
import torch_ema
from math import inf
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch_ema
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import relu
from torch.optim import AdamW

class BayesianFlowNetwork2D(nn.Module):
    def __init__(self, network, D, K, beta=3.0):
        super(BayesianFlowNetwork2D, self).__init__()
        self.beta = beta
        self.D = D
        self.K = K
        self.network = network

    def forward(self, theta: torch.Tensor, t: torch.Tensor, ema: torch_ema.ExponentialMovingAverage) -> torch.Tensor:
        """
        Performs a forward pass.

        Args:
            theta (torch.Tensor)
            t (torch.Tensor): time factor
            ema (torch_ema.ExponentialMovingAverage): EMA used in the model.

        Returns:
            torch.Tensor
        """
        theta = (theta * 2) - 1  # scaled in [-1, 1]
        theta = torch.transpose(theta, 1, 3)
        if ema is not None:
          with ema.average_parameters():
            output = self.network(theta + t[:, None, None, None])  # (B, D, D, K)
        else:
          output = self.network(theta + t[:, None, None, None])  # (B, D, D, K)
        return torch.transpose(output, 1, 3)

    def discrete_output_distribution(self, theta: torch.Tensor, t: torch.Tensor, ema: torch_ema.ExponentialMovingAverage=None) -> torch.Tensor:
        """
        Computes the discrete output distribution of the model.

        Args:
            theta (torch.Tensor)
            t (torch.Tensor): time factor
            ema (torch_ema.ExponentialMovingAverage, optional): EMA used in the model. Defaults to None.

        Returns:
            torch.Tensor
        """
        # Forward pass
        output = self.forward(theta, t, ema=ema)
        # Compute the output probabilities accordingly
        if self.K == 2:
            p0_1 = torch.sigmoid(output)  # (B, D, D, 1)
            p0_2 = 1 - p0_1
            p0 = torch.cat((p0_1, p0_2), dim=-1)  # (B, D, D, 2)
        else:
            p0 = torch.nn.functional.softmax(output, dim=-1)
        return p0

    def process(self, x: torch.Tensor, time: float=None, training: bool=True) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a step of the process.

        Args:
            x (torch.Tensor): input data.
            time (float, optional): time factor in the process. In [0, 1]. Defaults to None.
            training (bool, optional): weither it is during training phase or not. Defaults to True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: if training, only returns the current loss. Otherwise, returns the loss, the generated images, the categorical images and the time factor
        """
        # Step 1: Sample t from U(0, 1) if not provided
        t = torch.tensor(time, device=x.device, dtype=torch.float32)[None] if time else torch.rand((x.size(0),), device=x.device, dtype=torch.float32)
        # Step 2: Calculate Beta
        beta = self.beta * (t ** 2)  # (B,)
        # Step 3: Sample y from N(beta * (K * one_hot(X))
        one_hot_x = F.one_hot(x.permute(0, 2, 3, 1).to(torch.int64), num_classes=self.K).float().squeeze()  # (B, D, D, K)
        mean = beta[:, None, None, None] * (self.K * one_hot_x - 1)
        std = (beta * self.K)[:, None, None, None].sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        # Step 4: Compute the Theta
        theta = F.softmax(y, dim=-1)
        # Step 5: Calculate the output distribution
        p_0 = self.discrete_output_distribution(theta, t)  # (B, D, D, K)
        e_x = one_hot_x
        e_hat = p_0  # (B, D, D, K)
        L_infinity = self.K * self.beta * t[:, None, None, None] * ((e_x - e_hat) ** 2)
        if training:
          return L_infinity.mean()
        else:
           k = torch.distributions.Categorical(probs=p_0).sample()
           return L_infinity.mean(), y, k, t

    @torch.inference_mode()
    def sample(self, batch_size: int=128, nb_steps: int=10, ema: torch_ema.ExponentialMovingAverage=None, device: str='cpu') -> torch.Tensor:
        self.eval()
        # get prior
        theta = torch.ones((batch_size, self.D, self.D, self.K), device=device) / self.K
        t = torch.zeros((theta.shape[0]), device=theta.device, dtype=theta.dtype)
        for i in tqdm(range(1, nb_steps+1)):
            t = (i-1) / nb_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)
            k_probs = self.discrete_output_distribution(theta, t, ema=ema)  # (B, D, D, K)
            k = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D, D)
            alpha = self.beta * (2 * i - 1) / (nb_steps ** 2)
            e_k = F.one_hot(k, num_classes=self.K).float()  # (B, D, D, K)
            mean = alpha * (self.K * e_k - 1)
            var = (alpha * self.K)
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            y = mean + std * eps  # (B, D, D, K)
            theta_prime = torch.exp(y) * theta
            theta = theta_prime / theta_prime.sum(-1, keepdim=True)
        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t), ema=ema)
        k_final = torch.distributions.Categorical(probs=k_probs_final).sample()
        return k_final
