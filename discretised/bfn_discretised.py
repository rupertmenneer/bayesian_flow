import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as dist
from typing import Tuple

CONST_log_min = 1e-10
CONST_exp_range = 10

class BayesianFlowNetworkDiscretised(nn.Module):
    """
        Bayesian Flow Network for Discretised data. Handles training and sampling!

        Bayesian Flow Network Paper: https://arxiv.org/pdf/2308.07037.pdf

        Args:
            model (nn.Module): The underlying model.
            k (int): The number of classes.
            sigma_one (float): The value of sigma_one (see paper).
            device (str): The device to run the model on.

        Attributes:
            device (str): The device to run the model on.
            k (int): The number of classes.
            d (int): The dimension of the data.
            model (nn.Module): The underlying model.
            sigma_one (torch.Tensor): The value of sigma_one.
            k_centers (torch.Tensor): The value of the center of each bin.
            k_lower (torch.Tensor): The lower bounds for all bins.
            k_upper (torch.Tensor): The upper bounds for all bins.

        Methods:
            get_gamma_t(t): Returns the value of gamma at time t.
            sample_t_uniformly(bs): Samples continuous time from a uniform distribution.
            get_time_at_t(t, bs): Returns specific time at t, broadcasts across batch.
            get_K_centers(): Returns the value of the center of each bin for discretised data.
            forward(mu, t): Forward pass of the model.
            discretised_output_distribution(mu, t, gamma, t_min, min_variance): Calculates the discretised output distribution.
            continuous_time_loss_for_discretised_data(discretised_data, min_loss_variance): Calculates the continuous time loss for discretised data.
            sample_generation_for_discretised_data(bs, n_steps): Generates samples for discretised data.
            get_alpha(i, n_steps): Returns the value of alpha at step i.
            update_prior_distribution(prior_mean, prior_precision, y, alpha): Updates the prior distribution.
            get_normal_sample(mu, std): Samples from a normal distribution.
    """

    def __init__(self, model: nn.Module, k: int =16, sigma_one: float = 0.02, device: str ='cpu') -> None:

        super(BayesianFlowNetworkDiscretised, self).__init__()
        assert (k >= 2)
        self.device = device
        self.k = k
        self.model = model
        self.sigma_one = torch.tensor(sigma_one, device=device)

        # Calculate the lower and upper bounds for all bins -> shape: Tensor[K]
        self.k_centers = self.get_k_centers()
        self.k_lower = self.k_centers - (1/self.k)
        self.k_upper = self.k_centers + (1/self.k)


    def get_gamma_t(self, t) -> Tensor:
        return 1 - self.sigma_one.pow(2*t)

    # samples continuous time from a uniform distribution -> shape: Tensor[bs, 1]
    def sample_t_uniformly(self, bs: int) -> Tensor:
        return torch.rand((bs, 1), device=self.device)

    # returns specific time at t, broadcasts across batch -> shape: Tensor[bs, 1]
    def get_time_at_t(self, t: float, bs: int) -> Tensor:
        return torch.tensor(t, device=self.device).repeat(bs).unsqueeze(1)

    # for discretised data with K bins, returns the value of the center of each bin -> shape: Tensor[K]
    def get_k_centers(self) -> Tensor:
        K_ = torch.linspace(1, self.k, self.k, device=self.device)
        return ((2 * K_ - 1)/self.k) - 1

    def forward(self, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # Shape -> Tensor[B, D, 2], Tensor [B] run current estimate of mu and the time through the learnable model
        output = self.model(mu, t)
        # print('Model output:', output.shape)
        # Shape -> Tensor[B, D, 2], split the output into the mean and log variance
        mean, log_var = torch.split(output, 1, dim=-1)
        return mean.squeeze(-1), log_var.squeeze(-1)

    
    def discretised_output_distribution(self, mu: Tensor, t: Tensor, gamma: Tensor, t_min=1e-6) -> Tensor:
        """
        Calculates a discretised output distribution based on the given parameters. 
        E.g. given a normal distribution and K bins, this function will return the probability of each bin.
        Where values of the normal distribution fall outside the -1 and 1 range, the probability is clipped
        and added to the first and last bin (see paper equation 108).

        Args:
            mu (torch.Tensor): The mean values of the input distribution. Shape: [B, D]
            t (torch.Tensor): The time values. Shape: [B]
            gamma (float): The gamma value used for scaling.
            t_min (float, optional): The minimum threshold for time. Defaults to 1e-6.
            min_variance (float, optional): The minimum variance value. Defaults to 1e-6.

        Returns:
            torch.Tensor: The discretised output distribution. Shape: [B, D]
        """

        # run the samples through the model to get prediction of mean and log(sigma) of noise -> Tensor[B, D, 2]
        # print('Model inputs:', mu.shape, t.shape)
        mu_eps, ln_sigma_eps = self.forward(mu, t)

        # update prediction of data w.r.t noise predictions
        var_scale = torch.sqrt((1-gamma)/gamma)
        # print("Mu", mu.shape, 'gamma', gamma.shape, 'var_scale', var_scale.shape, 'mu_eps', mu_eps.shape, 'ln_sigma_eps', ln_sigma_eps.shape)
        mu_x = (mu/gamma) - (var_scale * mu_eps)
        sigma_x = torch.clamp(var_scale * safe_exp(ln_sigma_eps), self.sigma_one)

        # clip output distribution if time is lower than min threshold
        mu_x = torch.where(t < t_min, torch.zeros_like(mu_x), mu_x)
        sigma_x = torch.where(t < t_min, torch.ones_like(sigma_x), sigma_x)

        # Calculate the discretised output distribution using vectorized operations
        # print("Mu_x", mu_x.shape, 'sigma_x', sigma_x.shape, 'k_lower', self.k_lower.shape, 'k_upper', self.k_upper.shape)
        normal_dist = dist.Normal(mu_x, sigma_x)
        broadcasted_k_lower = self.k_lower.repeat(mu_x.shape[1], mu_x.shape[0], 1).transpose(0, 2)
        broadcast_k_upper = self.k_upper.repeat(mu_x.shape[1], mu_x.shape[0], 1).transpose(0, 2)
        cdf_values_lower = normal_dist.cdf(broadcasted_k_lower)
        cdf_values_upper = normal_dist.cdf(broadcast_k_upper)

        # make sure the lower cdf is bounded at 0 and the upper cdf at 1, this has the effect of clipping the distribution (see eq 108) and ensures the total sums to 1
        cdf_values_lower = torch.where(broadcasted_k_lower<=-1, torch.zeros_like(cdf_values_lower), cdf_values_lower)
        cdf_values_upper = torch.where(broadcast_k_upper>=1, torch.ones_like(cdf_values_upper), cdf_values_upper)

        # calculate area in each bin
        discretised_output_dist = (cdf_values_upper - cdf_values_lower).permute(1, 2, 0)

        # print(discretised_output_dist.shape, 'discretised_output_dist')

        return discretised_output_dist

    def continuous_time_loss_for_discretised_data(self, discretised_data: Tensor) -> Tensor:
        """
        Calculates the continuous time loss for discretised data. A.k.a Loss infinity from algorithm 5 page 26.

        1) sample time uniformly
        2) set gamma to be 1-sigma_one^2*t
        3) create an estimate for the mean of the sender sample (this is noisy) µ ~ N(gamma*x, gamma(1-gamma))
        4) pass the noisy samples to the model, output a continuous distribution and rediscretise it
        5) calculate KL based loss between the discretised output distribution and the original discretised data

        Args:
            discretised_data (Tensor): Real data samples from the discretised distribution. Shape: [B, D]

        Returns:
            Tensor: Loss 'Infinity' value.
        """

        # Shape-> Tensor[B, D] data is samples from the discretised distribution

        batch_size = discretised_data.shape[0]
        # flatten data, this makes data that may be 2d such as images Tensor[B, H, W] to flat Tensor[B, H*W]
        discretised_data = discretised_data.view(batch_size, -1)


        # time, gamma -> Tensor[B, 1]
        t = right_pad_dims_to(self.sample_t_uniformly(batch_size), discretised_data)
        gamma = right_pad_dims_to(self.get_gamma_t(t), discretised_data)

        # Shape-> Tensor[B, D] from the discretised data, create noisy sender sample from a normal centered around data and known variance
        std = torch.sqrt(gamma*(1-gamma))
        mean = discretised_data*gamma
        sender_mu_sample = self.get_normal_sample(mean, std)

        # Shape-> Tensor[B, D, K] bins pass the noisy samples to the model, output a continuous distribution and rediscretise it
        output_distribution = self.discretised_output_distribution(sender_mu_sample, t=t, gamma=gamma)

        gmm = output_distribution*self.k_centers

        # Shape-> Tensor[B, D] sum out over final distribution - weighted sums
        K_hat = torch.sum(gmm, dim=-1).view(batch_size, -1)


        # Shape-> Tensor[B, D]
        diff = (discretised_data - K_hat).pow(2)
        # loss infinity algorithm 5
        loss = -safe_log(self.sigma_one) * self.sigma_one.pow(-2*t) * diff
        loss = torch.mean(loss)

        return loss

    @torch.inference_mode()
    def sample_generation_for_discretised_data(self, sample_shape:tuple = (8, 32, 32, 3), n_steps:int = 50):
        """
        Generates new discretised samples using the Bayesian flow model.
        Sample generation algorithm 6 page 26.

        Args:
            bs (int): Batch size. Default is 64.
            n_steps (int): Number of steps to iterate over. Default is 50.

        Returns:
            output_mean (torch.Tensor): Final predictive distribution mean. Shape: [B, D]
            prior_tracker (torch.Tensor): Prior distribution tracker over time. Shape: [B, D, 2, n_steps+1]
        """

        bs = sample_shape[0]

        # initialise prior with a standard normal distribution ~ N(0, 1)
        prior_mu = torch.zeros(sample_shape, device=self.device).view(bs, -1)
        prior_precision = torch.tensor(1, device=self.device)
        d = prior_mu.shape[1]

        # tracks prior distribution over time (mean and var)
        prior_tracker = torch.zeros(bs, d, 2, n_steps+1)

        # iterate over n_steps
        for i in range(1, n_steps+1):

            # Tensor[B, 1] time is a linear step from 0 to 1, a fraction based off current time
            t = right_pad_dims_to(self.get_time_at_t((i-1)/n_steps, bs=bs), prior_mu)
            gamma = right_pad_dims_to(self.get_gamma_t(t), prior_mu)

            # Tensor[B, D, K]
            output_distribution = self.discretised_output_distribution(prior_mu, t, gamma=gamma)

            # Tensor [1]
            alpha = self.get_alpha(i, n_steps)

            # sample from y distribution centered around 'K centers'
            # Tensor[B, D]
            mean = torch.sum(output_distribution*self.k_centers, dim=-1)
            # Tensor[1]
            std = torch.sqrt(1/alpha)
            # Tensor[B, D]
            y_sample = self.get_normal_sample(mean, std)

            # Logging: track the prior precisions and means
            prior_tracker[:, :, 0, i] = y_sample
            prior_tracker[:, :, 1, i] = prior_precision

            # Tensor[B, D] - updated prior distribution based off new estimates
            prior_mu, prior_precision = self.update_prior_distribution(prior_mu, prior_precision, y_sample, alpha)

        # Tensor[B, D, K] final pass of our distribution through the model to get final predictive distribution
        t = right_pad_dims_to(torch.ones_like(t), prior_mu)
        gamma = right_pad_dims_to(self.get_gamma_t(t), prior_mu)
        output_distribution = self.discretised_output_distribution(prior_mu, t, gamma=gamma)
        # Tensor[B, D] 
        output_mean = torch.sum(output_distribution*self.k_centers, dim=-1)
        # final reshape back into requested sample shape
        output_mean = output_mean.view(sample_shape)

        return output_mean, prior_tracker

    # alpha parameter used during sample generation - algorithm 6
    def get_alpha(self, i, n_steps) -> float:
        return (self.sigma_one ** (-2 * i / n_steps)) * (1 - self.sigma_one ** (2 / n_steps))

    # update the priors based off new estimate - algorithm 6
    def update_prior_distribution(self, prior_mean: Tensor, prior_precision: Tensor, y: Tensor, alpha: float) -> Tuple[Tensor, Tensor]:
        new_precision = prior_precision + alpha
        new_mean = ((prior_precision * prior_mean.squeeze()) + (alpha * y)) / new_precision
        return new_mean, new_precision
    
    # given a mean and std, samples from a normal distribution with random noise of shape mean
    def get_normal_sample(self, mu: Tensor, std: Tensor) -> Tensor:
        eps = torch.randn_like(mu).to(self.device)
        return mu + (std * eps)

def right_pad_dims_to(input_tensor, target_shape):
    padding_dims = target_shape.ndim - input_tensor.ndim
    if padding_dims <= 0:
        return input_tensor
    return input_tensor.view(*input_tensor.shape, *((1,) * padding_dims))

def safe_log(data: Tensor):
    return data.clamp(min=CONST_log_min).log()

def safe_exp(data: Tensor):
    return data.clamp(min=-CONST_exp_range, max=CONST_exp_range).exp()
