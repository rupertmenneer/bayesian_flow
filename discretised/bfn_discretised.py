import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor
import math
import torch.distributions as dist

CONST_log_range = 20
CONST_log_min = 1e-10
CONST_summary_rescale = 10
CONST_exp_range = 10
CONST_min_std_dev = math.exp(-CONST_exp_range)

class BayesianFlowNetworkDiscretised(nn.Module):
    """
    Bayesian Flow Network class.

    Args:
        model (nn.Module): The underlying model.
        k (int): The number of classes.

    Attributes:
        k (int): The number of classes.
        model (nn.Module): The underlying model.
    """

    def __init__(self, model, k=16, sigma_one=0.02):
        super(BayesianFlowNetworkDiscretised, self).__init__()
        assert (k >= 2)
        self.k = k
        self.d = 1
        self.model = model
        self.sigma_one = torch.tensor(sigma_one)
        # Calculate the lower and upper bounds for all bins -> shape: K,
        self.k_centers = self.get_k_centers()
        self.k_lower = self.k_centers - (1/self.k)
        self.k_upper = self.k_centers + (1/self.k)
    
    def sample_from_closed_form_bayesian_update(self, x, gamma):
        std = gamma*(1-gamma)
        eps = torch.randn_like(x)
        return (x*gamma) + (std * eps)

    def get_gamma_t(self, t):
        return 1 - self.sigma_one.pow(2*t)

    def sample_t_uniformly(self, n):
        return torch.rand((n, 1))
    
    def get_time_at_t(self, t, bs):
        return torch.tensor(t).repeat(bs).unsqueeze(1)
    
    def get_k_centers(self):
        k_ = torch.linspace(1, self.k, self.k )
        return ((2 * k_ - 1)/self.k) - 1
        
    def forward(self, mu, t):
        # Shape-> B*(D+1) concatenate time onto the means
        input = torch.cat((mu, t), dim=-1)
        # run this through the model
        output = self.model(input)
        # Shape-> B*D*2, split the output into the mean and log variance
        mean, log_var = torch.split(output, 1, dim=-1)
        return mean.squeeze(-1), log_var.squeeze(-1)

    def discretised_output_distribution(self, mu, t, gamma, t_min=1e-6):

        # input is B x D one for each mean

        # run the samples through the model -> B x D x 2
        mu_eps, ln_sigma_eps = self.forward(mu, t)
        var_scale = torch.sqrt((1-gamma)/gamma)
        # update w.r.t noise predictions
        mu_x = (mu/gamma) - (var_scale * mu_eps)
        sigma_x = var_scale * safe_exp(ln_sigma_eps)

        #  clip output distribution if time is lower than min threshold
        mu_x = torch.where(t < t_min, torch.zeros_like(mu_x), mu_x)
        sigma_x = torch.where(t < t_min, torch.ones_like(sigma_x), sigma_x)

        # print(mu_x)
        # # Calculate the discretised output distribution using vectorized operations
        normal_dist = dist.Normal(mu_x, sigma_x)
        cdf_values_lower = normal_dist.cdf(self.k_lower)
        # ensure first bin has area 0
        cdf_values_lower = torch.where(self.k_lower<=-1, torch.zeros_like(cdf_values_lower), cdf_values_lower)
        cdf_values_upper = normal_dist.cdf(self.k_upper)
        # ensure last bin has area 1
        cdf_values_upper = torch.where(self.k_upper>=1, torch.ones_like(cdf_values_upper), cdf_values_upper)

        discretised_output_dist = cdf_values_upper - cdf_values_lower

        return discretised_output_dist

    def continuous_time_loss_for_discretised_data(self, discretised_data):

        # Shape-> B*D data is samples from the discretised distribution

        batch_size = discretised_data.shape[0]
        # time, gamma -> B, 1
        t = self.sample_t_uniformly(batch_size)
        gamma = self.get_gamma_t(t)

        # Shape-> B*D from the discretised data, create noisy sender sample from a normal centered around data and known variance
        sender_mu_sample = self.sample_from_closed_form_bayesian_update(discretised_data, gamma=gamma)

        # print('t', t.shape, 'gamma', gamma.shape, 'sender_mu_sample', sender_mu_sample.shape, 'discretised_data', discretised_data.shape)

        # Shape-> B*D*K bins pass the noisy samples to the model, output a continuous distribution and rediscretise it
        output_distribution = self.discretised_output_distribution(sender_mu_sample, t=t, gamma=gamma)

        # print('output_distribution', output_distribution.shape)
        gmm = output_distribution*self.k_centers
        # Shape-> B*D sum out over final distribution - weighted sums
        k_hat = torch.sum(gmm, dim=-1).view(-1, 1)

        # print('k_hat', k_hat.shape)
        
        # Shape-> B*D
        diff = (discretised_data - k_hat).pow(2)
        # print('diff', diff.shape)
        # Shape-> scalar, then B*1, B*D
        loss = -safe_log(self.sigma_one) * self.sigma_one.pow(-2*t) * diff
        loss = torch.mean(loss)

        return loss
    
    # algorithm 9
    def sample_generation_for_discretised_data(self, bs=64, n_steps=20):

        # initialise prior with uniform distribution
        prior_mu = torch.zeros(bs, self.d)
        prior_precision = torch.tensor(1)

        prior_tracker = torch.zeros(bs, self.d, 2, n_steps+1)
        # iterate over n_steps
        for i in range(1, n_steps+1):

            # SHAPE B,1 time is set to fraction from 
            t = self.get_time_at_t((i-1)/n_steps, bs=bs)
            # SHAPE B,1
            # gamma = self.get_gamma_t(t)

            # B x D x K
            output_distribution = self.discretised_output_distribution(prior_mu, t, gamma=(1-self.sigma_one.pow(2*t)))

            # SHAPE scalar
            alpha = self.get_alpha(i, n_steps)

            # sample from y distribution centered around 'k centers'

            # B, 1
            eps = torch.randn(bs)
            # SHAPE B x D
            mean = torch.sum(output_distribution*self.k_centers, dim=-1)
            y_sample = mean + ((1/alpha) * eps)

            # update our prior precisions and means w.r.t to our new sample
            prior_tracker[:, :, 0, i] = prior_mu
            prior_tracker[:, :, 1, i] = prior_precision

            # SHAPE B x D
            # prior_mu = (prior_precision*prior_mu.squeeze() + alpha*y_sample) / (prior_precision + alpha)
            # prior_mu = prior_mu.unsqueeze(1)
            # # shape scalar
            # prior_precision = alpha + prior_precision
            prior_mu, prior_precision = self.update_input_params((prior_mu.squeeze(), prior_precision), y_sample, alpha)
            prior_mu = prior_mu.unsqueeze(1)
    
        # final pass of our distribution through the model to get final predictive distribution
        output_distribution = self.discretised_output_distribution(prior_mu, torch.ones_like(t), gamma=(1-self.sigma_one.pow(2)))
        # SHAPE B x D
        output_mean = torch.sum(output_distribution*self.k_centers, dim=-1)

        return output_mean, prior_tracker
    
    def get_alpha(self, i, n_steps, min_variance: float = 1e-6):
        sigma_1 = math.sqrt(min_variance)
        return (sigma_1 ** (-2 * i / n_steps)) * (1 - sigma_1 ** (2 / n_steps))
    
    def update_input_params(self, input_params, y, alpha):
        input_mean, input_precision = input_params
        new_precision = input_precision + alpha
        new_mean = ((input_precision * input_mean) + (alpha * y)) / new_precision
        # print(y.shape, new_mean.shape, new_precision.shape, input_mean.shape, input_precision.shape)
        return new_mean, new_precision

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def safe_log(data: Tensor):
    return data.clamp(min=CONST_log_min).log()

def safe_exp(data: Tensor):
    return data.clamp(min=-CONST_exp_range, max=CONST_exp_range).exp()





    #  def accuracy_schedule(self, t):
    #     if t == 0:
    #         return torch.tensor(0).unsqueeze(1)
    #     return (self.sigma_one**(-2*t) - 1).unsqueeze(1)
    
    # def accuracy_rate(self, t):
    #     return (2*torch.log(self.sigma_one)) / (self.sigma_one**(2*t))
    
    # def bayesian_update(self, mean_0, precision_0, mean_1, precision_1):
    #     new_precision = precision_0 + precision_1
    #     new_mean = (mean_0 * precision_0 + mean_1 * precision_1) / new_precision
    #     return new_mean, new_precision




    # # algorithm 8 line 3 sample from sender distribution
    # def get_sender_distribution_sample(self, beta, e_x):
    #     # mean is the noise scaled ground truth one hot vector - also scaled by the number of classes K
    #     mean = beta * ((self.k * e_x) - 1) 
    #     # std is a function of the noise and number of classes, covariance is identity (e.g. independence assumption)
    #     # !!! interestingly we don't need the idenity here - I guess its simply modeled as independent vectors? not sure when you would need the identity
    #     std = torch.sqrt(beta * self.k)
    #     # sample random noise so we can sample from the sender
    #     eps = torch.randn(e_x.shape[1])
    #     return mean + std * eps

    # # algorithm 8 line 4
    # def bayesian_update_of_theta(self, y):
    #     return F.softmax(y, dim=1)

    # # their pseudo function in section 6.13
    # def discrete_output_distribution(self, theta, t):
    #     # pass all theta and time into model
    #     output_distribution = self(theta, t)
    #     # ensure output is valid probability distribution
    #     if self.k == 2:
    #         # is this really necessary? don't softmax and sigmoid give same results
    #         output_distribution = torch.sigmoid(output_distribution)
    #         output_k_2 = 1. - output_distribution
    #         output_distribution= torch.cat((output_distribution, output_k_2), dim=-1)
    #     else:
    #         output_distribution = F.softmax(output_distribution, axis=1)
            
    #     return output_distribution
    
    # # since one_hot we can just return exactly? the probability for class K in this case its equivalent
    # def estimate_e_hat(self, output_distribution):
    #     # !!! QUESTION !!! do you use ground truth to select this probability? e.g. if probs for datapoint 1 are [0.1, 0.9] and ground truth is [0, 1], do you use 0.9?
    #     return output_distribution

    # def vectorised_cdf(self, mu, sigma, x):

    #     # ensure shapes align for correct broadcasting
    #     mu = mu.unsqueeze(-1)  # Shape: [B, D, 1]
    #     sigma = sigma.unsqueeze(-1)  # Shape: [B, D, 1]
    #     x = x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, K]
    #     assert mu.dim() == sigma.dim() == x.dim()

    #     cdf_func = 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))
    #     cdf_func = torch.clip(cdf_func, -1, 1)

    #     # Apply conditions directly without squeezing, using broadcasting
    #     # lower_mask = x < -1
    #     # upper_mask = x > 1

    #     # # Apply masks
    #     # cdf_func = torch.where(lower_mask, torch.zeros_like(cdf_func), cdf_func)
    #     # cdf_func = torch.where(upper_mask, torch.ones_like(cdf_func), cdf_func)

    #     return cdf_func