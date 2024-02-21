import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def sample_from_closed_form_bayesian_update(self, x, gamma):
        mu = gamma*x
        std = gamma*(1-gamma)
        eps = torch.randn_like(x)
        return mu + (std * eps)

    def get_gamma_t(self, t):
        return 1 - self.sigma_one**(2*t)

    def sample_t_uniformly(self, n):
        return torch.rand(n).unsqueeze(1)
    
    def get_time_at_t(self, t, bs):
        return torch.tensor(t).repeat(bs).unsqueeze(1)
    
    # def discretised_cdf(self, mu, sigma, x):
    #     if x < -1:
    #         return 0
    #     if x > 1:
    #         return 1
    #     else:
    #         return 0.5 * (1 + torch.erf((x-mu) / ( sigma*torch.sqrt( torch.tensor(2.0) ) ) ) )
        
    # def vectorised_discretised_cdf(self, mu, sigma, bounds):
    #     # input is mu, sigma -> B x D, and bounds -> K
    #     lower_mask = bounds < -1
    #     upper_mask = bounds > 1

    #     # output is B x D x K
    #     result = torch.zeros(mu.shape[0], mu.shape[1], bounds.shape[0])
    #     result = 0.5 * (1 + torch.erf((bounds - mu.unsqueeze(-1)) / (sigma.unsqueeze(-1) * torch.sqrt(torch.tensor(2.0)))))
    #     # clip result depending on bounds
    #     result = result.masked_fill(lower_mask,  0.)
    #     result = result.masked_fill(upper_mask, 1.)
    #     return result

    def vectorised_cdf(self, mu, sigma, x):

        # ensure shapes align for correct broadcasting
        mu = mu.unsqueeze(-1)  # Shape: [B, D, 1]
        sigma = sigma.unsqueeze(-1)  # Shape: [B, D, 1]
        x = x.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, K]
        assert mu.dim() == sigma.dim() == x.dim()

        cdf_func = 0.5 * (1 + torch.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

        # Apply conditions directly without squeezing, using broadcasting
        lower_mask = x < -1
        upper_mask = x > 1

        # Apply masks
        cdf_func = torch.where(lower_mask, torch.zeros_like(cdf_func), cdf_func)
        cdf_func = torch.where(upper_mask, torch.ones_like(cdf_func), cdf_func)

        return cdf_func
        
    def forward(self, mu, t):
        # concatenate time onto the means
        input = torch.cat((mu, t), dim=-1)
        # run this through the model
        output = self.model(input)
        # split the output into the mean and log variance
        mean, log_var = torch.split(output, 1, dim=-1)
        return mean.squeeze(-1), log_var.squeeze(-1)

    def discretised_output_distribution(self, mu, t, gamma, t_min=1e-6):

        # input is B x D one for each mean

        # run the samples through the model -> B x D x 2
        mu_eps, ln_sigma_eps = self.forward(mu, t)
        var_scale = (1-gamma)/gamma
        # update w.r.t noise predictions
        mu_x = (mu/gamma) - torch.sqrt(var_scale * torch.exp(mu_eps))
        sigma_x = torch.sqrt(var_scale * torch.exp(ln_sigma_eps))
        #  clip output distribution if time is lower than min threshold
        time_out_of_bound_mask = t < t_min
        mu_x = mu_x.masked_fill(time_out_of_bound_mask, 0.)
        sigma_x = sigma_x.masked_fill(time_out_of_bound_mask, 1.)

        # Calculate the lower and upper bounds for all bins -> shape: K,
        lower_bounds = self.get_lower_bin_bound(torch.arange(1, self.k+1))
        upper_bounds = self.get_upper_bin_bound(torch.arange(1, self.k+1))

        # # Calculate the discretised output distribution using vectorized operations
        discretised_cdf_lower = self.vectorised_cdf(mu_x, sigma_x, lower_bounds)
        discretised_cdf_upper = self.vectorised_cdf(mu_x, sigma_x, upper_bounds)
        discretised_output_dist = discretised_cdf_upper - discretised_cdf_lower

        return discretised_output_dist

    def get_lower_bin_bound(self, j):
        return ((2 * (j-1))/self.k) - 1
    
    def get_upper_bin_bound(self, j):
        return ((2 * j)/self.k) - 1

    def get_bin_centers(self):
        k_ = torch.linspace(1, self.k, self.k )
        return ((2 * k_ - 1)/self.k) - 1

    def continuous_time_loss_for_discretised_data(self, discretised_data):

        # Shape-> B*D data is samples from the discretised distribution

        batch_size = discretised_data.shape[0]
        # time, gamma -> B
        t = self.sample_t_uniformly(batch_size)
        gamma = self.get_gamma_t(t)

        # Shape-> B*D from the discretised data, create noisy sender sample from a normal centered around data and known variance
        sender_mu_sample = self.sample_from_closed_form_bayesian_update(discretised_data, gamma=gamma)

        # Shape-> B*D*K bins pass the noisy samples to the model, output a continuous distribution and rediscretise it
        output_distribution = self.discretised_output_distribution(sender_mu_sample, t=t, gamma=gamma)

        k_c = self.get_bin_centers()
        # Shape-> B*D sum out over final distribution - weighted sums
        k_hat = torch.sum(output_distribution*k_c, dim=-1)
        
        loss = torch.mean(-torch.log(self.sigma_one) * self.sigma_one**(-2*t) * (discretised_data - k_hat)**2)

        return loss
    
    # algorithm 9
    def sample_generation_for_discretised_data(self, bs=64, n_steps=20):

        # initialise prior with uniform distribution
        prior_mu = torch.zeros(bs, self.d)
        prior_precision = torch.ones(bs, self.d)

        # iterate over n_steps
        for i in range(1, n_steps+1):

            # SHAPE B time is set to fraction from 
            t = self.get_time_at_t((i-1)/n_steps, bs=bs)

            # SHAPE B
            gamma = self.get_gamma_t(t)

            # B x D x K
            output_distribution = self.discretised_output_distribution(prior_mu, t, gamma=gamma)
            print('output_distribution', output_distribution.shape, 'max and min', output_distribution.max(), output_distribution.min())

            # SHAPE B
            alpha = self.sigma_one**( (-2*i) / n_steps ) * (1 - self.sigma_one**(-2/n_steps))

            # sample from y distribution centered around 'k centers'
            eps = torch.randn(bs).unsqueeze(-1)
            std = torch.zeros_like(eps).fill_(1/alpha)
            k_centers = self.get_bin_centers()
            # SHAPE B x D
            mean = torch.sum(output_distribution*k_centers, dim=-1)
            y_sample = mean + std * eps

            # update our prior precisions and means w.r.t to our new sample
            prior_mu = (prior_precision*prior_mu + alpha*y_sample) / (prior_precision + alpha)
            prior_precision = alpha + prior_precision
    
        # final pass of our distribution through the model to get final predictive distribution
        output_distribution = self.discretised_output_distribution(prior_mu, t, gamma=gamma)
        k_centers = self.get_bin_centers()
        # SHAPE B x D
        output_mean = torch.sum(output_distribution*k_centers, dim=-1)

        return output_mean

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))






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