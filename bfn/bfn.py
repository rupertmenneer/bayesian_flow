import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianFlowNetwork(nn.Module):
    """
    Bayesian Flow Network class.

    Args:
        model (nn.Module): The underlying model.
        k (int): The number of classes.

    Attributes:
        k (int): The number of classes.
        model (nn.Module): The underlying model.
    """

    def __init__(self, model, k=2, beta=3.0):
        super(BayesianFlowNetwork, self).__init__()
        assert (k >= 2)
        self.k = k
        self.d = 2
        self.model = model
        # !!! note this is different from githubs selection !!! in the paper they say sqrt(beta(1)) = 3.0 for K=2?
        self.beta_one = torch.tensor(beta**2)

    def forward(self, theta, t):
        theta = (theta * 2) - 1  # scaled in [-1, 1]
        theta = theta.view(theta.shape[0], -1)  # (B, D * K)
        input = torch.cat((theta, t), dim=-1)
        return self.model(input)

    # algorithm 182 - unclear what to set the value for beta(1)?
    def accuracy_schedule(self, t):
        return (t**2 * self.beta_one).unsqueeze(1)

    # algorithm 8 line 1 
    def sample_t_uniformly(self, n):
        return torch.rand(n).unsqueeze(1)
    
    def get_time_at_t(self, t, bs):
        return torch.tensor(t).repeat(bs).unsqueeze(1)

    # algorithm 8 line 3 sample from sender distribution
    def get_sender_distribution_sample(self, beta, e_x):
        # mean is the noise scaled ground truth one hot vector - also scaled by the number of classes K
        mean = beta * ((self.k * e_x) - 1) 
        # std is a function of the noise and number of classes, covariance is identity (e.g. independence assumption)
        # !!! interestingly we don't need the idenity here - I guess its simply modeled as independent vectors? not sure when you would need the identity
        std = torch.sqrt(beta * self.k)
        # sample random noise so we can sample from the sender
        eps = torch.randn(e_x.shape[1])
        return mean + std * eps

    # algorithm 8 line 4
    def bayesian_update_of_theta(self, y):
        return F.softmax(y, dim=1)

    # their pseudo function in section 6.13
    def discrete_output_distribution(self, theta, t):
        # pass all theta and time into model
        output_distribution = self(theta, t)
        # ensure output is valid probability distribution
        if self.k == 2:
            # is this really necessary? don't softmax and sigmoid give same results
            output_distribution = torch.sigmoid(output_distribution)
            output_k_2 = 1. - output_distribution
            output_distribution= torch.cat((output_distribution, output_k_2), dim=-1)
        else:
            output_distribution = F.softmax(output_distribution, axis=1)
            
        return output_distribution
    
    # since one_hot we can just return exactly? the probability for class K in this case its equivalent
    def estimate_e_hat(self, output_distribution):
        # !!! QUESTION !!! do you use ground truth to select this probability? e.g. if probs for datapoint 1 are [0.1, 0.9] and ground truth is [0, 1], do you use 0.9?
        return output_distribution

    # algorithm 8
    def continuous_time_loss_for_discrete_data(self, e_x):

        batch_size = e_x.shape[0]
        # input data is -> B * D. One hot changes to B * D * K. e.g. [0, 0] -> [[1, 0], [1, 0], and [1, 0] -> [[0, 1], [1, 0]]
        e_x = F.one_hot(e_x, num_classes=self.k).float()

        # alg line 1 t -> B
        t = self.sample_t_uniformly(batch_size)
        # alg line 2 beta -> B
        beta = self.accuracy_schedule(t)

        # y, theta, output_distribution, e_hat all have same shape -> B * D * K
        # alg line 3
        y = self.get_sender_distribution_sample(beta, e_x)
        # alg line 4
        theta = self.bayesian_update_of_theta(y)
        # alg line 5
        output_distribution = self.discrete_output_distribution(theta, t)
        # alg line 6 ehat (select the probability you predicted class K)
        e_hat = self.estimate_e_hat(output_distribution)
        # alg line 7 L infinity
        return torch.mean(self.k * self.beta_one * t[:, None] * (e_x - e_hat)**2)
    
    # algorithm 9
    def sample_generation_for_discrete_data(self, d=2, bs=64, n_steps=20):

        # initialise prior with uniform distribution
        prior = torch.ones(bs, d, self.k) / self.k

        # iterate over n_steps
        for i in range(1, n_steps+1):

            # time is set to fraction from 
            t = self.get_time_at_t((i-1)/n_steps, bs=bs)

            output_distribution = self.discrete_output_distribution(prior, t)
            # sample from output distribution to get 'prediction' of true class label
            e_k = F.one_hot(torch.distributions.Categorical(probs=output_distribution).sample()).float()

            # create alpha from beta(1) and t
            alpha = torch.tensor(self.beta_one * (((2*i)-1) / (n_steps**2)))
            # convert our prediction back to 'sender' distribution
            eps = torch.randn_like(e_k)
            y_mean = alpha * (self.k * e_k - 1)
            y_std = torch.sqrt(alpha * self.k)
            y = y_mean + (y_std * eps)

            # bayesian update our sample 
            theta_prime = torch.exp(y) * prior

            # convert back to a probability distribution
            prior = theta_prime / torch.sum(theta_prime, dim=-1, keepdim=True)
            # print(prior[0])

        # one final pass of our distribution through the model to get final predictive distribution
        output_distribution = self.discrete_output_distribution(prior, torch.ones_like(t))
        prediction = torch.distributions.Categorical(probs=output_distribution).sample()

        return prediction

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))