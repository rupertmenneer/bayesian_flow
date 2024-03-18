# Bayesian Flow Network - Math Cheat Sheet

## Distribution Glossary


### Prior Distribution a.k.a. Input Distribution
This is the prior over data. Each dimension is considered independent. The input distribution the same dimensionality as the data. One point of confusion is they use $\theta$ to describe these prior parameters, these aren't related to the neural network which commonly has $\theta$ assigned for its learnable weights.

$$p_{\text{input}}/p_{\text{prior}} = p_I(\mathbf{x} | \theta) = \prod_{d=1}^{D} p_I(x^{(d)} | \theta^{(d)}).$$

Note! in training we never use the prior, the point of BFNs is that we instead recieve an 'updated' prior, therefore our 'input distribution' is always the prior + updated with a noisy sample and known variance.

Gaussian for discretised/continuous. Categorical for discrete.

<!-- $$ p_{\text{prior}}(\mathbf{x}_{\text{data}} | \theta_{\text{prior}} )  = \prod_{d=1}^{\text{Dims}} p_{\text{prior}}(x_{\text{data}}^{(d)} | \theta_{\text{prior}}^{(d)}).$$ -->

### Noisy Data a.k.a. Sender Distribution
In BFNs data is considered to be a distribution with a known variance (or accuracy/precision) over time. When the variance is infinite, there is no information about the original data. When variance is 0, a delta function centered on the data is recovered. Information is progressively recovered with a variance (accuracy) scheduler.

$$p_{\text{sender}}/p_{\text{noisy data}} = p_S(y | x; \alpha) = \prod_{d=1}^{D} p_S(y^{(d)} | x^{(d)}; \alpha)$$


### Neural Network Output a.k.a Output Distribution
A neural network takes as input the updated prior and the time (which informs the network about the accuracy/precision level)

$$p_{\text{output}}/p_{\text{Data expectation}} = p_O(\mathbf{x} | \theta, t) = \prod_{d=1}^{D} p_O(x^{(d)} | \Psi^{(d)}(\theta, t)).
$$

### Data Expectation a.k.a. Receiver Distribution 

$$ p_R(y | \theta; t, \alpha) = \frac{\mathbb{E}}{p_O(x'|\theta;t)} p_S(y | x'; \alpha). $$

### Bayesian Update
$$p_U(\theta' | \theta, x; \alpha) = \mathbb{E}_{p_S(y|x;\alpha)} \left[ \delta(\theta' - h(\theta, y, \alpha)) \right],$$


### Bayesian Flow
$$p_F(\theta | x; t) = p_U(\theta | \theta_0, x; \beta(t)).$$


## Discretised Data

Discretised data is data considered to be truly continuous, but must fit within K bins e.g. an image captures continuous light spectrum but is binned into 3x256 RGB values.

### Accuracy scheduler

Accuracy scheduler $\beta(t)$ for discretised and continuous data is derived with the requirement that the expected entropy of the input distribution linearly decreases with t. Intuitively, this means that information flows into the input distribution at a constant rate.
The level of measurement noise informs the choice of $\sigma_1$ which informs accuracy schedule $\beta(t)$, hence the choice of $\sigma_1$ is informed by the width of the discretisation bins, as these place a natural limit on how precisely the data needs to be transmitted. E.g. for 8-bit data with 256 bins and hence a bin width of 1/128, setting  $\sigma_1$ = 1e−3.

 $$ \beta(t) = 1 - \sigma_1^{2t}$$


### Discretised Data Loss - Continuous Time

$$
L^{\infty}(x) = -\ln \sigma_{1} \sigma_{1}^{-2t} \left\| x - \hat{k}(\theta, t) \right\|^2
$$

the scaling term $\ln \sigma_1 \sigma_1^{-2t}$ adds bigger penalty towards the end of the run, as accuracy is scheduled to increase with $t$ (i.e. the noise level decreases), hence the bigger penalty for an easier task.

'''

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


'''

### Discretised Data Loss - Discrete Time

Discrete time loss

$$
\text{5.4 DISCRETE-TIME LOSS } L^n(\mathbf{x}) \\
\text{From Eqs. 86 and 113,} \\
$$
$$
D_{KL}(p_s(\cdot | \mathbf{x}, \alpha_i) || p_R(\cdot | \theta_{i-1}, t_{i-1}, \alpha_i)) \tag{116} \\
$$

$$
= D_{KL} \left( \mathcal{N}(\mathbf{x}, \alpha_i^{-1}I) \bigg|\bigg| \prod_{d=1}^{D} \prod_{k=1}^{K} p_O^{(d)}(k | \theta_{i-1}, t_{i-1})\mathcal{N}(k_c, \alpha_i^{-1}) \right), \tag{117} \\
$$

$$
\text{which cannot be calculated in closed form, but can be estimated with Monte-Carlo sampling.} \\
\text{Substituting into Eq. 24,} \\$$

$$
L^n(\mathbf{x}) = n \mathbb{E}_{t, p_f, \mathcal{N}(y|\mathbf{x}, \alpha_i^{-1}I)} \ln (\mathcal{N}(y|\mathbf{x}, \alpha_i^{-1}I) )\\
- \sum_{d=1}^{D} \ln (\sum_{k=1}^{K}  \left( p_O^{(d)}(k | \theta, t_{i-1})\mathcal{N}(y^{(d)} | k_c, \alpha_i^{-1})) \right). \tag{119}$$

'''

        # Sender dist - use Monte Carlo sampling
        y_sender_distribution = dist.Normal(discretised_data, torch.sqrt(1/alpha))
        y_sender_samples = y_sender_distribution.sample(torch.Size([monte_carlo_samples]))

        # Receiver distribution - GMM
        receiver_mix_dist = dist.Categorical(probs=output_distribution)
        receiver_components = dist.Normal(self.k_centers, torch.sqrt(1/alpha).unsqueeze(-1))
        receiver_dist = dist.MixtureSameFamily(receiver_mix_dist, receiver_components)
        
        # Calculating the loss KL between the sender and receiver
        log_prob_y_sender = y_sender_distribution.log_prob(y_sender_samples)
        log_prob_y_receiver = receiver_dist.log_prob(y_sender_samples)
        loss = n * torch.mean(log_prob_y_sender - log_prob_y_receiver)

'''

### Discretised Output Distribution
$$\mu_x = 
\begin{cases} 
0 & \text{if } t < t_{\text{min}}, \\
\mu - \frac{1-\gamma(t)}{\gamma(t)} \mu_e & \text{otherwise},
\end{cases} \tag{105}$$

$$
\sigma_x = 
\begin{cases} 
1 & \text{if } t < t_{\text{min}}, \\
\sqrt{\frac{1-\gamma(t)}{\gamma(t)}} \exp(\ln \sigma_e) & \text{otherwise}.
\end{cases} \tag{106}$$

$$
\text{For each } d \in \{1, D\}, \text{ define the following univariate Gaussian cdf}
$$

$$
F(x | \mu_x^{(d)}, \sigma_x^{(d)}) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x - \mu_x^{(d)}}{\sigma_x^{(d)} \sqrt{2}}\right)\right], \tag{107}$$

$$
\text{and clip at } [-1,1] \text{ to obtain}

G(x | \mu_x^{(d)}, \sigma_x^{(d)}) = 
\begin{cases} 
0 & \text{if } x \leq -1, \\
1 & \text{if } x \geq 1, \\
F(x | \mu_x^{(d)}, \sigma_x^{(d)}) & \text{otherwise}.
\end{cases} \tag{108}
$$

$$
\text{Then, for } k \in \{1, K\},

p_O^{(d)}(k | \theta; t) \triangleq G(k | \mu_x^{(d)}, \sigma_x^{(d)}) - G(k-1 | \mu_x^{(d)}, \sigma_x^{(d)}), \tag{109}
$$

'''

        # run the samples through the model to get prediction of mean and log(sigma) of noise -> Tensor[B, D, 2]
        mu_eps, ln_sigma_eps = self.forward(mu, t)

        # update prediction of data w.r.t noise predictions
        var_scale = torch.sqrt((1-gamma)/gamma)
        mu_x = (mu/gamma) - (var_scale * mu_eps)
        sigma_x = torch.clamp(var_scale * safe_exp(ln_sigma_eps), self.sigma_one)

        # clip output distribution if time is lower than min threshold
        mu_x = torch.where(t < t_min, torch.zeros_like(mu_x), mu_x)
        sigma_x = torch.where(t < t_min, torch.ones_like(sigma_x), sigma_x)

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


'''



## Discrete Data

### Accuracy scheduler
The guiding heuristic for accuracy scheduler β(t) is to decrease the expected entropy of the input distribution linearly with t.
$$\beta(t) =  t^2 \beta(1), \tag{182}$$
$\alpha(t)$ is related to $\beta(t)$ in the following way:

$$\alpha(t) = \frac{d\beta(t)}{dt} = \beta(1)2t, \tag{183}$$

$\beta(1)$ is determined empirically for each experiment.

![accuracy](./images/discrete_accuracy_scheduler.png )

**Figure 9: Accuracy schedule vs. expected entropy for discrete data.** The surface plot shows the expectation over the parameter distribution $p(θ | x; β)$ of the entropy of the categorical input distribution $p(x | θ)$ for K = 2 to 30 and $\sqrtβ$ = 0.01 to 3. The red and cyan lines highlight the entropy curves for 2 and 27 classes, the two values that occur in the original paper experiments. The red and cyan stars show the corresponding values we chose for p$\sqrtβ(1)$. (Graves et. al. 2023, p.34)


### Discrete time loss

Discrete-time loss:  

$$
L^n(\mathbf{x}) = n \mathbf{E}_{t\sim U\{1,n\},P(\mathbf{\theta}|\mathbf{x},t_{i-1}), \mathcal{N}\left(\mathbf{y}|\alpha_{i}(K \mathbf{e_x} - 1),\alpha_{i}K\mathbf{I}\right) } \left[ \mathcal{N}\left(\mathbf{y}|\alpha_{i}(K \mathbf{e_x} - 1),\alpha_{i}K\mathbf{I}\right) \right] \tag{189}
$$

$$
= -\sum_{d=1}^{D} \ln \left( \sum_{k=1}^{K} p_{o}^{(d)}(k \mid \theta; t_{i-1}) \mathcal{N}\left(y^{(d)} \mid \alpha_{i}(K\mathbf{e}_k - 1), \alpha_{i} K \mathbf{I} \right) \right),         \tag{190}
$$

where  
$K$ is the number of classes,
$t$ is time
$\mathbf{\theta}$ are parameters of data distribution (i.e. Categorical)
$\mathbf{y}$ is the noisy sample from sender
$\mathbf{e_x}$ is the one-hot vector of data



### Continuous time loss


$$
L^{\infty}(\mathbf{x}) = K\beta(1) \mathbb{E}_{t\sim U(0,1),P_F(\mathbf{\theta}|\mathbf{x},t)} \left[ t\left\| \mathbf{e_x} - \mathbf{\hat{e}}(\mathbf{\theta}, t) \right\|^2 \right] \tag{205}
$$


###  Training algorithm
**Algorithm 8 Continuous-Time Loss $L^\infty(\mathbf{x})$ for Discrete Data**

**Require**: \( \beta(1) \in \mathbb{R}^+ \), number of classes \( K \in \mathbb{N} \)  
**Input**: discrete data \( x \in \{1, K\}^D \)  

$ \tau \sim U(0, 1) $
- \( \beta \leftarrow \beta(1)\tau^2 \)
- \( y \sim \mathcal{N} (\beta (Ke_x - 1) , \beta KI) \)
- \( \theta \leftarrow \text{softmax}(y) \)
- \( p_o( \cdot \mid \theta; t) \leftarrow \text{DISCRETE\_OUTPUT\_DISTRIBUTION}(\theta, t) \)  -- output distribution
- \( \hat{e}(\theta, t) \leftarrow \left( \sum_k p_o^{(1)}(k \mid \theta; t)e_k, ..., \sum_k p_o^{(D)}(k \mid \theta; t)e_k \right) \) -- data expectation
- \(e_x = \text{one\_hot}(x, \text{num\_classes}=K) \)
- \( L^∞(x) \leftarrow K\beta(1)t \left\|e_x - \hat{e}(\theta, t)\right\|^2 \)
----------------------------------------------------------------------
**function** DISCRETE_OUTPUT_DISTRIBUTION(θ ∈ [0, 1]<sup>KD</sup>, t ∈ [0, 1])  
**Input** (θ, t) to network, receive Ψ(θ, t) as output  
**for** d ∈ {1, D} **do**  
  **if** k = 2 **then**  
   $ p_o^{(d)}(1 | \theta; t) \leftarrow \sigma(\Psi^{(d)}(\theta, t))$   
   $ p_o^{(d)}(2 | \theta; t) \leftarrow 1 - p_o^{(d)}(1 | \theta; t) $  
  **else**  
   \( p_o^{(d)}( \cdot | \theta; t) \leftarrow \text{softmax}(\Psi^{(d)}(\theta, t)) \)  
  **end if**  
**end for**  
**Return** \( p_o( \cdot | \theta; t) \)  
**end function**  
