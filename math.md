# Bayesian Flow Network - Math Cheat Sheet

## Distribution Glossary

### Input Distribution
This is the prior over data. Each dimension is considered independent. The input distribution the same dimensionality as the data. One point of confusion is they use $\theta$ to describe these prior parameters, these aren't related to the neural network which commonly has $\theta$ assigned for its learnable weights.

$$p_I(\mathbf{x} | \theta) = \prod_{d=1}^{D} p_I(x^{(d)} | \theta^{(d)}).$$


## Discrete Data

### Discrete time loss


$$
L^n(\mathbf{x}) = n \mathbb{E}_{t\sim U\{1,n\},P(\mathbb{\theta}|\mathbb{x},t_{i-1}), \mathcal{N}\left(\mathbf{y}|\alpha_{i}(K \mathbf{e_x} - 1),\alpha_{i}K\mathbf{I}\right) } \left[ \mathcal{N}\left(\mathbf{y}|\alpha_{i}(K \mathbf{e_x} - 1),\alpha_{i}K\mathbf{I}\right) \right] \tag{189}
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
$
\alpha(t) = \frac{d\beta(t)}{dt} = \beta(1)2t.
$
$β(1)$ is determined empirically for each experiment.



### Continuous time loss


$$
L^{\infty}(x) = K\beta(1) \mathbb{E}_{t\sim U(0,1),P_F(\mathbb{\theta}|\mathbb{x},t)} \left[ t\left\| \mathbb{e_x} - \mathbb{\hat{e}}(\mathbb{\theta}, t) \right\|^2 \right] \tag{205}
$$


