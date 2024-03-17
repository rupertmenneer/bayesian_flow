# Bayesian Flow Network - Math Cheat Sheet

## Distribution Glossary

### Input Distribution
This is the prior over data. Each dimension is considered independent. The input distribution the same dimensionality as the data. One point of confusion is they use $\theta$ to describe these prior parameters, these aren't related to the neural network which commonly has $\theta$ assigned for its learnable weights.

$$p_I(\mathbf{x} | \theta) = \prod_{d=1}^{D} p_I(x^{(d)} | \theta^{(d)}).$$
