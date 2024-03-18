# Bayesian Flow Networks

This repo is a simple replication of the discrete and discretised implementations of Bayesian flow networks (BFNs).

## CIFAR 10 Sampling

Data Expectation Distribution (Output distribution)

<img src="gifs/seed_113_data_expectation.gif" alt="CIFAR 10 data expectation over sampling" style="width:600px;"/>

Updated Prior (Input distribution)

<img src="gifs/seed_113_updated_prior.gif" alt="CIFAR 10 updated prior over sampling" style="width:600px;"/>

## Discretised Examples

### Distributions over time

<img src="gifs/figure_8_discretised.gif" alt="Different distributions over time during training example" style="width:600px;"/>

### Updated Prior & Data Expectation Trajectories and Probability Flow

<img src="gifs/updated_prior_trajectories.gif" alt="Discretised 5 bin example, probability flow and example trajectories over time for updated prior." style="width:600px;"/>

<img src="gifs/data_expectation_trajectories.gif" alt="Discretised 5 bin example, probability flow and example trajectories over time for data expectation." style="width:600px;"/>


### Sampling with different steps
![sampling_bfn](https://github.com/rupertmenneer/bayesian_flow/assets/71332436/925f03f8-9584-4e7c-a33b-228569523498)

### Un-conditional inpainting (using something similar to repaint r=20)
![inpainting_bfn](https://github.com/rupertmenneer/bayesian_flow/assets/71332436/ce91efd2-3ab8-4964-8540-19a6f9f8e8f7)

### Toy examples (discretised) with 5 bins
![sample plots](https://github.com/rupertmenneer/bayesian_flow/assets/71332436/823f4924-56ad-451a-b4c1-2fae1f290906)


# How does BFN work?

We create a 'flow' that uses bayesian updates to transform from a prior to a data sample

![Bayesian update frame](https://github.com/rupertmenneer/bayesian_flow/assets/71332436/59609d53-7ee3-44dc-b730-a25be77cf310)

During training we learn the flow, from our prior using noisy samples from our data 

![Training frame](https://github.com/rupertmenneer/bayesian_flow/assets/71332436/1e4ed9a0-6071-4f86-9f59-a1e711474382)

At sampling time we simply swap out our sender distribution with our receiver distribution

![Sampling frame](https://github.com/rupertmenneer/bayesian_flow/assets/71332436/13959781-1106-4cf1-8c86-2127788c522c)


### This Repo

- Replicates BFN with simple toy examples for discrete and discretised datasets.
- Train a discretised model on the CIFAR-10 dataset (see results above).
- Provides some math breakdown in the_math.md

The original paper can be found here: https://arxiv.org/pdf/2308.07037.pdf

The official code implementation here: https://github.com/nnaisense/bayesian-flow-networks

This repo was part of a paper replication project at the University of Cambridge 2024.
