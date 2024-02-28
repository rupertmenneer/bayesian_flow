from datasets.cifar_10 import get_cifar10_dataloaders
from typing import Union
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from models.unet_vdm import UNetVDM
from torch.optim import AdamW
from discretised.bfn_discretised import BayesianFlowNetworkDiscretised
from models.adapters import FourierImageInputAdapter, OutputAdapter
import torch
import wandb
import math

class DiscretisedBFNTrainer():

    def __init__(self,
                 k:int = 16,
                 device: str = 'cpu',
                 bs: int = 32,
                 input_height: int = 32,
                 input_channels: int = 3,
                 lr: float = 0.0002,
                 betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.01,
                 sigma_one: float = math.sqrt(0.001),
                 model: Union[None, nn.Module] = None,
                 optimizer: Union[None, torch.optim.Optimizer] = None,
                 dataset: str = 'cifar10',
                 wandb_project_name: str = "bayesian_flow"):
       
        self.k = k
        self.device = device
        self.input_height = input_height
        self.input_channels = input_channels

        # load dataset
        if dataset == 'cifar10':
            self.train_dls, self.val_dls, self.test_dls = get_cifar10_dataloaders(batch_size=bs, num_bins=k)
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
        # init model
        if model is None:
            data_adapters = {
                "input_adapter": FourierImageInputAdapter(),
                "output_adapter": OutputAdapter(self.input_height, self.input_channels, self.input_height),
            } 
            self.net = UNetVDM(data_adapters)
        else:
            self.net = model

        # init BFN model
        self.bfn_model = BayesianFlowNetworkDiscretised(self.net, device=device, k=k, sigma_one=sigma_one).to(device)

        # init ema
        self.ema = ExponentialMovingAverage(self.bfn_model.parameters(), decay=0.9999)

        # init optimizer
        if optimizer is None:
            self.optim = AdamW(self.bfn_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        if wandb_project_name is not None:
            wandb.init(project=wandb_project_name)


    def train(self, num_epochs: int = 10, validation_interval: int = 1, sampling_interval: int = 5, clip_grad: float = 2.0):
        

        for i in range(num_epochs):
            epoch_losses = []

            # run through training batches
            for _, batch in enumerate(self.train_dls):

                self.optim.zero_grad()
                print('Input shape:', batch.shape)
                # model inference
                loss = self.bfn_model.continuous_time_loss_for_discretised_data(batch.to(self.device))

                loss.backward()
                # clip grads
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.bfn_model.parameters(), max_norm=clip_grad)

                # update steps
                self.optim.step()
                self.ema.update()

                # logging
                if self.wandb_project_name is not None:
                    wandb.log({"batch_train_loss": loss.item()})

                epoch_losses.append(loss.item())

            if self.wandb_project_name is not None:
                wandb.log({"epoch_train_loss": torch.mean(torch.tensor(epoch_losses))})

            # # Validation check
            # if (i + 1) % validation_interval == 0:
            #     self.validate()

            # # Sampling stage
            # if (i + 1) % sampling_interval == 0:
            #     self.sample()



        