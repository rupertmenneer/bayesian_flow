from datasets.cifar_10 import get_cifar10_dataloaders
from datasets.utils import get_image_grid_from_tensor
from typing import Union
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from models.unet_vdm import UNetVDM
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from discretised.bfn_discretised import BayesianFlowNetworkDiscretised
from models.adapters import FourierImageInputAdapter, OutputAdapter
import torch
import wandb
import math
import numpy as np

class DiscretisedBFNTrainer():

    def __init__(self,
                 k:int = 16,
                 device: str = None,
                 bs: int = 64,
                 num_epochs: int = 350,
                 input_height: int = 32,
                 input_channels: int = 3,
                 lr: float = 0.0002,
                 max_lr = 0.0004,
                 betas: tuple = (0.9, 0.99),
                 weight_decay: float = 0.01,
                 sigma_one: float = math.sqrt(0.001),
                 model: Union[None, nn.Module] = None,
                 optimizer: Union[None, torch.optim.Optimizer] = None,
                 dataset: str = 'cifar10',
                 wandb_project_name: str = "bayesian_flow",
                 checkpoint_file: str = None,
                 save_path: str = './bfn_model_checkpoint.pth'):
       
        self.k = k
        self.device = device
        self.input_height = input_height
        self.input_channels = input_channels
        self.best_val_loss = torch.tensor(float('inf'))
        self.step = 0
        self.epoch = 0
        self.num_epochs = num_epochs
        self.save_path = save_path

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # load dataset
        if dataset == 'cifar10':
            self.train_dls, self.val_dls, self.test_dls = get_cifar10_dataloaders(batch_size=bs, num_bins=k)
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
       
        # init model
        if model is None:
            data_adapters = {
                "input_adapter": FourierImageInputAdapter(),
                # model output, rgb channels, 2 model outputs (mean and std)
                "output_adapter": OutputAdapter(160, self.input_channels, 2),
            } 
            self.net = UNetVDM(data_adapters)
        else:
            self.net = model
        # init BFN model
        self.bfn_model = BayesianFlowNetworkDiscretised(self.net, device=self.device, k=k, sigma_one=sigma_one).to(self.device)
        # init optimizer
        if optimizer is None:
            self.optim = AdamW(self.bfn_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        steps_per_epoch = len(self.train_dls) 
        total_steps = self.num_epochs * steps_per_epoch
        self.lr_sched = OneCycleLR(self.optim, max_lr, total_steps=total_steps, pct_start=0.001)

        # load checkpoint if provided
        if checkpoint_file is not None:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file)
            # Load each part
            self.bfn_model.load_state_dict(checkpoint['model'])
            self.lr_sched = checkpoint['lr_sched']
            self.optim.load_state_dict(checkpoint['optim'])
            self.step = checkpoint['step']
            self.epoch = checkpoint['epoch']
            print(f'Loaded pretrained model checkpoint {checkpoint_file}')

        # init ema
        self.ema = ExponentialMovingAverage(self.bfn_model.parameters(), decay=0.9999)

        self.wandb_project_name = wandb_project_name
        if wandb_project_name is not None:
            wandb.init(project=wandb_project_name)


    def train(self,
              num_epochs: int = None,
              validation_interval_epoch: int = 1,
              sampling_interval_step: int = 250,
              clip_grad: float = 2.0,
              n_test_batches: int = 0):
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        for i in range(num_epochs):
            epoch_losses = []

            # run through training batches
            for j, batch in enumerate(self.train_dls):

                print(j)
                
                if n_test_batches != 0:
                    if j > n_test_batches:
                        break

                self.optim.zero_grad()

                # model inference
                loss = self.bfn_model.continuous_time_loss_for_discretised_data(batch.to(self.device))

                loss.backward()
                # clip grads
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.bfn_model.parameters(), max_norm=clip_grad)

                # update steps
                self.optim.step()
                self.ema.update()

                # lr step
                self.lr_sched.step()

                # logging
                if self.wandb_project_name is not None:
                    wandb.log({"batch_train_loss": loss.item(), "lr": self.optim.param_groups[0]['lr']})

                epoch_losses.append(loss.item())

                # Sampling stage
                if self.step % sampling_interval_step == 0:
                    self.sample()

                # for every batch iterate
                self.step += 1

            if self.wandb_project_name is not None:
                wandb.log({"epoch_train_loss": torch.mean(torch.tensor(epoch_losses))})

            # Validation check
            if i % validation_interval_epoch == 0:
                self.validate()

        self.epoch += 1        
            


    @torch.no_grad()
    def validate(self):
        self.bfn_model.eval()
        val_losses = []

        for _, batch in enumerate(self.val_dls):
            loss = self.bfn_model.continuous_time_loss_for_discretised_data(batch.to(self.device))
            val_losses.append(loss.item())

        epoch_val_loss = torch.mean(torch.tensor(val_losses))
        if self.wandb_project_name is not None:
            wandb.log({"validation_loss": epoch_val_loss})

        if epoch_val_loss < self.best_val_loss:
            self.best_val_loss = epoch_val_loss
            self.save_model()


    @torch.no_grad()
    def sample(self, sample_shape = (8, 32, 32, 3)):
        self.bfn_model.eval()
        
        # Generate samples and priors
        with self.ema.average_parameters():
            samples, priors = self.bfn_model.sample_generation_for_discretised_data(sample_shape=sample_shape)
            samples = samples.to(torch.float32)
        
        image_grid = get_image_grid_from_tensor(samples.transpose(1, 3))

        # Convert samples and priors to numpy arrays
        image_grid = image_grid.detach().cpu().numpy()
        image_grid = np.transpose(image_grid, (2, 1, 0))

        # priors_np = priors.detach().numpy()
        
        # Plot histograms
        if self.wandb_project_name is not None:
            images = wandb.Image(image_grid, caption="CIFAR10 - Sampled Images from BFN")
            wandb.log({"image_samples": images})

        
    def save_model(self):
        self.bfn_model.eval()

        checkpoint = { 
        'epoch': self.epoch,
        'step': self.step,
        'model': self.bfn_model.state_dict(),
        'optim': self.optim.state_dict(),
        'lr_sched': self.lr_sched}
        print(self.save_path)
        torch.save(checkpoint, self.save_path)
    