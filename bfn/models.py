import torch
import torch.nn as nn

class SimpleNeuralNetworkDiscretised(nn.Module):

    def __init__(self, d = 1, hidden_dim = 8):
        super(SimpleNeuralNetworkDiscretised, self).__init__()

        # Define the number of output classes based on K
        self.d = d

        # feed forward layers 
        self.layers = nn.Sequential(
            # input is number of class vectors K with dimensionality D, +1 dim for time
            nn.Linear(d+1, hidden_dim),
            nn.GELU(),
            # output is number of class vectors K or 1 in case of binary task with dimensionality D
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, input):

        """
        Forward pass of simple neural net. Takes input tensor, vecotorizes and concatenates with time, then passes through feed forward layers.
        
        Parameters
        ----------
        theta : torch.Tensor
            Tensor of shape (B, D,).
        t : torch.Tensor
            Tensor of shape (B,).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, D, 2).
        """
        output = self.layers(input)  # (B, D)
        output = output.view(output.shape[0], self.d, 2) # (B, D, 2)
        return output

class SimpleNeuralNetworkDiscrete(nn.Module):

    def __init__(self, k = 2, d = 2, hidden_dim = 64):
        super(SimpleNeuralNetworkDiscrete, self).__init__()

        # Define the number of output classes based on K
        output_classes = k if k > 2 else 1
        self.d = d
        self.k = k

        # feed forward layers 
        self.layers = nn.Sequential(
            # input is number of class vectors K with dimensionality D, +1 dim for time
            nn.Linear((k*d)+1, hidden_dim),
            nn.GELU(),
            # output is number of class vectors K or 1 in case of binary task with dimensionality D
            nn.Linear(hidden_dim, (output_classes*d))
        )

    def forward(self, input):

        """
        Forward pass of simple neural net. Takes input tensor, vecotorizes and concatenates with time, then passes through feed forward layers.
        
        Parameters
        ----------
        theta : torch.Tensor
            Tensor of shape (B, D, K).
        t : torch.Tensor
            Tensor of shape (B,).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, D, K).
        """
        output = self.layers(input)  # (B, D * K)
        output = output.view(output.shape[0], self.d, -1) # (B, d, output_classes)
        return output