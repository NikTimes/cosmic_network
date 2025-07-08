import numpy as np 

import torch 
from torch.nn import nn

import torch.nn.functional as F


class CosmoActivation(nn.Module):
    """
    f(x) = [γ + σ(-β x)*(1-γ)] * x
    where σ is the logistic sigmoid.
    γ and β are learnable scalars (one per layer).
    """
    def __init__(self, init_beta: float = 1.0, init_gamma: float = 0.5):
        super().__init__()
        # register as parameters so they are trained with the rest of the net
        self.beta  = nn.Parameter(torch.tensor(init_beta))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

    def forward(self, x):
        # logistic sigmoid
        sigm = torch.sigmoid(-self.beta * x)
        pref = self.gamma + sigm * (1.0 - self.gamma)
        return pref * x


class CosmicNetwork(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 799,
                 hidden: int = 128, activation=CosmoActivation):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            activation(),          
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.model(x)

