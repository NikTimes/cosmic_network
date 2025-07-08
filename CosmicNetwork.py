import numpy as np 

import torch 
import torch.nn as nn

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


class CosmicNetwork_v2(nn.Module):
    """
    Flexible MLP for CosmoPower-style emulation
    -------------------------------------------------
    Args
    ----
    in_dim        : # input parameters   (default 2:  w_b , w_cdm)
    out_dim       : # outputs            (default 799: ell=2..800)
    hidden_dim    : width of each hidden layer
    hidden_layers : number of hidden layers (≥1)
    activation    : activation class (default CosmoActivation)
    """
    def __init__(self, in_dim=2, out_dim=799,
                 hidden_dim=128, hidden_layers=3,
                 activation=CosmoActivation):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

