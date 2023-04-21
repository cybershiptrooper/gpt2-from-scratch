from . import *
from utils.config import Activation
import numpy as np

class GeluNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (
            0.5 * x * ( 1.0 + torch.tanh( 
                np.sqrt(2.0 / np.pi) *
                (x + 0.044715 * torch.pow(x, 3.0)) 
            ) )
        )

class FeedForward(nn.Module):
    def __init__(self, d_model, d_mlp, init_range=0.02, activation=Activation.relu, verbose = False):
        super().__init__()
        activation_layer = nn.ReLU
        if(activation==Activation.gelu):activation_layer = GeluNew

        self.net = torch.nn.Sequential(
            nn.Linear(d_model, d_mlp),
            activation_layer(),
            nn.Linear(d_mlp, d_model)
        )
        for i in self.net:
            if isinstance(i, nn.Linear):
                nn.init.normal_(i.weight, mean=0, std=init_range)
                nn.init.normal_(i.bias, mean=0, std=init_range)
        
        self.verbose = verbose

    def forward(self, x):
        return x+self.net(x)