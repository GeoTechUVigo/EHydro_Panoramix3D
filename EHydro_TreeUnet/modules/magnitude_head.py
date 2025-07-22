import torch.nn.functional as F

from torch import nn
from torchsparse import nn as spnn


class MagHead(nn.Module):
    def __init__(self, latent_dim, beta=0.5):
        super().__init__()
        self.conv = spnn.Conv3d(latent_dim, 1, 1, bias=True)
        self.act = nn.Softplus(beta=beta)

    def forward(self, x):
        directions = self.conv(x)
        directions.F = self.act(directions.F)

        return directions
