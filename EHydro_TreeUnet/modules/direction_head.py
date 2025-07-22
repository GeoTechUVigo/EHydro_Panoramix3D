import torch.nn.functional as F

from torch import nn
from torchsparse import nn as spnn


class DirHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = spnn.Conv3d(latent_dim, 3, 1, bias=True)

    def forward(self, x):
        directions = self.conv(x)
        directions.F = F.normalize(directions.F, dim=1)

        return directions
