from torch import nn
from torchsparse import nn as spnn, SparseTensor


class OffsetHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = spnn.Conv3d(latent_dim, 3, 1, bias=True)

    def forward(self, feats: SparseTensor) -> SparseTensor:
        return self.conv(feats)
