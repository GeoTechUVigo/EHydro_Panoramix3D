from torch import nn
from torchsparse import nn as SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class OffsetHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            SparseConvBlock(latent_dim, 3, 1),
            # SparseConvBlock(latent_dim // 2, 3, 1)
        )

    def forward(self, feats: SparseTensor) -> SparseTensor:
        return self.conv(feats)
