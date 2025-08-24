import math

from torch import nn
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class CentroidHead(nn.Module):
    def __init__(self, latent_dim: int, instance_density: float = 0.01):
        super().__init__()
        # self.conv = nn.Sequential(
        #     spnn.Conv3d(latent_dim, latent_dim // 2, 1, bias=True),
        #     spnn.ReLU(True),
        #     spnn.Conv3d(latent_dim // 2, 1, 1, bias=True),
        #     spnn.ReLU(True),
        # )
        self.conv = nn.Sequential(
            SparseConvBlock(latent_dim, 1, 1),
            # SparseConvBlock(latent_dim // 2, 1, 1)
        )

        # self.conv = spnn.Conv3d(latent_dim, 1, 1, bias=True)

        # self.act = nn.Sigmoid()

        # nn.init.constant_(self.conv.bias, val = math.log(instance_density / (1 - instance_density)))
        #nn.init.constant_(self.conv[2].bias, val = math.log(instance_density / (1 - instance_density)))
        #nn.init.kaiming_normal_(self.conv[0][0].kernel, mode='fan_out')

    def forward(self, feats: SparseTensor, semantic_output: SparseTensor) -> SparseTensor:
        # centroid_score = self.conv(feats)
        # centroid_score.F = self.act(centroid_score.F)
        # centroid_score.F = centroid_score.F * (1.0 - semantic_output.F.softmax(dim=-1)[:, 0:1])

        return self.conv(feats)
