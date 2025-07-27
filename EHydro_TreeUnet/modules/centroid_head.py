import math

from torch import nn
from torchsparse import nn as spnn, SparseTensor


class CentroidHead(nn.Module):
    def __init__(self, latent_dim: int, instance_density: float = 0.01):
        super().__init__()
        self.conv = spnn.Conv3d(latent_dim, 1, 1, bias=True)
        self.act = nn.Sigmoid()

        nn.init.constant_(self.conv.bias, val = math.log(instance_density / (1 - instance_density)))
        # nn.init.kaiming_normal_(self.conv.kernel, mode='fan_out')

    def forward(self, feats: SparseTensor, semantic_output: SparseTensor) -> SparseTensor:
        centroid_score = self.conv(feats)
        centroid_score.F = self.act(centroid_score.F)
        # centroid_score.F = centroid_score.F * (1.0 - semantic_output.F.softmax(dim=-1)[:, 0:1])

        return centroid_score
