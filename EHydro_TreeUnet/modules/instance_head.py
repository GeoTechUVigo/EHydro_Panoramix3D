import math
import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import Tuple

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim):
        super().__init__()
        # self.background_descriptor = nn.Parameter(torch.empty(1, descriptor_dim), requires_grad=True)
        self.voxel_descriptor = nn.Sequential(
            spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True),
            # spnn.ReLU(True),
            # spnn.Conv3d(latent_dim // 2, descriptor_dim, 1),
            # spnn.ReLU(True),
        )


    def forward(self, cluster_feats: SparseTensor, centroid_feats: SparseTensor, centroid_confidences: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
        cluster_descriptors = self.voxel_descriptor(cluster_feats)
        cluster_descriptors.F = F.normalize(cluster_descriptors.F, p=2, dim=1)

        if centroid_feats.F.size(0) == 0:
            return SparseTensor(coords=cluster_feats.C, feats=torch.empty(cluster_feats.F.size(0), 0, dtype=cluster_descriptors.F.dtype, device=cluster_feats.F.device))
        
        centroid_descriptors = self.voxel_descriptor(centroid_feats)
        centroid_descriptors.F = centroid_confidences.F * F.normalize(centroid_descriptors.F, p=2, dim=1)

        instance_output = cluster_descriptors.F @ centroid_descriptors.F.T

        return SparseTensor(coords=cluster_feats.C, feats=instance_output)
