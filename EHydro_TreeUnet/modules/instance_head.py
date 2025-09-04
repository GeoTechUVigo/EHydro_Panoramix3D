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
        self.background_descriptor = nn.Parameter(torch.empty(1, descriptor_dim), requires_grad=True)
        self.voxel_descriptor = nn.Sequential(
            spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True),
            spnn.ReLU(True),
            # spnn.Conv3d(latent_dim // 2, descriptor_dim, 1),
            # spnn.ReLU(True),
        )

        nn.init.normal_(self.background_descriptor, mean=0., std=0.02)

    def forward(self, cluster_feats: SparseTensor, centroid_feats: SparseTensor, centroid_confidences: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
        cluster_descriptors = self.voxel_descriptor(cluster_feats)
        if centroid_feats.F.size(0) == 0:
            centroid_descriptors = self.background_descriptor
        else:
            centroid_descriptors = cat([self.background_descriptor, centroid_confidences.F * self.voxel_descriptor(centroid_feats).F], dim=0)

        instance_output = cluster_descriptors.F @ centroid_descriptors.T

        return SparseTensor(coords=cluster_feats.C, feats=instance_output)
