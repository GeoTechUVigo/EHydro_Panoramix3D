import torch
import torch.nn.functional as F

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim):
        super().__init__()
        self.voxel_descriptor = spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True)
        self.center_descriptor = spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True)
        self.background_descriptor = nn.Parameter(torch.empty(descriptor_dim), requires_grad=True)

        nn.init.normal_(self.background_descriptor, mean=0., std=0.02)

    def forward(self, voxel_feats, center_feats, center_confidences):
        voxel_descriptors = self.voxel_descriptor(voxel_feats)
        center_descriptors = self.center_descriptor(center_feats)

        instance_output = voxel_descriptors.F @ cat([
            self.background_descriptor.unsqueeze(0),
            (center_confidences.F * center_descriptors.F)
        ]).T

        return SparseTensor(coords=voxel_descriptors.C, feats=instance_output)
