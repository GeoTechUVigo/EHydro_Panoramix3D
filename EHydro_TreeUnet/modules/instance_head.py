import torch.nn.functional as F

from torch import nn
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim):
        super().__init__()
        self.voxel_descriptor = SparseConvBlock(latent_dim, descriptor_dim, 1)
        self.center_descriptor = SparseConvBlock(latent_dim, descriptor_dim, 1)

    def forward(self, voxels, centers):
        voxel_descriptors = self.voxel_descriptor(voxels)
        center_descriptors = self.center_descriptor(centers)

        return SparseTensor(coords=voxel_descriptors.C, feats=voxel_descriptors.F @ center_descriptors.F)
