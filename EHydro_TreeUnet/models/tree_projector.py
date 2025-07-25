import torch

from ..modules import VoxelDecoder, DirHead, MagHead, InstanceHead, CentroidHead
from torch import nn
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.resnet import SparseResNet


class TreeProjector(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            channels = [16, 32, 64, 128],
            latent_dim = 512,
            instance_density = 0.01,
            centroid_thres = 0.1,
            peak_radius = 1,
            min_score_for_center = 0.5,
            descriptor_dim = 16
        ):
        super().__init__()
        blocks = [(3, channels[0], 3, 1)]
        for channel in channels[1:]:
            blocks.append((3, channel, 3, 2))

        self.encoder = SparseResNet(
            blocks = blocks,
            in_channels=in_channels
        )

        self.voxel_decoder = VoxelDecoder(channels, latent_dim)
        self.semantic_head = spnn.Conv3d(latent_dim, num_classes, 1, bias=True)
        self.centroid_head = CentroidHead(latent_dim, instance_density=instance_density, tau=centroid_thres, peak_radius=peak_radius, min_score_for_center=min_score_for_center)
        self.instance_head = InstanceHead(latent_dim, descriptor_dim)

    def forward(self, x):
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)
        
        feats_pond = SparseTensor(coords=feats.C, feats=feats.F * (1.0 - semantic_output.F.softmax(dim=-1)[:, 0:1]))
        centroid_score_output, centroid_feats_output, centroid_confidence_output = self.centroid_head(feats_pond)
        instance_output = self.instance_head(feats, centroid_feats_output, centroid_confidence_output)

        return semantic_output, centroid_score_output, centroid_confidence_output, instance_output
    