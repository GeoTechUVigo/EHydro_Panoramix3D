from ..modules import VoxelDecoder
from torch import nn
from torchsparse import nn as spnn
from torchsparse.backbones.resnet import SparseResNet


class TreeProjector(nn.Module):
    def __init__(self, in_channels, num_classes, max_instances, channels = [16, 32, 64, 128], latent_dim = 512):
        super().__init__()
        blocks = [(3, channels[0], 3, 1)]
        for channel in channels[1:]:
            blocks.append((3, channel, 3, 2))

        self.encoder = SparseResNet(
            blocks = blocks,
            in_channels=in_channels
        )

        self.voxel_decoder = VoxelDecoder(channels, latent_dim)
        self.semantic_head = spnn.Conv3d(latent_dim, num_classes, 1, bias=False)
        self.instance_head = spnn.Conv3d(latent_dim, max_instances, 1, bias=False)

    def forward(self, x):
        feats = self.voxel_decoder(self.encoder(x))

        return self.semantic_head(feats), self.instance_head(feats)
