import torch

from ..modules import VoxelDecoder, VoxelCompressor
from torch import nn
from torchsparse import nn as spnn, SparseTensor
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
        self.offset_head = spnn.Conv3d(latent_dim, 3, 1, bias=False)
        self.instance_head = VoxelCompressor(latent_dim, max_instances)

    def forward(self, x):
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)
        
        # with torch.no_grad():
        semantic_labels = torch.argmax(semantic_output.F, dim=1)
        mask = (semantic_labels != 0)

        idx  = mask.nonzero(as_tuple=False).squeeze(1)
        sub_feats = SparseTensor(coords=feats.C[idx], feats=feats.F[idx], stride=feats.stride)

        offset_output = self.offset_head(sub_feats)
        instance_output = self.instance_head(sub_feats, offset_output)
        
        full_offset_output = torch.zeros(feats.F.size(0), offset_output.F.size(1), device=feats.F.device, dtype=offset_output.F.dtype)
        full_offset_output.index_copy_(0, idx, offset_output.F)

        full_instance_output = torch.zeros(feats.F.size(0), instance_output.F.size(1), device=feats.F.device, dtype=instance_output.F.dtype)
        full_instance_output.index_copy_(0, idx, instance_output.F)

        offset_output.C = feats.C
        offset_output.F = full_offset_output

        instance_output.C = feats.C
        instance_output.F = full_instance_output

        return semantic_output, instance_output, offset_output
