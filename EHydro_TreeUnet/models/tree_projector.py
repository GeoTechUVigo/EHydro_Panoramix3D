import torch

from ..modules import VoxelDecoder, DirHead, MagHead
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
        self.semantic_head = spnn.Conv3d(latent_dim, num_classes, 1, bias=True)
        self.offset_dir_head = DirHead(latent_dim)
        self.offset_mag_head = MagHead(latent_dim)
        # self.instance_head = VoxelCompressor(latent_dim, max_instances)

    def _filter_by_class(self, feats, semantic_output, ignore_cls):
        # with torch.no_grad():
        semantic_labels = torch.argmax(semantic_output.F, dim=1)
        mask = torch.ones_like(semantic_labels, dtype=torch.bool)
        for cls in ignore_cls:
            mask &= (semantic_labels != cls)

        idx  = mask.nonzero(as_tuple=False).squeeze(1)
        sub_feats = SparseTensor(coords=feats.C[idx], feats=feats.F[idx], stride=feats.stride)

        return sub_feats, idx
    
    def _restore_dimensionality(self, feats, sub_feats, idx):
        full_output = torch.zeros(feats.F.size(0), sub_feats.F.size(1), device=sub_feats.F.device, dtype=sub_feats.F.dtype)
        full_output.index_copy_(0, idx, sub_feats.F)

        return SparseTensor(coords=feats.C, feats=full_output)

    def forward(self, x):
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)
        
        # sub_feats, idx = self._filter_by_class(feats, semantic_output, [0])

        offset_dir_output = self.offset_dir_head(feats)
        offset_mag_output = self.offset_mag_head(feats)

        # offset_dir_output = self._restore_dimensionality(feats, offset_dir_output, idx)
        # offset_mag_output = self._restore_dimensionality(feats, offset_mag_output, idx)

        return semantic_output, offset_dir_output, offset_mag_output