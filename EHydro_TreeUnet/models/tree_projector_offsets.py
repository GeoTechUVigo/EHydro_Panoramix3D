import torch

from ..modules import VoxelDecoder, DirHead, MagHead, InstanceHead
from torch import nn
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.resnet import SparseResNet


class TreeProjector(nn.Module):
    def __init__(self, in_channels, num_classes, descriptor_dim, channels = [16, 32, 64, 128], latent_dim = 512):
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
        self.instance_head = InstanceHead(latent_dim, descriptor_dim)

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

        votes = feats.C + (offset_dir_output.F * torch.expm1(offset_mag_output.F))
        min_corner = torch.floor(votes.min(axis=0)).to(torch.int)
        grid_coords = votes - min_corner
        grid_idx = torch.round(grid_coords).to(torch.int)

        shape = grid_idx.max(axis=0) + 1
        accum = torch.zeros(shape, dtype=torch.float32)

        for v in grid_idx:
            accum[tuple(v)] += 1.0

        sigma = 1.0
        thr_frac = 0.1

        accum = gaussian_filter(accum, sigma=sigma, mode="constant")
        neigh = maximum_filter(accum, size=3, mode="constant")
        peaks = (accum == neigh) & (accum > thr_frac * accum.max())

        centers = torch.argwhere(peaks) + min_corner  # (K,3) en voxeles originales

        instance_output = self.instance_head(feats, centroids)

        # offset_dir_output = self._restore_dimensionality(feats, offset_dir_output, idx)
        # offset_mag_output = self._restore_dimensionality(feats, offset_mag_output, idx)

        return semantic_output, offset_dir_output, offset_mag_output
    