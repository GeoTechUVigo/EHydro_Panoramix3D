import torch
import torchsparse.nn.functional as spf

from torch import nn, cat
from torch_scatter import scatter_sum
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class VoxelCompressor(nn.Module):
    def __init__(self, latent_dim, max_instances):
        super().__init__()

        self.mixer = nn.Sequential(
            SparseConvBlock(latent_dim, latent_dim, kernel_size=3),
            SparseConvBlock(latent_dim, latent_dim, kernel_size=3),
            SparseConvBlock(latent_dim, latent_dim, kernel_size=1)
        )

        self.instance_head = spnn.Conv3d(latent_dim, max_instances, 1, bias=False)

    def forward(self, feats, offsets):
        offsets_ = torch.cat([
            torch.zeros(feats.F.size(0), 1, device=offsets.F.device, dtype=torch.int32),
            offsets.F.to(torch.int32)
        ], dim=1)

        new_coords = feats.C + offsets_

        pc_hash      = spf.sphash(new_coords)
        voxel_hash   = torch.unique(pc_hash)
        idx_query    = spf.sphashquery(pc_hash, voxel_hash)
        counts       = spf.spcount(idx_query, voxel_hash.numel())
        out_coords   = spf.spvoxelize(new_coords.float(), idx_query, counts).int()
        out_feats    = scatter_sum(feats.F, idx_query.to(torch.int64), dim=0)

        cluster_feats = SparseTensor(
            feats=out_feats,
            coords=out_coords,
            stride=feats.stride
        )

        instance_output = self.instance_head(self.mixer(cluster_feats))
        instance_output.C = feats.C
        instance_output.F = instance_output.F[idx_query.to(torch.long)]

        return instance_output
