import torch
import torchsparse.nn.functional as spf

from typing import Tuple

from torch import nn
from torch_scatter import scatter_sum
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock


class OffsetHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # self.conv = spnn.Conv3d(latent_dim, 3, 1, bias=True)
        self.conv = nn.Sequential(
            SparseConvBlock(latent_dim, latent_dim // 2, 3),
            SparseConvBlock(latent_dim // 2, latent_dim // 4, 3),
            spnn.Conv3d(latent_dim // 4, 3, 1, bias=True)
        )

    # @torch.no_grad()
    def _revoxelize(self, voxel_feats: SparseTensor, offsets: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        offsets_ = torch.cat([
            torch.zeros(voxel_feats.F.size(0), 1, device=offsets.F.device, dtype=torch.int32),
            (offsets.F.sign() * offsets.F.abs().expm1()).to(torch.int32)
        ], dim=1)

        new_coords = voxel_feats.C + offsets_

        pc_hash        = spf.sphash(new_coords)
        voxel_hash     = torch.unique(pc_hash)
        idx_query      = spf.sphashquery(pc_hash, voxel_hash)
        idx_query_long = idx_query.to(torch.int64)
        counts         = spf.spcount(idx_query, voxel_hash.numel())
        out_coords     = spf.spvoxelize(new_coords.float(), idx_query, counts).int()
        counts         = counts.index_select(0, idx_query_long).unsqueeze(1)
        out_feats      = scatter_sum(voxel_feats.F / counts, idx_query_long, dim=0)

        out_scores     = counts.log1p().to(out_feats.dtype)
        return SparseTensor(coords=out_coords, feats=out_feats), SparseTensor(coords=out_coords, feats=out_scores), idx_query_long

    def forward(self, feats: SparseTensor, offset_labels: SparseTensor = None) -> SparseTensor:
        offsets = self.conv(feats)
        cluster_feats, centroid_scores, inv_map = self._revoxelize(feats, offsets if offset_labels is None else offset_labels)
        return offsets, cluster_feats, centroid_scores, inv_map
