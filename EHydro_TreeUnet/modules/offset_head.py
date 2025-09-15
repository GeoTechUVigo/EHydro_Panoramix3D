import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import List, Tuple, Union

from torch import nn, Tensor, cat
from torchsparse import SparseTensor

from . import FeatDecoder


class OffsetHead(nn.Module):
    def __init__(self, 
            decoder_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]],
            descriptor_dim: int = 16
        ):
        super().__init__()

        self.decoder = FeatDecoder(decoder_blocks, 3, bias=True)
        self.descriptor = FeatDecoder(decoder_blocks, descriptor_dim, bias=True)

    @torch.no_grad()
    def _revoxelize(self, offsets: SparseTensor) -> Tuple[SparseTensor, Tensor]:
        offsets_ = cat([
            torch.zeros(offsets.F.size(0), 1, device=offsets.F.device, dtype=torch.int32),
            (offsets.F.sign() * offsets.F.abs().expm1()).round().to(torch.int32)
        ], dim=1)

        new_coords = offsets.C + offsets_

        pc_hash        = spf.sphash(new_coords)
        voxel_hash     = torch.unique(pc_hash)
        idx_query      = spf.sphashquery(pc_hash, voxel_hash)
        idx_query_long = idx_query.to(torch.int64)
        counts         = spf.spcount(idx_query, voxel_hash.numel())
        out_coords     = spf.spvoxelize(new_coords.float(), idx_query, counts).int()

        counts         = counts.index_select(0, idx_query_long).unsqueeze(1)
        out_scores     = counts.log1p().to(offsets.F.dtype)

        return SparseTensor(coords=out_coords, feats=out_scores), idx_query_long

    def forward(self, feats: SparseTensor, mask: Tensor, offset_labels: SparseTensor = None) -> SparseTensor:
        offsets = self.decoder(feats, mask=mask)
        centroid_scores, inv_map = self._revoxelize(offsets if offset_labels is None else offset_labels)

        cluster_descriptors = self.descriptor(feats, mask=mask, new_coords=centroid_scores.C, reduce=inv_map)
        cluster_descriptors.F = F.normalize(cluster_descriptors.F, p=2, dim=1)
        
        return offsets, cluster_descriptors, centroid_scores, inv_map
