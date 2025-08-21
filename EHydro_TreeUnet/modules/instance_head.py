import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import Tuple

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock
from torch_scatter import scatter_sum
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.pool import SparseMaxPool, SparseAvgPool


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim, tau: float = 0.1):
        super().__init__()
        # self.voxel_descriptor = SparseConvBlock(in_channels=latent_dim, out_channels=descriptor_dim, kernel_size=1)
        self.voxel_descriptor = nn.Sequential(
            spnn.Conv3d(latent_dim, descriptor_dim, 1),
            spnn.ReLU(True),
        )

        self.background_descriptor = nn.Parameter(torch.empty(1, descriptor_dim), requires_grad=True)

        self.max_pool = SparseMaxPool(3, kernel_size=3, stride=1, padding=1, subm=True)
        self.avg_pool = SparseAvgPool(3, kernel_size=3, stride=1, padding=1, subm=True)

        nn.init.normal_(self.background_descriptor, mean=0., std=0.02)
        self._tau = tau

    @torch.no_grad()
    def _revoxelize(self, voxel_feats: SparseTensor, offsets: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        offsets_ = torch.cat([
            torch.zeros(voxel_feats.F.size(0), 1, device=offsets.F.device, dtype=torch.int32),
            offsets.F.to(torch.int32)
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

        cluster_feats_spconv = SparseConvTensor(
            features=out_feats,
            indices=out_coords,
            spatial_shape=tuple((out_coords[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(out_coords[:,0].max().item()) + 1
        )

        out_feats = self.avg_pool(cluster_feats_spconv)
        out_feats = out_feats.features.index_select(0, idx_query_long)
        out_scores = counts.to(out_feats.dtype)

        return SparseTensor(coords=voxel_feats.C, feats=out_feats), SparseTensor(coords=voxel_feats.C, feats=out_scores)

    @torch.no_grad()
    def _find_centroid_peaks(self, cluster_feats: SparseTensor, centroid_scores: SparseTensor, cluster_scores: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        mask = (centroid_scores.F > self._tau).squeeze(1)
        if mask.sum() == 0:
            empty_coords = cluster_feats.C.new_empty(0, cluster_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, cluster_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, 1))
        
        centroid_scores_spconv = SparseConvTensor(
            features=centroid_scores.F[mask],
            indices=centroid_scores.C[mask],
            spatial_shape=tuple((centroid_scores.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(centroid_scores.C[:,0].max().item()) + 1
        )

        hmax = self.max_pool(centroid_scores_spconv)

        peak_mask = (hmax.features[:, 0] == centroid_scores_spconv.features[:, 0])
        if peak_mask.sum() == 0:
            empty_coords = cluster_feats.C.new_empty(0, cluster_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, cluster_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, 1))

        peak_coords = centroid_scores_spconv.indices[peak_mask]
        peak_scores = centroid_scores_spconv.features[peak_mask]
        peak_feats = cluster_feats.F[mask][peak_mask]

        if peak_scores.size(0) > 128:
            peak_scores, topk_indices = torch.topk(peak_scores, k=128, dim=0)
            topk_indices = topk_indices.squeeze(1)
            peak_coords = peak_coords[topk_indices]
            peak_feats = peak_feats[topk_indices]

        return SparseTensor(coords=peak_coords, feats=peak_feats), SparseTensor(coords=peak_coords, feats=peak_scores)

    def forward(self, voxel_feats: SparseTensor, centroid_scores: SparseTensor, offsets: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        voxel_descriptors = self.voxel_descriptor(voxel_feats)

        cluster_feats, cluster_scores = self._revoxelize(voxel_feats, offsets)
        centroid_feats, centroid_confidences = self._find_centroid_peaks(cluster_feats, centroid_scores, cluster_scores)
        centroid_descriptors = cat([self.background_descriptor, centroid_confidences.F * self.voxel_descriptor(centroid_feats).F], dim=0)

        instance_output = voxel_descriptors.F @ centroid_descriptors.T

        return centroid_confidences, SparseTensor(coords=voxel_feats.C, feats=instance_output)
