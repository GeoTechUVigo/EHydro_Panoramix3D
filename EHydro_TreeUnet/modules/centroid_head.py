import math
import torch
import torch.nn.functional as F

import torchsparse.nn.functional as spF

from torch import nn
from torchsparse import nn as spnn, SparseTensor
from torch_cluster import radius_graph
from torch_scatter import scatter_max, scatter_mean


class CentroidHead(nn.Module):
    def __init__(self, latent_dim: int, instance_density: float = 0.01, tau: float = 0.1, peak_radius: int = 1, min_score_for_center: int = 0.5):
        super().__init__()
        self.conv = spnn.Conv3d(latent_dim, 1, 1, bias=True)
        self.act = nn.Sigmoid()

        nn.init.constant_(self.conv.bias, val = math.log(instance_density / (1 - instance_density)))
        # nn.init.kaiming_normal_(self.conv.kernel, mode='fan_out')

        self._tau = tau
        self._peak_radius = peak_radius
        self._min_score_for_center = min_score_for_center
        self._max_neigh = ((peak_radius * 2) + 1) ** 3

    @torch.no_grad()
    def _find_centroid_peaks(self, voxel_feats: SparseTensor, centroid_scores: SparseTensor):
        mask = (centroid_scores.F > self._tau).squeeze(1)
        if mask.sum() == 0:
            empty_coords = voxel_feats.C.new_empty(0, voxel_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, voxel_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, 1))
        
        coords = centroid_scores.C[mask]
        feats = voxel_feats.F[mask]
        scores = centroid_scores.F[mask].squeeze(1)

        edge = radius_graph(
            x=coords[:,1:].float(),
            r=self._peak_radius + 0.1,
            batch=coords[:,0].long(),
            loop=True,
            max_num_neighbors=self._max_neigh
        )

        neigh_max, _ = scatter_max(
            src=scores[edge[1]],
            index=edge[0],
            dim=0,
            dim_size=scores.size(0)
        )

        peak_mask = (scores >= neigh_max - 1e-6) & (scores >= self._min_score_for_center)
        if peak_mask.sum() == 0:
            empty_coords = voxel_feats.C.new_empty(0, voxel_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, voxel_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, 1))

        peak_idx = torch.nonzero(peak_mask).squeeze(1)
        peak_coords = coords[peak_idx]
        peak_scores = scores[peak_idx]
        
        if peak_scores.size(0) > 256:
            topk_scores, topk_indices = torch.topk(peak_scores, k=256, dim=0)
            peak_idx = peak_idx[topk_indices.squeeze()]
            scores = topk_scores[:, None]
        else:
            scores = peak_scores[:, None]

        owner = torch.full((coords.size(0),), -1, device=coords.device, dtype=torch.long)
        owner[peak_idx] = torch.arange(peak_idx.size(0), device=coords.device)

        src, dst = edge
        is_dst_peak = peak_mask[dst]
        owner[src[is_dst_peak]] = owner[dst[is_dst_peak]]

        valid_mask = owner >= 0
        feat_mean  = scatter_mean(
            feats[valid_mask],
            owner[valid_mask],
            dim=0,
            dim_size=peak_idx.size(0)
        )

        return SparseTensor(coords=peak_coords, feats=feat_mean), SparseTensor(coords=peak_coords, feats=scores)

    def forward(self, feats, semantic_output):
        centroid_score = self.conv(feats)
        centroid_score.F = self.act(centroid_score.F)
        centroid_score.F = centroid_score.F * (1.0 - semantic_output.F.softmax(dim=-1)[:, 0:1])
        centroid_feat, centroid_confidence = self._find_centroid_peaks(feats, centroid_score)

        return centroid_score, centroid_feat, centroid_confidence
