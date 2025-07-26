import torch
import torchsparse

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from torch_cluster import radius_graph
from torch_scatter import scatter_max, scatter_mean


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim, tau: float = 0.1, peak_radius: int = 1, min_score_for_center: int = 0.5):
        super().__init__()
        self.voxel_descriptor = spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True)
        self.center_descriptor = spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True)
        self.background_descriptor = nn.Parameter(torch.empty(descriptor_dim), requires_grad=True)

        nn.init.normal_(self.background_descriptor, mean=0., std=0.02)

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

    def forward(self, voxel_feats, centroid_scores):
        centroid_feats, centroid_confidences = self._find_centroid_peaks(voxel_feats, centroid_scores)
        
        voxel_descriptors = self.voxel_descriptor(voxel_feats)
        center_descriptors = self.center_descriptor(centroid_feats)

        instance_output = voxel_descriptors.F @ cat([
            self.background_descriptor.unsqueeze(0),
            (centroid_confidences.F * center_descriptors.F)
        ]).T

        return centroid_confidences, SparseTensor(coords=voxel_descriptors.C, feats=instance_output)
