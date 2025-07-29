import torch
import torch.nn.functional as F

from typing import Tuple

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.pool import SparseMaxPool, SparseAvgPool


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim, tau: float = 0.1):
        super().__init__()
        self.voxel_descriptor = spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True)
        # self.center_descriptor = spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True)

        self.background_descriptor = nn.Parameter(torch.empty(descriptor_dim), requires_grad=True)
        nn.init.normal_(self.background_descriptor, mean=0., std=0.02)

        self.max_pool = SparseMaxPool(3, kernel_size=3, stride=1, padding=1, subm=True)
        self.avg_pool = SparseAvgPool(3, kernel_size=3, stride=1, padding=1, subm=True)

        self._tau = tau

    @torch.no_grad()
    def _find_centroid_peaks(self, voxel_feats: SparseTensor, centroid_confidences: SparseTensor, num_batches: int) -> Tuple[SparseTensor, SparseTensor]:
        mask = (centroid_confidences.F > self._tau).squeeze(1)
        if mask.sum() == 0:
            empty_coords = voxel_feats.C.new_empty(0, voxel_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, voxel_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, 1))
        
        centroid_confidences_spconv = SparseConvTensor(
            features=centroid_confidences.F[mask],
            indices=centroid_confidences.C[mask],
            spatial_shape=tuple((centroid_confidences.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(centroid_confidences.C[:,0].max().item()) + 1
        )

        voxel_feats_spconv = SparseConvTensor(
            features=voxel_feats.F[mask],
            indices=voxel_feats.C[mask],
            spatial_shape=tuple((voxel_feats.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(voxel_feats.C[:,0].max().item()) + 1
        )

        hmax = self.max_pool(centroid_confidences_spconv)
        peak_feats = self.avg_pool(voxel_feats_spconv)

        peak_mask = (hmax.features[:, 0] == centroid_confidences_spconv.features[:, 0])
        if peak_mask.sum() == 0:
            empty_coords = voxel_feats.C.new_empty(0, voxel_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, voxel_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=voxel_feats.F.new_empty(0, 1))

        peak_coords = centroid_confidences_spconv.indices[peak_mask]
        peak_scores = centroid_confidences_spconv.features[peak_mask]
        peak_feats = peak_feats.features[peak_mask]

        max_peaks = num_batches * 128
        if peak_scores.size(0) > max_peaks:
            peak_scores, topk_indices = torch.topk(peak_scores, k=max_peaks, dim=0)
            topk_indices = topk_indices.squeeze(1)
            peak_coords = peak_coords[topk_indices]
            peak_feats = peak_feats[topk_indices]

        return SparseTensor(coords=peak_coords, feats=peak_feats), SparseTensor(coords=peak_coords, feats=peak_scores)

    def forward(self, voxel_feats: SparseTensor, centroid_scores: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
        batches = torch.unique(voxel_descriptors.C[:, 0]).tolist()
        centroid_feats, centroid_confidences = self._find_centroid_peaks(voxel_feats, centroid_scores, batches.size(0))

        voxel_descriptors = self.voxel_descriptor(voxel_feats)
        centroid_descriptors = self.voxel_descriptor(centroid_feats)

        voxel_descriptors.F = F.normalize(voxel_descriptors.F, p=2, dim=1)
        centroid_descriptors.F = F.normalize(centroid_descriptors.F, p=2, dim=1)

        full_centroid_descriptors = torch.zeros((centroid_descriptors.F.size(0) + len(batches), centroid_descriptors.F.size(1)), dtype=centroid_descriptors.F.dtype, device=centroid_descriptors.F.device)
        curr_idx = 0

        for batch in batches:
            batch_mask = centroid_descriptors.C[:, 0] == batch

            full_centroid_descriptors[curr_idx] = self.background_descriptor
            curr_idx += 1

            next_idx = curr_idx + batch_mask.sum().item()
            full_centroid_descriptors[curr_idx:next_idx] = centroid_descriptors.F[batch_mask]
            curr_idx = next_idx

        return centroid_confidences, voxel_descriptors.F, full_centroid_descriptors
