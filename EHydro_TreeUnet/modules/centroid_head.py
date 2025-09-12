import math
import torch
import torchsparse.nn.functional as spf

from typing import Tuple

from torch import nn, Tensor, cat
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.modules import SparseConvBlock
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.pool import SparseMaxPool, SparseAvgPool


class CentroidHead(nn.Module):
    def __init__(self, latent_dim: int, instance_density: float = 0.01, score_thres: float = 0.1, centroid_thres: float = 0.2):
        super().__init__()
        # self.conv = spnn.Conv3d(latent_dim, 1, 1, bias=True)
        self.conv = nn.Sequential(
            SparseConvBlock(latent_dim + 1, latent_dim // 2, 3),
            SparseConvBlock(latent_dim // 2, latent_dim // 4, 3),
            spnn.Conv3d(latent_dim // 4, 1, 1, bias=True)
        )

        self.act = nn.Sigmoid()
        self.max_pool = SparseMaxPool(3, kernel_size=3, stride=1, padding=1, subm=True)
        self.avg_pool = SparseAvgPool(3, kernel_size=3, stride=1, padding=1, subm=True)

        nn.init.constant_(self.conv[2].bias, val = math.log(instance_density / (1 - instance_density)))
        #nn.init.constant_(self.conv[2].bias, val = math.log(instance_density / (1 - instance_density)))
        #nn.init.kaiming_normal_(self.conv[0][0].kernel, mode='fan_out')

        self._score_threshold = score_thres
        self._centroid_threshold = centroid_thres

    def _union_sparse_layers(self, A: SparseTensor, B: SparseTensor) -> SparseTensor:
        hashA = spf.sphash(A.C)
        hashB = spf.sphash(B.C)

        idxA_to_B = spf.sphashquery(hashA, hashB)
        outB = torch.zeros((A.C.size(0), B.F.size(1)), device=A.F.device, dtype=A.F.dtype)
        mask = idxA_to_B >= 0
        if mask.any():
            ai = torch.nonzero(mask, as_tuple=False).squeeze(1)
            bi = idxA_to_B[mask].to(torch.long)
            outB[ai] = B.F[bi]
        
        return SparseTensor(coords=A.C, feats=cat((A.F, outB.to(A.F.dtype)), dim=1))
    
    @torch.no_grad()
    def _find_centroid_peaks(self, cluster_feats: SparseTensor, centroid_scores: SparseTensor, inv_map: Tensor) -> Tuple[SparseTensor, SparseTensor]:
        mask = (centroid_scores.F > self._score_threshold).squeeze(1)
        if mask.sum() == 0:
            empty_coords = cluster_feats.C.new_empty(0, cluster_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, cluster_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, 1))
        
        centroid_scores_spconv = SparseConvTensor(
            features=centroid_scores.F[mask],
            indices=centroid_scores.C[mask],
            spatial_shape=tuple((centroid_scores.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(centroid_scores.C[:,0].max().item()) + 1
        )

        cluster_feats_spconv = SparseConvTensor(
            features=cluster_feats.F,
            indices=cluster_feats.C,
            spatial_shape=tuple((cluster_feats.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(cluster_feats.C[:,0].max().item()) + 1
        )

        cluster_feats_spconv = self.avg_pool(cluster_feats_spconv)
        hmax = self.max_pool(centroid_scores_spconv)

        peak_mask = (hmax.features[:, 0] == centroid_scores_spconv.features[:, 0]) & (centroid_scores_spconv.features[:, 0] > self._centroid_threshold)
        if peak_mask.sum() == 0:
            empty_coords = cluster_feats.C.new_empty(0, cluster_feats.C.size(1))
            return SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, cluster_feats.F.size(1))), SparseTensor(coords=empty_coords, feats=cluster_feats.F.new_empty(0, 1))

        peak_coords = centroid_scores_spconv.indices[peak_mask]
        peak_scores = centroid_scores_spconv.features[peak_mask]
        peak_feats = cluster_feats_spconv.features.index_select(0, inv_map)[mask][peak_mask]

        if peak_scores.size(0) > 128:
            peak_scores, topk_indices = torch.topk(peak_scores, k=128, dim=0)
            topk_indices = topk_indices.squeeze(1)
            peak_coords = peak_coords[topk_indices]
            peak_feats = peak_feats[topk_indices]

        return SparseTensor(coords=peak_coords, feats=peak_feats), SparseTensor(coords=peak_coords, feats=peak_scores)

    def forward(self, voxel_feats: SparseTensor, cluster_feats: SparseTensor, centroid_scores_off: SparseTensor, inv_map: Tensor, centroid_score_labels: SparseTensor = None) -> SparseTensor:
        voxel_feats = self._union_sparse_layers(voxel_feats, centroid_scores_off)
        centroid_scores = self.conv(voxel_feats)
        centroid_scores.F = self.act(centroid_scores.F)

        centroid_feats, centroid_confidences = self._find_centroid_peaks(cluster_feats, centroid_scores if centroid_score_labels is None else centroid_score_labels, inv_map)

        return centroid_scores, centroid_feats, centroid_confidences
