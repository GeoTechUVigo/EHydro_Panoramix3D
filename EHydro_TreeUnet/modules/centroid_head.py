import math
import torch
import torch.nn.functional as F

from typing import List, Tuple, Union

from torch import nn, Tensor
from torchsparse import SparseTensor
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.pool import SparseMaxPool

from . import FeatDecoder


class CentroidHead(nn.Module):
    def __init__(self,
            decoder_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]],
            descriptor_dim: int = 16,
            instance_density: float = 0.01,
            score_thres: float = 0.1,
            centroid_thres: float = 0.2
        ):
        super().__init__()
        self.decoder = FeatDecoder(decoder_blocks, 1, aux_dim=1, bias=True, init_bias=math.log(instance_density / (1 - instance_density)))
        self.descriptor = FeatDecoder(decoder_blocks, descriptor_dim, bias=True)

        self.act = nn.Sigmoid()
        self.max_pool = SparseMaxPool(3, kernel_size=3, stride=1, padding=1, subm=True)
        # self.avg_pool = SparseAvgPool(3, kernel_size=3, stride=1, padding=1, subm=True)

        self._score_threshold = score_thres
        self._centroid_threshold = centroid_thres
    
    @torch.no_grad()
    def _find_centroid_peaks(self, centroid_scores: SparseTensor) -> Tuple[Tensor, SparseTensor]:
        mask = (centroid_scores.F > self._score_threshold).squeeze(1)
        if mask.sum() == 0:
            empty_coords = centroid_scores.C.new_empty(0, centroid_scores.C.size(1))
            return torch.empty(0, dtype=torch.int64, device=empty_coords.device), SparseTensor(coords=empty_coords, feats=centroid_scores.F.new_empty(0, 1))

        centroid_scores_spconv = SparseConvTensor(
            features=centroid_scores.F[mask],
            indices=centroid_scores.C[mask],
            spatial_shape=tuple((centroid_scores.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(centroid_scores.C[:,0].max().item()) + 1
        )

        '''
        cluster_descriptors_spconv = SparseConvTensor(
            features=cluster_descriptors.F,
            indices=cluster_descriptors.C,
            spatial_shape=tuple((cluster_descriptors.C[:, 1:].max(0).values + 1).tolist()),
            batch_size=int(cluster_descriptors.C[:,0].max().item()) + 1
        )

        cluster_descriptors_spconv = self.avg_pool(cluster_descriptors_spconv)
        '''

        hmax = self.max_pool(centroid_scores_spconv)
        peak_mask = (hmax.features[:, 0] == centroid_scores_spconv.features[:, 0]) & (centroid_scores_spconv.features[:, 0] > self._centroid_threshold)
        if peak_mask.sum() == 0:
            empty_coords = centroid_scores.C.new_empty(0, centroid_scores.C.size(1))
            return torch.empty(0, dtype=torch.int64, device=empty_coords.device), SparseTensor(coords=empty_coords, feats=centroid_scores.F.new_empty(0, 1))

        peak_coords = centroid_scores_spconv.indices[peak_mask]
        peak_scores = centroid_scores_spconv.features[peak_mask]
        peak_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)[peak_mask]

        if peak_scores.size(0) > 128:
            peak_scores, peak_indices = torch.topk(peak_scores, k=128, dim=0)
            peak_indices = peak_indices.squeeze(1)
            peak_coords = peak_coords[peak_indices]

        return peak_indices, SparseTensor(coords=peak_coords, feats=peak_scores)

    def forward(self, feats: SparseTensor, centroid_scores_off: SparseTensor, mask: Tensor, centroid_score_labels: SparseTensor = None) -> SparseTensor:
        centroid_scores = self.decoder(feats, aux=centroid_scores_off, mask=mask)
        centroid_scores.F = self.act(centroid_scores.F)

        peak_indices, centroid_confidences = self._find_centroid_peaks(centroid_scores if centroid_score_labels is None else centroid_score_labels)
        centroid_descriptors = self.descriptor(feats, mask=mask.nonzero(as_tuple=False).squeeze(1)[peak_indices])
        centroid_descriptors.F = centroid_confidences.F * F.normalize(centroid_descriptors.F, p=2, dim=1)

        return centroid_scores, centroid_descriptors, centroid_confidences
