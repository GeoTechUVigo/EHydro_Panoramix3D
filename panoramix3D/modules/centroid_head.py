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
    """
    This class predicts centroid scores for instance detection and performs
    non-maximum suppression to identify confident instance centers (peaks).
    It uses a feature decoder to generate per-voxel centroid scores, then
    applies 3D sparse max pooling to suppress non-maximal responses and
    thresholds the results to extract the most confident peaks.

    Args:
        decoder_blocks: A list of tuples defining the decoder architecture.
            Each tuple contains (dim_in, dim_out, kernel_size, stride).
        instance_density: Prior density of instances in the data, used to
            initialize the final layer bias to log(density / (1 - density)).
        score_thres: Minimum confidence threshold for considering a voxel
            as a potential centroid candidate during peak detection.
        centroid_thres: Final confidence threshold for accepting a peak
            as a valid centroid after non-maximum suppression.
        max_trees_per_scene: Maximum number of instances to detect per scene
            (applied per batch element). Enforced via top-k selection.

    Example:
        >>> centroid_head = CentroidHead(
        ...     decoder_blocks=[(64, 32, 3, 1), (32, 1, 1, 1)],
        ...     instance_density=0.01,
        ...     score_thres=0.1,
        ...     centroid_thres=0.2,
        ...     max_trees_per_scene=64
        ... )
        >>> centroid_scores, peak_indices, confidences = centroid_head(feats, mask)
        >>> print(f"Found {len(peak_indices)} peaks")
        Found 12 peaks
    """

    def __init__(self,
            decoder_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]],
            instance_density: float = 0.01,
            score_thres: float = 0.1,
            centroid_thres: float = 0.2,
            max_trees_per_scene: int = 64
        ):
        super().__init__()
        self.decoder = FeatDecoder(decoder_blocks, 1, aux_dim=0, bias=True, init_bias=math.log(instance_density / (1 - instance_density)))

        self.act = nn.Sigmoid()
        self.max_pool = SparseMaxPool(3, kernel_size=3, stride=1, padding=1, subm=True)
        
        self._score_threshold = score_thres
        self._centroid_threshold = centroid_thres
        self._max_trees_per_scene = max_trees_per_scene
    
    @torch.no_grad()
    def _find_centroid_peaks(self, centroid_scores: SparseTensor) -> Tuple[Tensor, SparseTensor]:
        """
        Find peaks in centroid scores using 3D sparse non-maximum suppression.
        
        This method converts the SparseTensor to SpConv format, applies 3x3x3
        max pooling, and identifies voxels where the original score equals
        the pooled maximum (indicating local peaks). Results are filtered by
        confidence threshold and limited by max_trees_per_scene.

        Args:
            centroid_scores: A SparseTensor with shape [N_voxels, 1] containing
                confidence scores per voxel. Coordinates (.C) encode batch and
                spatial indices.

        Returns:
            A tuple (peak_indices, centroid_confidences) where:
                - peak_indices: 1-D Long tensor of indices into the original
                  centroid_scores tensors, identifying which voxels are peaks.
                - centroid_confidences: SparseTensor containing only the peak
                  coordinates and their confidence scores.

        Notes:
            - Returns empty tensors if no peaks above score_threshold exist.
            - Applies top-k selection if more peaks than max_trees_per_scene.
            - Peak detection is performed independently per batch element.
        """

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

        hmax = self.max_pool(centroid_scores_spconv)
        peak_mask = (hmax.features[:, 0] == centroid_scores_spconv.features[:, 0]) & (centroid_scores_spconv.features[:, 0] > self._centroid_threshold)
        if peak_mask.sum() == 0:
            empty_coords = centroid_scores.C.new_empty(0, centroid_scores.C.size(1))
            return torch.empty(0, dtype=torch.int64, device=empty_coords.device), SparseTensor(coords=empty_coords, feats=centroid_scores.F.new_empty(0, 1))

        mask_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        peak_coords = centroid_scores_spconv.indices[peak_mask]
        peak_scores = centroid_scores_spconv.features[peak_mask]

        max_peaks = self._max_trees_per_scene * (int(centroid_scores.C[:,0].max().item()) + 1)
        if peak_scores.size(0) > max_peaks:
            peak_scores, peak_indices = torch.topk(peak_scores, k=max_peaks, dim=0)
            peak_indices = peak_indices.squeeze(1)
            peak_coords = peak_coords[peak_indices]
            peak_indices = mask_indices[peak_mask][peak_indices]
        else:
            peak_indices = mask_indices[peak_mask]

        return peak_indices, SparseTensor(coords=peak_coords, feats=peak_scores)

    def forward(self, feats: List[SparseTensor], mask: Tensor) -> Tuple[SparseTensor, Tensor, SparseTensor]:
        """
        Forward pass for centroid detection: decode features to scores and find peaks.

        Args:
            feats: List of SparseTensors from backbone/encoder, typically at
                different resolution levels for multi-scale feature decoding.
            mask: Boolean tensor [N_voxels] indicating which voxels should be
                considered for centroid prediction (e.g., non-ground points).

        Returns:
            A tuple (centroid_scores, peak_indices, centroid_confidences) where:
                - centroid_scores: SparseTensor [N_masked_voxels, 1] with
                  predicted confidence scores (post-sigmoid) for each valid voxel.
                - peak_indices: 1-D Long tensor indexing into centroid_scores,
                  identifying which voxels are detected as instance centers.
                - centroid_confidences: SparseTensor containing coordinates and
                  confidence scores of detected peaks only.

        Notes:
            - The decoder is applied only to masked voxels for efficiency.
            - Peak detection uses GT labels if provided, predictions otherwise.
            - Output indices reference the masked coordinate space.
        """

        centroid_scores = self.decoder(feats, mask=mask)
        centroid_scores.F = self.act(centroid_scores.F)

        peak_indices, centroid_confidences = self._find_centroid_peaks(centroid_scores)
        return centroid_scores, peak_indices, centroid_confidences
