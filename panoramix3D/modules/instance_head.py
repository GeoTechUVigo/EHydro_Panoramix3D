import math
import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import List, Tuple, Union

from torch import nn, Tensor, cat
from torchsparse import nn as spnn, SparseTensor
from torch_scatter import scatter_mean

from ..modules import FeatDecoder


class InstanceHead(nn.Module):
    """
    This class computes per-voxel instance assignment logits by learning to associate
    each voxel with detected centroid instances. It builds voxel and centroid descriptors,
    then uses a 2D convolutional fusion network to predict the affinity between each
    voxel-centroid pair, incorporating both descriptor similarity and spatial distances.

    The architecture follows a two-stage approach:
    1. Learn per-voxel descriptors using a feature decoder and cluster aggregation
    2. For each batch, construct a 2D "interaction map" between voxels and centroids,
       incorporating descriptors and distance features, then predict assignment logits

    Args:
        resnet_blocks: List of tuples defining the decoder architecture.
            Each tuple contains (dim_in, dim_out, kernel_size, stride).
        descriptor_dim: Dimensionality of the learned voxel/centroid descriptors.
        min_tree_voxels: Minimum number of voxels required to form a valid instance
            (currently unused but reserved for future filtering).

    Example:
        >>> instance_head = InstanceHead(
        ...     resnet_blocks=[(64, 32, 3, 1), (32, 16, 3, 2)],
        ...     descriptor_dim=16,
        ...     min_tree_voxels=125
        ... )
        >>> logits = instance_head(feats, peak_indices, confidences, mask, coords, clusters)
        >>> print(f"Instance logits shape: {logits.F.shape}")
        Instance logits shape: torch.Size([N_voxels, N_centroids])
    """
    def __init__(self,
            resnet_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]]=[
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, (1, 1, 3), (1, 1, 2)),
            ],
            descriptor_dim: int = 16,
            semantic_dim: int = 3,
            classification_dim: int = 2
        ):
        super().__init__()

        self.cluster_descriptor = FeatDecoder(
            blocks=resnet_blocks,
            out_dim=descriptor_dim,
            aux_dim=semantic_dim + classification_dim,
            bias=True,
            relu=True
        )
        #self.centroid_descriptor = FeatDecoder(resnet_blocks, descriptor_dim, aux_dim=1, bias=True, relu=True)
        '''
        self.descriptor_conv = nn.Sequential(
            spnn.Conv3d(
                in_channels=descriptor_dim * 2,
                out_channels=descriptor_dim,
                kernel_size=3,
                bias=False
            ),
            # spnn.BatchNorm(descriptor_dim),
            spnn.ReLU(inplace=True)
        )
        '''

        self.descriptor_conv_2d = nn.Sequential(
            nn.Conv2d(
                in_channels=descriptor_dim * 2 + 7,
                out_channels=descriptor_dim + 3,
                kernel_size=1,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=descriptor_dim + 3,
                out_channels=1,
                kernel_size=1,
                bias=True
            )
        )

    @torch.no_grad()
    def _log_inf_or_nan(self, tensor: Tensor, name: str):
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values")
        if torch.isneginf(tensor).any():
            print(f"{name} contains -Inf values")
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values")

    def forward(self,
            feats: List[SparseTensor],
            peak_indices: Tensor,
            centroid_confidences: SparseTensor,
            ng_mask: Tensor,
            semantic_output: SparseTensor,
            classification_output: SparseTensor,
            offset_output: SparseTensor
        ) -> SparseTensor:
        """
        Forward pass for instance assignment prediction.

        Args:
            feats: List of SparseTensors from encoder backbone, used for
                feature decoding to obtain voxel-level descriptors.
            peak_indices: 1-D Long tensor indexing into voxel_descriptors,
                identifying which voxels correspond to detected centroids.
            centroid_confidences: SparseTensor containing coordinates and
                confidence scores of detected centroid peaks.
            ng_mask: Boolean tensor [N_voxels] selecting non-ground voxels
                for instance prediction (ground voxels are typically excluded).
            cluster_coords: Float tensor [N_clusters, 4] with batch and spatial
                coordinates of revoxelized cluster centers from offset head.
            cluster_map: Long tensor [N_voxels] mapping each voxel to its
                cluster index for aggregating cluster-level descriptors.

        Returns:
            SparseTensor with instance assignment logits of shape [N_voxels, N_centroids].
            Each element [i, j] represents the logit for assigning voxel i to centroid j.
            When no centroids are detected, returns empty features with 0 columns.

        Notes:
            - Voxel descriptors are enhanced with cluster-aggregated features
            - Distance features (voxel-to-centroid, cluster-to-centroid) are computed
            - A 2D CNN processes the [voxel x centroid x features] tensor per batch
            - Logits are clamped to [-10, 10]
            - Processing is done independently per batch to handle variable centroid counts and mitigate memory usage
        """
        voxel_descriptors = self.voxel_descriptors = self.cluster_descriptor(feats, mask=ng_mask, aux=[semantic_output.F[ng_mask], classification_output.F])
        if peak_indices.size(0) == 0:
            return SparseTensor(coords=voxel_descriptors.C, feats=torch.empty(voxel_descriptors.F.size(0), 0, dtype=centroid_confidences.F.dtype, device=centroid_confidences.F.device))

        centroid_descriptors = SparseTensor(
            coords=voxel_descriptors.C[peak_indices],
            feats=cat([voxel_descriptors.F[peak_indices], centroid_confidences.F], dim=1)
        )

        new_coords = voxel_descriptors.C[:, 1:].to(offset_output.F.dtype) + offset_output.F
        voxel_descriptors.F = cat([offset_output.F, voxel_descriptors.F], dim=1)

        cluster_descriptors = voxel_descriptors

        batch_indices = torch.unique(cluster_descriptors.C[:, 0])
        output_features = torch.full((cluster_descriptors.F.size(0), centroid_descriptors.F.size(0)),
                                    fill_value=-float('inf'),
                                    dtype=cluster_descriptors.F.dtype,
                                    device=cluster_descriptors.F.device)

        for batch_idx in batch_indices:
            voxel_mask = cluster_descriptors.C[:, 0] == batch_idx
            centroid_mask = centroid_descriptors.C[:, 0] == batch_idx

            if not centroid_mask.any():
                continue

            batch_voxel_descriptors = cluster_descriptors.F[voxel_mask]
            batch_centroid_descriptors = centroid_descriptors.F[centroid_mask]

            with torch.no_grad():
                batch_cluster_dists = (new_coords[voxel_mask][:, None, :] - centroid_descriptors.C[centroid_mask][:, 1:][None, :, :]).to(new_coords.dtype)
                self._log_inf_or_nan(batch_cluster_dists, 'Cluster dists')

            batch_voxel_descriptors_exp = batch_voxel_descriptors.unsqueeze(1).expand(-1, batch_centroid_descriptors.size(0), -1)
            batch_centroid_descriptors_exp = batch_centroid_descriptors.unsqueeze(0).expand(batch_voxel_descriptors.size(0), -1, -1)
            self._log_inf_or_nan(batch_voxel_descriptors, 'Batch voxel descriptors')
            self._log_inf_or_nan(batch_centroid_descriptors_exp, 'Batch centroid descriptors expanded')
            self._log_inf_or_nan(batch_centroid_descriptors, 'Batch centroid descriptors')
            self._log_inf_or_nan(batch_centroid_descriptors_exp, 'Batch centroid descriptors expanded')

            del batch_voxel_descriptors, batch_centroid_descriptors
            batch_output = self.descriptor_conv_2d(torch.cat([
                batch_voxel_descriptors_exp,
                batch_centroid_descriptors_exp,
                batch_cluster_dists
            ], dim=-1).permute(2, 0, 1).contiguous().unsqueeze(0))[0, 0, :, :].to(output_features.dtype).clamp(min=-10.0, max=10.0).nan_to_num(0.0)

            self._log_inf_or_nan(batch_output, 'Batch output')

            rows = voxel_mask.nonzero(as_tuple=False)
            cols = centroid_mask.nonzero(as_tuple=False).squeeze(1)

            output_features[rows, cols] = batch_output
            del batch_output

        return SparseTensor(coords=voxel_descriptors.C, feats=output_features)
        