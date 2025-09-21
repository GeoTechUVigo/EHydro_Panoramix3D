import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import List, Tuple, Union

from torch import nn, Tensor, cat
from torchsparse import SparseTensor

from . import FeatDecoder


class OffsetHead(nn.Module):
    """
    This class predicts per-voxel offset vectors that shift voxels towards their
    instance centers, enabling instance clustering through spatial aggregation.
    After prediction, it performs revoxelization to group voxels that land on
    the same shifted coordinates, creating clusters for downstream instance processing.

    The offset prediction follows a simple decoder architecture, while revoxelization
    uses sparse hashing to efficiently group shifted coordinates and build cluster
    mappings for subsequent instance assignment.

    Args:
        decoder_blocks: List of tuples defining the decoder architecture.
            Each tuple contains (dim_in, dim_out, kernel_size, stride).
            The decoder outputs 3-channel offset vectors (x, y, z displacements).

    Example:
        >>> offset_head = OffsetHead(
        ...     decoder_blocks=[(64, 32, 3, 1), (32, 16, 3, 2), (16, 3, 1, 1)]
        ... )
        >>> offsets, cluster_coords, inv_map = offset_head(feats, mask)
        >>> print(f"Predicted {offsets.F.shape[0]} offset vectors")
        >>> print(f"Created {cluster_coords.shape[0]} clusters")
        Predicted 1024 offset vectors
        Created 156 clusters
    """
    def __init__(self, 
            decoder_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]]
        ):
        super().__init__()

        self.decoder = FeatDecoder(decoder_blocks, 3, bias=True)

    @torch.no_grad()
    def _revoxelize(self, offsets: SparseTensor) -> Tuple[SparseTensor, Tensor]:
        """
        Revoxelize point cloud by applying offset vectors and clustering shifted coordinates.
        
        This method adds predicted offsets to original voxel coordinates, then uses
        sparse hashing to group voxels that land on the same shifted grid cells.
        The result is a set of cluster coordinates and an inverse mapping from
        original voxels to their assigned clusters.

        Args:
            offsets: SparseTensor with 3-channel offset predictions [N_voxels, 3].
                Coordinates (.C) contain original voxel positions.

        Returns:
            A tuple (cluster_coords, inv_map) where:
                - cluster_coords: Tensor [N_clusters, 4] containing batch and spatial
                  coordinates of unique cluster centers after offset application.
                - inv_map: Long tensor [N_voxels] mapping each original voxel to its
                  cluster index in cluster_coords.

        Notes:
            - Offsets are rounded to integer values before coordinate addition.
            - Sparse hashing enables efficient grouping of identical shifted coordinates.
            - Multiple original voxels may map to the same cluster.
        """
        offsets_ = cat([
            torch.zeros(offsets.F.size(0), 1, device=offsets.F.device, dtype=torch.int32),
            offsets.F.round().to(torch.int32)
        ], dim=1)

        new_coords = offsets.C + offsets_

        pc_hash        = spf.sphash(new_coords)
        voxel_hash     = torch.unique(pc_hash)
        idx_query      = spf.sphashquery(pc_hash, voxel_hash)
        idx_query_long = idx_query.to(torch.int64)
        counts         = spf.spcount(idx_query, voxel_hash.numel())
        out_coords     = spf.spvoxelize(new_coords.float(), idx_query, counts).int()

        return out_coords, idx_query_long

    def forward(self, feats: List[SparseTensor], mask: Tensor, offset_labels: SparseTensor = None) -> Tuple[SparseTensor, Tensor, Tensor]:
        """
        Forward pass for offset prediction and revoxelization.

        Args:
            feats: List of SparseTensors from encoder backbone, used for
                multi-scale feature decoding to predict offset vectors.
            mask: Boolean tensor [N_voxels] selecting which voxels to process
                (typically non-ground voxels for instance segmentation).
            offset_labels: Optional SparseTensor with ground truth offset vectors.
                When provided during training, GT offsets are used for revoxelization
                instead of predictions to ensure consistent supervision.

        Returns:
            A tuple (offsets, cluster_coords, inv_map) where:
                - offsets: SparseTensor [N_masked_voxels, 3] with predicted offset
                  vectors for each valid voxel.
                - cluster_coords: Tensor [N_clusters, 4] containing coordinates of
                  unique clusters formed after applying offsets and revoxelization.
                - inv_map: Long tensor [N_masked_voxels] mapping each original voxel
                  to its cluster index in cluster_coords.

        Notes:
            - Only masked voxels are processed for efficiency.
            - During training, GT offsets are used for revoxelization if provided.
            - The inv_map enables subsequent cluster-wise feature aggregation.
        """
        offsets = self.decoder(feats, mask=mask)
        out_coords, inv_map = self._revoxelize(offsets if offset_labels is None else offset_labels)

        return offsets, out_coords, inv_map
