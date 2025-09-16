import math
import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import List, Tuple, Union

from torch import nn, Tensor
from torchsparse import nn as spnn, SparseTensor

from ..modules import FeatDecoder


class InstanceHead(nn.Module):
    def __init__(self,
            resnet_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]]=[
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, (1, 1, 3), (1, 1, 2)),
            ],
            descriptor_dim = 16
        ):
        super().__init__()

        self.descriptor = FeatDecoder(resnet_blocks, descriptor_dim, bias=True)

    def forward(self, feats: List[SparseTensor], peak_indices: Tensor, centroid_confidences: SparseTensor, ng_mask: Tensor) -> SparseTensor:
        voxel_descriptors = self.descriptor(feats, mask=ng_mask)
        voxel_descriptors.F = F.normalize(voxel_descriptors.F, p=2, dim=1)

        centroid_descriptors = SparseTensor(
            coords=voxel_descriptors.C[peak_indices],
            feats=voxel_descriptors.F[peak_indices] * centroid_confidences.F
        )

        if centroid_descriptors.F.size(0) == 0:
            return SparseTensor(coords=voxel_descriptors.C, feats=torch.empty(voxel_descriptors.F.size(0), 0, dtype=voxel_descriptors.F.dtype, device=voxel_descriptors.F.device))

        batch_indices = torch.unique(voxel_descriptors.C[:, 0])
        output_features = torch.full((voxel_descriptors.F.size(0), centroid_descriptors.F.size(0)),
                                    fill_value=-float('inf'),
                                    dtype=voxel_descriptors.F.dtype,
                                    device=voxel_descriptors.F.device)

        for batch_idx in batch_indices:
            voxel_mask = voxel_descriptors.C[:, 0] == batch_idx
            centroid_mask = centroid_descriptors.C[:, 0] == batch_idx

            if not centroid_mask.any():
                continue

            with torch.no_grad():
                batch_dists = (voxel_descriptors.C[voxel_mask, 1:][:, None, :] - centroid_descriptors.C[centroid_mask, 1:][None, :, :]).to(voxel_descriptors.F.dtype)
                batch_dists = torch.norm(batch_dists, p=2, dim=-1).clamp(min=0.1)
                attention_weights = F.softmax(-batch_dists, dim=-1)
            
            batch_centroid_descriptors = centroid_descriptors.F[centroid_mask]
            batch_output = ((voxel_descriptors.F[voxel_mask] @ batch_centroid_descriptors.T) * attention_weights).clamp(min=-10, max=10)
            #print(f'max output: {batch_output.max().item():.4f}, min output: {batch_output.min().item():.4f}')

            rows = voxel_mask.nonzero(as_tuple=False)
            cols = centroid_mask.nonzero(as_tuple=False).squeeze(1)

            output_features[rows, cols] = batch_output.to(output_features.dtype)

        return SparseTensor(coords=voxel_descriptors.C, feats=output_features)
        