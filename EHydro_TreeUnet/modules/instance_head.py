import math
import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import Tuple

from torch import nn, Tensor
from torchsparse import nn as spnn, SparseTensor


class InstanceHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cluster_descriptors: SparseTensor, centroid_descriptors: SparseTensor) -> SparseTensor:
        if centroid_descriptors.F.size(0) == 0:
            return SparseTensor(coords=cluster_descriptors.C, feats=torch.empty(cluster_descriptors.F.size(0), 0, dtype=cluster_descriptors.F.dtype, device=cluster_descriptors.F.device))

        batch_indices = torch.unique(cluster_descriptors.C[:, 0])
        output_features = torch.full((cluster_descriptors.F.size(0), centroid_descriptors.F.size(0)),
                                    fill_value=-float('inf'),
                                    dtype=cluster_descriptors.F.dtype,
                                    device=cluster_descriptors.F.device)

        for batch_idx in batch_indices:
            cluster_mask = cluster_descriptors.C[:, 0] == batch_idx
            centroid_mask = centroid_descriptors.C[:, 0] == batch_idx

            if not centroid_mask.any():
                continue

            with torch.no_grad():
                batch_dists = (cluster_descriptors.C[cluster_mask, 1:][:, None, :] - centroid_descriptors.C[centroid_mask, 1:][None, :, :]).to(cluster_descriptors.F.dtype)
                batch_dists = torch.norm(batch_dists, p=2, dim=-1).clamp(min=0.1)
                attention_weights = F.softmax(-batch_dists, dim=-1)
            
            batch_centroid_descriptors = centroid_descriptors.F[centroid_mask]
            batch_output = ((cluster_descriptors.F[cluster_mask] @ batch_centroid_descriptors.T) * attention_weights).clamp(min=-10, max=10)
            #print(f'max output: {batch_output.max().item():.4f}, min output: {batch_output.min().item():.4f}')

            rows = cluster_mask.nonzero(as_tuple=False)
            cols = centroid_mask.nonzero(as_tuple=False).squeeze(1)

            output_features[rows, cols] = batch_output.to(output_features.dtype)

        return SparseTensor(coords=cluster_descriptors.C, feats=output_features)
        