import math
import torch
import torch.nn.functional as F
import torchsparse.nn.functional as spf

from typing import Tuple

from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor


class InstanceHead(nn.Module):
    def __init__(self, latent_dim, descriptor_dim):
        super().__init__()
        # self.background_descriptor = nn.Parameter(torch.empty(1, descriptor_dim), requires_grad=True)
        self.voxel_descriptor = nn.Sequential(
            spnn.Conv3d(latent_dim, descriptor_dim, 1, bias=True),
            # spnn.ReLU(True),
            # spnn.Conv3d(latent_dim // 2, descriptor_dim, 1),
            # spnn.ReLU(True),
        )


    def forward(self, cluster_feats: SparseTensor, centroid_feats: SparseTensor, centroid_confidences: SparseTensor) -> SparseTensor:
        cluster_descriptors = self.voxel_descriptor(cluster_feats)
        cluster_descriptors.F = F.normalize(cluster_descriptors.F, p=2, dim=1)

        if centroid_feats.F.size(0) == 0:
            return SparseTensor(coords=cluster_feats.C, feats=torch.empty(cluster_feats.F.size(0), 0, dtype=cluster_descriptors.F.dtype, device=cluster_feats.F.device))
        
        centroid_descriptors = self.voxel_descriptor(centroid_feats)
        centroid_descriptors.F = centroid_confidences.F * centroid_descriptors.F

        batch_indices = torch.unique(centroid_feats.C[:, 0])
        output_features = torch.zeros(cluster_feats.F.size(0), centroid_feats.F.size(0), 
                                    dtype=cluster_descriptors.F.dtype, 
                                    device=cluster_descriptors.F.device)

        centroid_offset = 0
        for batch_idx in batch_indices:
            cluster_mask = cluster_feats.C[:, 0] == batch_idx
            centroid_mask = centroid_feats.C[:, 0] == batch_idx

            if not centroid_mask.any():
                continue

            batch_dists = (cluster_feats.C[cluster_mask, 1:][:, None, :] - centroid_feats.C[centroid_mask, 1:][None, :, :]).to(cluster_descriptors.F.dtype)
            batch_dists = torch.norm(batch_dists, p=2, dim=-1).clamp(min=0.1)
            attention_weights = F.softmax(-batch_dists, dim=-1)
            
            batch_centroid_descriptors = centroid_descriptors.F[centroid_mask]
            batch_output = (cluster_descriptors.F[cluster_mask] @ batch_centroid_descriptors.T) * attention_weights

            output_features[cluster_mask, centroid_offset:centroid_offset + batch_centroid_descriptors.size(0)] = batch_output
            centroid_offset += batch_centroid_descriptors.size(0)

        return SparseTensor(coords=cluster_feats.C, feats=output_features)
