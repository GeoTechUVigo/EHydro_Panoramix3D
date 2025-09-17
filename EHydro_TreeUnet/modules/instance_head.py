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
        self.descriptor_conv = nn.Sequential(
            spnn.Conv3d(
                in_channels=descriptor_dim * 2,
                out_channels=descriptor_dim,
                kernel_size=3,
                bias=True
            ),
            spnn.BatchNorm(descriptor_dim),
            spnn.ReLU(inplace=True)
        )

        self.descriptor_conv_2d = nn.Sequential(
            nn.Conv2d(
                in_channels=descriptor_dim * 2 + 1,
                out_channels=descriptor_dim,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(descriptor_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=descriptor_dim,
                out_channels=1,
                kernel_size=1,
                bias=True
            )
        )

    def forward(self,
            feats: List[SparseTensor],
            peak_indices: Tensor,
            centroid_confidences: SparseTensor,
            ng_mask: Tensor,
            cluster_map: Tensor
        ) -> SparseTensor:
        voxel_descriptors = self.descriptor(feats, mask=ng_mask)
        # voxel_descriptors.F = F.normalize(voxel_descriptors.F, p=2, dim=1)
        cluster_descriptors = scatter_mean(voxel_descriptors.F, cluster_map, dim=0)
        cluster_descriptors = cluster_descriptors.index_select(0, cluster_map)
        voxel_descriptors.F = cat([voxel_descriptors.F, cluster_descriptors], dim=1)
        voxel_descriptors = self.descriptor_conv(voxel_descriptors)

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

            batch_voxel_descriptors = voxel_descriptors.F[voxel_mask]
            batch_centroid_descriptors = centroid_descriptors.F[centroid_mask]

            with torch.no_grad():
                batch_dists = (batch_voxel_descriptors[:, None, :] - batch_centroid_descriptors[None, :, :]).to(voxel_descriptors.F.dtype)
                batch_dists = torch.norm(batch_dists, p=2, dim=-1).clamp(min=1).unsqueeze(-1)

            batch_voxel_descriptors_exp = batch_voxel_descriptors.unsqueeze(1).expand(-1, batch_centroid_descriptors.size(0), -1)
            batch_centroid_descriptors_exp = batch_centroid_descriptors.unsqueeze(0).expand(batch_voxel_descriptors.size(0), -1, -1)
            batch_input = torch.cat([batch_voxel_descriptors_exp, batch_centroid_descriptors_exp, batch_dists], dim=-1).permute(2, 0, 1).contiguous().unsqueeze(0)
            batch_output = self.descriptor_conv_2d(batch_input)

            rows = voxel_mask.nonzero(as_tuple=False)
            cols = centroid_mask.nonzero(as_tuple=False).squeeze(1)

            output_features[rows, cols] = batch_output[0, 0, :, :].to(output_features.dtype)

        return SparseTensor(coords=voxel_descriptors.C, feats=output_features)
        