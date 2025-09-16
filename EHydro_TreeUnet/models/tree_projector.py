import torch
import torch.nn.functional as F

from typing import Tuple, List, Union

from ..modules import FeatDecoder, CentroidHead, OffsetHead, InstanceHead
from torch import nn
from torchsparse import SparseTensor
from torchsparse.backbones.resnet import SparseResNet


class TreeProjector(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            resnet_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]]=[
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, (1, 1, 3), (1, 1, 2)),
            ],
            instance_density = 0.01,
            score_thres = 0.1,
            centroid_thres = 0.2,
            descriptor_dim = 16
        ):

        super().__init__()
        
        self.encoder = SparseResNet(
            blocks=resnet_blocks,
            in_channels=in_channels
        )

        self.descriptor = FeatDecoder(resnet_blocks, descriptor_dim, bias=True)

        self.semantic_head = FeatDecoder(
            blocks=resnet_blocks,
            out_dim=num_classes,
            bias=True
        )

        self.offset_head = OffsetHead(
            decoder_blocks=resnet_blocks,
            descriptor_dim=descriptor_dim
        )

        self.centroid_head = CentroidHead(
            decoder_blocks=resnet_blocks,
            descriptor_dim=descriptor_dim,
            instance_density=instance_density,
            score_thres=score_thres,
            centroid_thres=centroid_thres
        )

        self.instance_head = InstanceHead()

    def forward(self, x: SparseTensor, semantic_labels: SparseTensor = None, centroid_score_labels: SparseTensor = None, offset_labels: SparseTensor = None) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor]:
        feats = self.encoder(x)
        semantic_output = self.semantic_head(feats)

        if semantic_labels is None:
            semantic_labels = (semantic_output.F.argmax(dim=1))
        else:
            semantic_labels = semantic_labels.F

        ng_mask = (semantic_labels != 0)
        #offsets, cluster_descriptors, centroid_scores_off, inv_map = self.offset_head(feats, mask=ng_mask, offset_labels=offset_labels)
        offsets, cluster_descriptors, centroid_scores_off, inv_map = self.offset_head(feats, mask=ng_mask, offset_labels=offset_labels)
        centroid_scores, peak_indices, centroid_confidences = self.centroid_head(feats, centroid_scores_off, mask=ng_mask, centroid_score_labels=centroid_score_labels)

        voxel_descriptors = self.descriptor(feats, mask=ng_mask)
        voxel_descriptors.F = F.normalize(voxel_descriptors.F, p=2, dim=1)

        centroid_descriptors = SparseTensor(
            coords=voxel_descriptors.C[peak_indices],
            feats=voxel_descriptors.F[peak_indices] * centroid_confidences.F
        )

        instance_output = self.instance_head(voxel_descriptors, centroid_descriptors)
        #instance_output = self.instance_head(cluster_descriptors, centroid_descriptors)
        #instance_output.C = offsets.C
        #instance_output.F = instance_output.F.index_select(0, inv_map)

        return semantic_output, centroid_scores, offsets, centroid_confidences, instance_output
    