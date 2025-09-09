import torch

from typing import Tuple, List, Union

from ..modules import VoxelDecoder, CentroidHead, OffsetHead, InstanceHead
from torch import nn, cat
from torchsparse import nn as spnn, SparseTensor
from torchsparse.backbones.resnet import SparseResNet
from torchsparse.backbones.modules import SparseConvBlock


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
            latent_dim = 512,
            instance_density = 0.01,
            score_thres = 0.1,
            centroid_thres = 0.2,
            descriptor_dim = 32
        ):

        super().__init__()
        
        self.encoder = SparseResNet(blocks=resnet_blocks,in_channels=in_channels)
        self.voxel_decoder = VoxelDecoder(resnet_blocks, latent_dim)
        self.semantic_head = SparseConvBlock(latent_dim, num_classes, 1)
        self.centroid_head = CentroidHead(latent_dim, instance_density=instance_density, score_thres=score_thres, centroid_thres=centroid_thres)
        self.offset_head = OffsetHead(latent_dim)
        self.instance_head = InstanceHead(latent_dim, descriptor_dim)

    def _sparse_select(self, st: SparseTensor, mask: torch.Tensor):
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            empty_coords = st.C.new_empty(0, st.C.size(1))
            empty_feats  = st.F.new_empty(0, st.F.size(1))
            return SparseTensor(coords=empty_coords, feats=empty_feats), idx
        
        return SparseTensor(coords=st.C.index_select(0, idx), feats=st.F.index_select(0, idx)), idx

    def _sparse_restore(self, sub: SparseTensor, idx: torch.Tensor, template: SparseTensor, fill=0.0):
        if sub.F.dim() == 1:
            C = 1
        else:
            C = sub.F.size(1)

        outF = template.F.new_full((template.F.size(0), C), fill)
        if idx.numel() > 0:
            outF.index_copy_(0, idx, sub.F)

        return SparseTensor(coords=template.C, feats=outF)

    def forward(self, x: SparseTensor, semantic_labels: SparseTensor = None, centroid_score_labels: SparseTensor = None, offset_labels: SparseTensor = None) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor]:
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)

        if semantic_labels is None:
            semantic_labels = (semantic_output.F.argmax(dim=1))
        else:
            semantic_labels = semantic_labels.F
        
        ng_mask = (semantic_labels != 0)
        feats_ng, idx_ng = self._sparse_select(feats, ng_mask)

        offsets, cluster_feats, centroid_scores_off, inv_map = self.offset_head(feats_ng, offset_labels)
        centroid_scores, centroid_feats, centroid_confidences = self.centroid_head(feats_ng, cluster_feats, centroid_scores_off, inv_map, centroid_score_labels)

        instance_output = self.instance_head(cluster_feats, centroid_feats, centroid_confidences)
        instance_output.C = feats_ng.C
        instance_output.F = instance_output.F.index_select(0, inv_map)

        return semantic_output, centroid_scores, offsets, centroid_confidences, instance_output
    