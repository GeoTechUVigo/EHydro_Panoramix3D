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
            descriptor_dim = 16
        ):

        super().__init__()
        
        self.encoder = SparseResNet(blocks=resnet_blocks,in_channels=in_channels)
        self.voxel_decoder = VoxelDecoder(resnet_blocks, latent_dim)
        self.semantic_head = SparseConvBlock(latent_dim, num_classes, 1)
        self.centroid_head = CentroidHead(latent_dim, instance_density=instance_density, score_thres=score_thres, centroid_thres=centroid_thres)
        self.offset_head = OffsetHead(latent_dim)
        self.instance_head = InstanceHead(latent_dim, descriptor_dim)

    def forward(self, x: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor]:
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)

        offsets, cluster_feats, inv_map = self.offset_head(feats)
        centroid_scores, centroid_feats, centroid_confidences = self.centroid_head(feats, cluster_feats, inv_map)
        instance_output = self.instance_head(cluster_feats, centroid_feats, centroid_confidences)
        instance_output.F = instance_output.F.index_select(0, inv_map)

        return semantic_output, centroid_scores, offsets, centroid_confidences, instance_output
    