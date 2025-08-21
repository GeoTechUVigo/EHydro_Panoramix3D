from typing import Tuple, List, Union

from ..modules import VoxelDecoder, CentroidHead, OffsetHead, InstanceHead
from torch import nn
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
            centroid_thres = 0.1,
            descriptor_dim = 16
        ):

        super().__init__()
        
        self.encoder = SparseResNet(blocks=resnet_blocks,in_channels=in_channels)
        self.voxel_decoder = VoxelDecoder(resnet_blocks, latent_dim)
        self.semantic_head = nn.Sequential(
            SparseConvBlock(latent_dim, latent_dim // 2, 3),
            SparseConvBlock(latent_dim // 2, num_classes, 3)
        )
        self.centroid_head = CentroidHead(latent_dim)
        self.offset_head = OffsetHead(latent_dim)
        self.instance_head = InstanceHead(latent_dim, descriptor_dim, tau=centroid_thres, instance_density=instance_density)

    def forward(self, x: SparseTensor, centroid_score_labels: SparseTensor = None, offset_labels: SparseTensor = None) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor]:
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)
        centroid_score_output = self.centroid_head(feats, semantic_output)
        offset_output = self.offset_head(feats)

        # if centroid_score_labels is None or offset_labels is None:
        refined_centroid_scores, centroid_confidence_output, instance_output = self.instance_head(feats, centroid_score_output, offset_output)
        # else:
        #     refined_centroid_scores, centroid_confidence_output, instance_output = self.instance_head(feats, centroid_score_labels, offset_labels)

        return semantic_output, refined_centroid_scores, offset_output, centroid_confidence_output, instance_output
    