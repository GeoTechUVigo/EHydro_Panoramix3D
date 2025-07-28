from typing import Tuple, List, Union

from ..modules import VoxelDecoder, InstanceHead, CentroidHead
from torch import nn
from torchsparse import nn as spnn, SparseTensor
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
            latent_dim = 512,
            instance_density = 0.01,
            centroid_thres = 0.1,
            descriptor_dim = 16
        ):

        super().__init__()
        
        self.encoder = SparseResNet(blocks=resnet_blocks,in_channels=in_channels)
        self.voxel_decoder = VoxelDecoder(resnet_blocks, latent_dim)
        self.semantic_head = spnn.Conv3d(latent_dim, num_classes, 1, bias=True)
        self.centroid_head = CentroidHead(latent_dim, instance_density=instance_density)
        self.instance_head = InstanceHead(latent_dim, descriptor_dim, tau=centroid_thres)

    def forward(self, x: SparseTensor, centroid_score_labels: SparseTensor = None) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor]:
        feats = self.voxel_decoder(self.encoder(x))
        semantic_output = self.semantic_head(feats)
        centroid_score_output = self.centroid_head(feats, semantic_output)

        if centroid_score_labels is None:
            centroid_confidence_output, voxel_descriptors, center_descriptors = self.instance_head(feats, centroid_score_output)
        else:
            centroid_confidence_output, voxel_descriptors, center_descriptors = self.instance_head(feats, centroid_score_labels)

        return semantic_output, centroid_score_output, centroid_confidence_output, voxel_descriptors, center_descriptors
    