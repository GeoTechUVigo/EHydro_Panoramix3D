import torch

from torch import nn
from torchsparse import SparseTensor
from torch_scatter import scatter_mean

from typing import Tuple
from pathlib import Path

from ..modules import SparseResNet, FeatDecoder, CentroidHead, OffsetHead, InstanceHead
from ..config import ModelConfig

class Panoramix3D(nn.Module):
    """
    This class implements a multi-task 3D sparse convolutional network that simultaneously
    solves semantic segmentation, instance detection, offset prediction, and instance
    assignment for point cloud data. It consists of a shared encoder backbone followed
    by specialized decoder heads that work together to provide comprehensive scene
    understanding in a single forward pass.

    The architecture addresses four fundamental computer vision questions:
    1. What is this voxel? (semantic segmentation)
    2. Where are the instance centers? (centroid detection)
    3. How should voxels move towards centers? (offset prediction)
    4. Which voxels belong to which instance? (instance assignment)

    Args:
        cfg: ModelConfig object containing all model configuration parameters:
            - cfg.backbone.in_channels: Number of input feature channels (default: 1)
            - cfg.backbone.resnet_blocks: List of layer specifications for encoder/decoder
                Each tuple contains (num_blocks, out_channels, kernel_size, stride)
            - cfg.semantic_head.num_classes: Number of semantic classes (default: 3)
            - cfg.centroid_head.instance_density: Prior instance density for bias init (default: 0.01)
            - cfg.centroid_head.score_thres: Initial centroid filtering threshold (default: 0.2)
            - cfg.centroid_head.centroid_thres: Final centroid acceptance threshold (default: 0.3)
            - cfg.centroid_head.max_instances_per_scene: Max detectable instances (default: 64)
            - cfg.instance_head.descriptor_dim: Descriptor dimensionality for assignment (default: 16)

    Example:
        >>> from panoramix3D.config import ModelConfig
        >>> cfg = ModelConfig.from_yaml('model_config.yaml')
        >>> model = Panoramix3D(cfg)
        >>> outputs = model(sparse_input)
        >>> semantic, centroids, offsets, confidences, instances = outputs
        >>> print(f"Detected {confidences.F.shape[0]} instance centers")
        Detected 12 instance centers

    Notes:
        - Configuration provides type validation and default values via Pydantic
        - All parameters can be overridden through YAML configuration files
        - Backbone architecture supports flexible layer specifications with tuples
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        
        self.encoder = SparseResNet(
            blocks=cfg.backbone.resnet_blocks,
            in_channels=cfg.backbone.in_channels
        )

        self.semantic_head = FeatDecoder(
            blocks=cfg.backbone.resnet_blocks,
            out_dim=cfg.semantic_head.num_classes,
            bias=True
        )

        self.classification_head = FeatDecoder(
            blocks=cfg.backbone.resnet_blocks,
            out_dim=cfg.classification_head.num_classes,
            bias=True
        )

        self.offset_head = OffsetHead(
            decoder_blocks=cfg.backbone.resnet_blocks
        )

        self.centroid_head = CentroidHead(
            decoder_blocks=cfg.backbone.resnet_blocks,
            instance_density=cfg.centroid_head.instance_density,
            score_thres=cfg.centroid_head.score_thres,
            centroid_thres=cfg.centroid_head.centroid_thres,
            max_trees_per_scene=cfg.centroid_head.max_instances_per_scene
        )

        self.instance_head = InstanceHead(
            resnet_blocks=cfg.backbone.resnet_blocks,
            descriptor_dim=cfg.instance_head.descriptor_dim,
            semantic_dim=cfg.semantic_head.num_classes,
            classification_dim=cfg.classification_head.num_classes
        )

        foreground_classes = torch.tensor(cfg.foreground_classes)
        self.register_buffer('foreground_classes', foreground_classes, persistent=False)

    def forward(self, x: SparseTensor, semantic_labels: SparseTensor = None) -> Tuple[SparseTensor, SparseTensor, SparseTensor, SparseTensor, SparseTensor]:
        """
        Forward pass through the multi-task network.

        This method performs a complete forward pass through all heads, orchestrating
        the flow of information between different components. The semantic predictions
        are used to create a non-ground mask, which filters subsequent instance-related
        computations for efficiency and accuracy.

        Args:
            x: Input SparseTensor containing voxelized point cloud data with
                features of shape [N_voxels, in_channels].
            semantic_labels: Optional SparseTensor with ground truth semantic
                labels. When provided during training, GT labels are used for
                creating the non-ground mask instead of predictions.
            centroid_score_labels: Optional SparseTensor with ground truth
                centroid confidence scores. Used for consistent supervision
                during training in the centroid head.
            offset_labels: Optional SparseTensor with ground truth offset vectors.
                Used for consistent revoxelization during training in the offset head.

        Returns:
            A tuple (semantic_output, centroid_scores, offsets, centroid_confidences, instance_output) where:
                - semantic_output: SparseTensor [N_voxels, num_classes] with per-voxel
                  class logits for semantic segmentation.
                - centroid_scores: SparseTensor [N_non_ground_voxels, 1] with confidence
                  scores for centroid detection at each valid location.
                - offsets: SparseTensor [N_non_ground_voxels, 3] with predicted
                  displacement vectors for moving voxels towards instance centers.
                - centroid_confidences: SparseTensor [N_detected_peaks, 1] containing
                  coordinates and confidence scores of detected instance centers only.
                - instance_output: SparseTensor [N_non_ground_voxels, N_detected_peaks]
                  with assignment logits between each voxel and each detected centroid.

        Notes:
            - Non-ground masking is applied after semantic prediction to focus
              instance processing on relevant voxels (semantic class != 0).
            - During training, GT labels can replace predictions for consistent supervision.
            - The number of output columns in instance_output equals the number of
              detected centroids, which varies per scene and batch element.
            - All heads share the same encoder features but process different subsets
              of voxels based on the semantic mask.
        """
        feats = self.encoder(x)
        if feats is None:
            return None

        semantic_output = self.semantic_head(feats)

        if semantic_labels is None:
            semantic_labels = (semantic_output.F.argmax(dim=1))
        else:
            semantic_labels = semantic_labels.F

        fg_mask = torch.isin(semantic_labels, self.foreground_classes)
        classification_output = self.classification_head(feats, mask=fg_mask)
        offset_output = self.offset_head(feats, mask=fg_mask)
        centroid_scores, peak_indices, centroid_confidences = self.centroid_head(feats, mask=fg_mask)
        instance_output = self.instance_head(feats, peak_indices, centroid_confidences, fg_mask, semantic_output, classification_output, offset_output)

        if instance_output.F.size(0) > 0 and instance_output.F.size(1) > 0:
            with torch.no_grad():
                instance_labels = torch.argmax(instance_output.F, dim=1)
            classification_output.F = scatter_mean(classification_output.F, instance_labels, dim=0).index_select(0, instance_labels)

        return semantic_output, classification_output, centroid_scores, offset_output, centroid_confidences, instance_output

    def load_weights(self, ckpt: dict | str | Path, key: str = 'model_state_dict') -> None:
        if isinstance(ckpt, str) or isinstance(ckpt, Path):
            ckpt = torch.load(ckpt, map_location=next(self.parameters()).device)

        state = ckpt.get(key, ckpt)
        self.load_state_dict(state)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "Panoramix3D":
        model = cls(cfg)
        if cfg.weights_file is not None:
            model.load_weights(cfg.weights_file)

        return model
