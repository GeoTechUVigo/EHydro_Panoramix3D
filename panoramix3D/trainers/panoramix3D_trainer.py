import shutil
import torch
import sys
import matplotlib.pyplot as plt
import open3d as o3d

import torchsparse
import numpy as np

from typing import Dict
from open3d.visualization.tensorboard_plugin.util import to_dict_batch

from torch import nn, Tensor
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex, Precision, Recall, F1Score

from torchsparse import SparseTensor

from pathlib import Path
from tqdm import tqdm

from .utils import sparse_unique_id_collate_fn

from ..datasets import Panoramix3DDataset
from ..models import Panoramix3D
from ..modules import CenterNetFocalLoss, HungarianInstanceLoss, OffsetLoss
from ..schedulers import build_lr_scheduler, build_bn_scheduler

from ..config import AppConfig


class Panoramix3DTrainer:
    """
    Main training and validation orchestrator for the Panoramix3D model.
    This class handles the complete training pipeline including model initialization,
    loss computation, metrics calculation, checkpointing, and logging.
    
    The trainer supports multi-task learning with semantic segmentation, centroid 
    detection, offset regression, and instance segmentation. It includes advanced
    features like mixed precision training, learning rate scheduling, module freezing,
    and comprehensive metric tracking.
    
    Args:
        cfg: Global application configuration containing all model, dataset,
            and training parameters.
    
    Attributes:
        _cfg: Application configuration object
        _device: Torch device (CUDA/CPU) for training
        _model: Panoramix3D model instance
        _optimizer: AdamW optimizer with per-module learning rates
        _scaler: Automatic mixed precision scaler
        _criterion_*: Loss functions for different tasks
        _metric_*: Evaluation metrics for semantic segmentation
        _writer: TensorBoard summary writer for logging
    """
    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._foreground_classes = torch.tensor(cfg.model.foreground_classes, dtype=torch.long, device=self._device)

        root_folder = Path(self._cfg.trainer.root)
        if not root_folder.exists():
            raise FileNotFoundError(f"Root folder {root_folder} does not exist.")
        
        self._root_folder = root_folder / self._cfg.trainer.version_name
        if self._cfg.trainer.mode == 'train' and self._cfg.trainer.start_epoch == 0 and self._root_folder.exists():
            shutil.rmtree(self._root_folder)

        self._weights_folder = self._root_folder / 'weights'
        self._checkpoint_folder = self._weights_folder / 'checkpoints'
        self._checkpoint_folder.mkdir(parents=True, exist_ok=True)

        self._logs_folder = self._root_folder / 'logs'
        self._logs_folder.mkdir(parents=True, exist_ok=True)

        if self._cfg.trainer.start_epoch > 0:
            self._cfg.model.weights_file = self._checkpoint_folder / f'{self._cfg.trainer.version_name}_epoch_{self._cfg.trainer.start_epoch}.pth'
        elif self._cfg.trainer.mode == 'train':
            self._cfg.model.weights_file = None
        else:
            self._cfg.model.weights_file = self._weights_folder / f'{self._cfg.trainer.version_name}.pth'

        self._model = Panoramix3D.from_config(self._cfg.model).to(device=self._device)
        self._optimizer = torch.optim.AdamW([
            {'params': self._model.encoder.parameters(), 'lr': self._cfg.trainer.learning_rates.backbone, 'weight_decay': self._cfg.trainer.weight_decays.backbone},
            {'params': self._model.semantic_head.parameters(), 'lr': self._cfg.trainer.learning_rates.semantic, 'weight_decay': self._cfg.trainer.weight_decays.semantic},
            {'params': self._model.classification_head.parameters(), 'lr': self._cfg.trainer.learning_rates.classification, 'weight_decay': self._cfg.trainer.weight_decays.classification},
            {'params': self._model.offset_head.parameters(), 'lr': self._cfg.trainer.learning_rates.offset, 'weight_decay': self._cfg.trainer.weight_decays.offset},
            {'params': self._model.centroid_head.parameters(), 'lr': self._cfg.trainer.learning_rates.centroid, 'weight_decay': self._cfg.trainer.weight_decays.centroid},
            {'params': self._model.instance_head.parameters(), 'lr': self._cfg.trainer.learning_rates.instance, 'weight_decay': self._cfg.trainer.weight_decays.instance}
        ])
        self._scaler = amp.GradScaler(enabled=True)

        self._lr_scheduler = build_lr_scheduler(self._cfg.trainer.lr_scheduler, self._optimizer)
        self._bn_scheduler = build_bn_scheduler(self._cfg.trainer.bn_scheduler, self._model, self._cfg.trainer.start_epoch)

        if self._cfg.trainer.start_epoch > 0:
            self._load_ckpt(self._checkpoint_folder / f'{self._cfg.trainer.version_name}_epoch_{self._cfg.trainer.start_epoch}.pth')

        semantic_class_counts, classification_class_counts = self._get_class_counts()
        semantic_weights = self._compute_weights(semantic_class_counts)
        classification_weights = self._compute_weights(classification_class_counts)

        print(f'Distribution of: semantic_labels:')
        total_points = semantic_class_counts.sum()
        for cls in self._cfg.dataset.semantic_classes:
            print(f'\t* {cls.name} ({cls.id}): {semantic_class_counts[cls.id]:.0f} points ({(semantic_class_counts[cls.id] / total_points * 100):.2f} %), weight: {semantic_weights[cls.id]:.4f}.')
        
        print(f'Distribution of: classification_labels:')
        total_points = classification_class_counts.sum()
        for cls in self._cfg.dataset.instance_classes:
            if cls.id == 0:
                continue

            print(f'\t* {cls.name} ({cls.id}): {classification_class_counts[cls.id - 1]:.0f} points ({(classification_class_counts[cls.id - 1] / total_points * 100):.2f} %), weight: {classification_weights[cls.id - 1]:.4f}.')

        self._criterion_semantic = nn.CrossEntropyLoss(weight=semantic_weights)
        self._criterion_classification = nn.CrossEntropyLoss(weight=classification_weights)
        self._criterion_centroid = CenterNetFocalLoss()
        self._criterion_offset = OffsetLoss()
        self._criterion_instance = HungarianInstanceLoss(
            lambda_matched=self._cfg.trainer.loss_coeffs.instance_matched_loss_coeff,
            lambda_unmatched=self._cfg.trainer.loss_coeffs.instance_unmatched_loss_coeff,
            lambda_bce=self._cfg.trainer.loss_coeffs.instance_bce_loss_coeff,
            lambda_dice=self._cfg.trainer.loss_coeffs.instance_dice_loss_coeff
        )

        self._metric_semantic_iou = MulticlassJaccardIndex(num_classes=self._cfg.model.semantic_head.num_classes, average='none').to(device=self._device)
        self._metric_semantic_precision = Precision(task='multiclass', num_classes=self._cfg.model.semantic_head.num_classes, average='none').to(device=self._device)
        self._metric_semantic_recall = Recall(task='multiclass', num_classes=self._cfg.model.semantic_head.num_classes, average='none').to(device=self._device)
        self._metric_semantic_f1 = F1Score(task='multiclass', num_classes=self._cfg.model.semantic_head.num_classes, average='none').to(device=self._device)
        self._metric_semantic_confmat = ConfusionMatrix(
            task='multiclass', 
            num_classes=self._cfg.model.semantic_head.num_classes
        ).to(device=self._device)

        self._metric_classification_iou = MulticlassJaccardIndex(num_classes=self._cfg.model.classification_head.num_classes, average='none').to(device=self._device)
        self._metric_classification_precision = Precision(task='multiclass', num_classes=self._cfg.model.classification_head.num_classes, average='none').to(device=self._device)
        self._metric_classification_recall = Recall(task='multiclass', num_classes=self._cfg.model.classification_head.num_classes, average='none').to(device=self._device)
        self._metric_classification_f1 = F1Score(task='multiclass', num_classes=self._cfg.model.classification_head.num_classes, average='none').to(device=self._device)

        # self._writer = SummaryWriter(log_dir=self._logs_folder, flush_secs=30)
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print('Resnet generates features at the following scales:')
        scales = [1, 1, 1]
        scales_m = [0, 0, 0]
        out_channels_total = 0
        for i, (_, out_channels, _, strides) in enumerate(cfg.model.backbone.resnet_blocks):
            if isinstance(strides, int):
                strides = (strides, strides, strides)

            scales = [scale * stride for scale, stride in zip(scales, strides)]
            scales_m = [cfg.dataset.voxel_size * scale for scale in scales]

            out_channels_total += out_channels
            print(f'\t* ({scales_m[0]:.1f}, {scales_m[1]:.1f}, {scales_m[2]:.1f}) meters -> {out_channels} feats.')

        scales_m = [3 * scale for scale in scales_m]
        print(f'\nMinimum scene size: ({scales_m[0]:.1f}, {scales_m[1]:.1f}, {scales_m[2]:.1f}) meters')

        self._writer = SummaryWriter(log_dir=self._logs_folder, flush_secs=30)

    def _get_class_counts(self) -> Tensor:
        """
        Compute class counts for semantic segmentation and instance classification labels in the training dataset.
        
        Returns:
            Tensor of class counts for each semantic and classification class
        """

        dataset = Panoramix3DDataset(self._cfg.dataset, split='train')
        semantic_counts = torch.zeros(self._cfg.model.semantic_head.num_classes, dtype=torch.float32, device=self._device)
        classification_counts = torch.zeros(self._cfg.model.classification_head.num_classes, dtype=torch.float32, device=self._device)

        for result in dataset:
            for cls in range(self._cfg.model.semantic_head.num_classes):
                semantic_counts[cls] += (result['semantic_labels'].F == cls).sum()
            for cls in range(self._cfg.model.classification_head.num_classes):
                classification_counts[cls] += (result['classification_labels'].F == (cls + 1)).sum()
                
        return semantic_counts, classification_counts

    def _compute_weights(self, class_counts) -> Tensor:
        """
        Compute class weights inversely proportional to class frequencies for handling class imbalance.

        Args:
            class_counts: 'semantic_labels' or 'classification_labels' to compute weights for respective classes
        Returns:
            Tensor of class weights
        """

        weights = class_counts.mean() / class_counts
        if self._cfg.trainer.class_weights == 'sqrt':
            weights = torch.sqrt(weights).to(self._device)
        elif self._cfg.trainer.class_weights == 'log':
            weights = torch.log(1.1 + weights / weights.sum()).to(self._device)

        return weights / weights.sum()

    def _load_ckpt(self, ckpt_path: Path) -> int:
        """
        Load training checkpoint including model weights, optimizer state,
        and AMP scaler state for resuming training from a specific epoch.
        
        Args:
            ckpt_path: Path to the checkpoint file to load
            
        Returns:
            The epoch number from the loaded checkpoint
        """
        ckpt = torch.load(ckpt_path, map_location=self._device)

        self._optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self._scaler.load_state_dict(ckpt['scaler_state_dict'])

    def _save_ckpt(self, epoch: int) -> None:
        """
        Save complete training checkpoint including model state, optimizer state,
        and AMP scaler state for resuming training later.
        
        Args:
            epoch: Current epoch number to include in checkpoint
        """
        ckpt_path = self._checkpoint_folder / f'{self._cfg.trainer.version_name}_epoch_{epoch}.pth'
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scaler_state_dict': self._scaler.state_dict(),
        }

        torch.save(ckpt, ckpt_path)

    def _save_weights(self) -> None:
        """
        Save only the model weights (state_dict) to the final weights file
        for inference and deployment purposes.
        """
        weights_path = self._weights_folder / f'{self._cfg.trainer.version_name}.pth'
        torch.save(self._model.state_dict(), weights_path)

    def _compute_loss(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            classification_output: SparseTensor,
            classification_labels: SparseTensor,
            centroid_score_output: SparseTensor,
            centroid_score_labels: SparseTensor,
            offset_output: SparseTensor,
            offset_labels: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: SparseTensor,
            centroid_confidences: SparseTensor
        ) -> Tensor:
        """
        Compute multi-task loss combining semantic segmentation, centroid detection,
        offset regression, and instance segmentation losses with configurable coefficients.
        
        Args:
            semantic_output: Model predictions for semantic classes
            semantic_labels: Ground truth semantic labels
            centroid_score_output: Model predictions for centroid scores
            centroid_score_labels: Ground truth centroid labels
            offset_output: Model predictions for centroid offsets
            offset_labels: Ground truth offset vectors
            instance_output: Model predictions for instance segmentation
            instance_labels: Ground truth instance labels
            centroid_confidences: Confidence scores for detected centroids
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        
        loss_sem = self._criterion_semantic(semantic_output.F, semantic_labels.F) * self._cfg.trainer.loss_coeffs.semantic_loss_coeff
        loss_classification = self._criterion_classification(classification_output.F, classification_labels.F) * self._cfg.trainer.loss_coeffs.classification_loss_coeff
        loss_centroid = self._criterion_centroid(centroid_score_output.F, centroid_score_labels.F) * self._cfg.trainer.loss_coeffs.centroid_loss_coeff
        loss_offset_norm, loss_offset_dir = self._criterion_offset(offset_output.F, offset_labels.F, offset_labels.F.shape[0])
        loss_inst, remap_info = self._criterion_instance(instance_output,
                                                         instance_labels,
                                                         centroid_batches=centroid_confidences.C[:, 0])

        loss_offset_norm *= self._cfg.trainer.loss_coeffs.offset_norm_loss_coeff
        loss_offset_dir *= self._cfg.trainer.loss_coeffs.offset_dir_loss_coeff
        total_loss_offset = (loss_offset_norm + loss_offset_dir) * self._cfg.trainer.loss_coeffs.offset_loss_coeff
        total_loss_inst = loss_inst['total_loss'] * self._cfg.trainer.loss_coeffs.instance_loss_coeff
        total_loss = loss_sem + loss_classification + loss_centroid + total_loss_offset + total_loss_inst

        return {
            'total_loss': total_loss,
            'semantic_loss': loss_sem,
            'classification_loss': loss_classification,
            'centroid_loss': loss_centroid,
            'offset_loss': total_loss_offset,
            'offset_norm_loss': loss_offset_norm,
            'offset_dir_loss': loss_offset_dir,
            'instance_loss': total_loss_inst,
            'matched_loss': loss_inst['matched_loss'],
            'unmatched_loss': loss_inst['unmatched_loss'],
            'bce_loss': loss_inst['bce_loss'],
            'dice_loss': loss_inst['dice_loss'],
            'remap_info': remap_info
        }
    
    @torch.no_grad()
    def _freeze_params(self, module: nn.Module, freeze: bool) -> None:
        """
        Freeze or unfreeze parameters of a specific module for staged training.
        Useful for fine-tuning scenarios where certain parts of the network
        should remain fixed during specific training phases.
        
        Args:
            module: Neural network module to freeze/unfreeze
            freeze: If True, freeze parameters; if False, unfreeze them
        """
        for param in module.parameters():
            param.requires_grad = not freeze

    @torch.no_grad()
    def _compute_semantic_metrics(
        self,
        semantic_output: SparseTensor,
        semantic_labels: SparseTensor
    ) -> Dict:
        """
        Compute comprehensive semantic segmentation metrics including IoU,
        precision, recall, and F1-score for each class and overall averages.
        
        Args:
            semantic_output: Model predictions for semantic segmentation
            semantic_labels: Ground truth semantic class labels
            
        Returns:
            Tuple of (IoU, precision, recall, F1) tensors for each class and the confusion matrix
        """
        semantic_iou = self._metric_semantic_iou(semantic_output.F, semantic_labels.F)

        semantic_precision = self._metric_semantic_precision(semantic_output.F, semantic_labels.F)
        semantic_recall = self._metric_semantic_recall(semantic_output.F, semantic_labels.F)
        semantic_f1 = self._metric_semantic_f1(semantic_output.F, semantic_labels.F)

        confmat = self._metric_semantic_confmat(semantic_output.F, semantic_labels.F)

        return semantic_iou, semantic_precision, semantic_recall, semantic_f1, confmat
    
    @torch.no_grad()
    def _compute_classification_metrics(
        self,
        classification_output: SparseTensor,
        classification_labels: SparseTensor
    ) -> Dict:
        """
        Compute comprehensive classification metrics including IoU,
        precision, recall, and F1-score for each class and overall averages.
        
        Args:
            classification_output: Model predictions for classification
            classification_labels: Ground truth classification labels

        Returns:
            Tuple of (IoU, precision, recall, F1) tensors for each class
        """
        classification_iou = self._metric_classification_iou(classification_output.F, classification_labels.F)

        classification_precision = self._metric_classification_precision(classification_output.F, classification_labels.F)
        classification_recall = self._metric_classification_recall(classification_output.F, classification_labels.F)
        classification_f1 = self._metric_classification_f1(classification_output.F, classification_labels.F)

        return classification_iou, classification_precision, classification_recall, classification_f1

    @torch.no_grad()
    def _compute_instance_metrics(
        self,
        instance_output: SparseTensor,
        instance_labels: SparseTensor,
        remap_info: Dict,
        iou_thresh: float = 0.5
    ):
        """
        Compute instance segmentation metrics including true positives, false positives,
        false negatives, precision, recall, and F1-score based on IoU thresholding.
        
        Uses the Hungarian matching results to properly align predicted and ground truth
        instances before computing IoU and determining detection success.
        
        Args:
            instance_output: Model predictions for instance segmentation
            instance_labels: Ground truth instance labels
            remap_info: Hungarian matching results mapping predictions to ground truth
            iou_thresh: IoU threshold for considering a detection as positive
            
        Returns:
            Tuple of (TP, FP, FN, iou.mean(), precision, recall, F1) metrics
        """
        if remap_info['num_instances'] == 0:
            return 0, 0, 0, float('nan'), float('nan'), float('nan'), float('nan')
        
        instance_output_remap = torch.zeros((instance_output.F.size(0), remap_info['num_instances']), device=instance_output.F.device, dtype=instance_output.F.dtype)
        instance_output_remap[:, remap_info['gt_indices']] = instance_output.F[:, remap_info['pred_indices']]

        if remap_info['num_instances'] == 1:
            metric = BinaryJaccardIndex().to(instance_output.F.device)
            iou = torch.tensor([metric(instance_output_remap[:, 0], instance_labels.F)], device=instance_output.F.device)
        else:
            metric = MulticlassJaccardIndex(
                num_classes=remap_info['num_instances'],
                average='none'
            ).to(instance_output.F.device)
            iou = metric(instance_output_remap, instance_labels.F)

        tp_mask = iou >= iou_thresh
        tp = int(tp_mask.sum().item())
        fp = remap_info['num_matches'] - tp
        fn = remap_info['num_instances'] - tp

        precision = tp / remap_info['num_matches'] if remap_info['num_matches'] > 0 else float('nan')
        recall = tp / remap_info['num_instances'] if remap_info['num_instances'] > 0 else float('nan')
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else float('nan')

        return tp, fp, fn, iou.mean(), precision, recall, f1_score

    @torch.no_grad()    
    def _compute_metrics(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            classification_output: SparseTensor,
            classification_labels: SparseTensor,
            centroid_confidences: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: SparseTensor,
            remap_info: Dict,
        ) -> Dict:
        """
        Compute comprehensive evaluation metrics for all tasks including semantic
        segmentation, centroid detection, and instance segmentation performance.
        
        Args:
            semantic_output: Model predictions for semantic segmentation
            semantic_labels: Ground truth semantic labels
            centroid_confidences: Confidence scores for detected centroids
            instance_output: Model predictions for instance segmentation
            instance_labels: Ground truth instance labels
            remap_info: Hungarian matching results from instance loss computation
            
        Returns:
            Dictionary containing all computed metrics for logging and monitoring
        """

        semantic_iou, semantic_precision, semantic_recall, semantic_f1, confmat = self._compute_semantic_metrics(semantic_output, semantic_labels)
        classification_iou, classification_precision, classification_recall, classification_f1 = self._compute_classification_metrics(classification_output, classification_labels)
        instance_tp, instance_fp, instance_fn, iou, instance_precision, instance_recall, instance_f1 = self._compute_instance_metrics(
            instance_output,
            instance_labels,
            iou_thresh=0.5,
            remap_info=remap_info
        )

        return {
            'semantic_iou': semantic_iou.cpu().numpy(),
            # 'mean_semantic_iou': semantic_iou.mean(dim=0).item(),
            'mean_semantic_iou': torch.nanmean(semantic_iou.masked_fill(semantic_iou == 0, float('nan')), dim=0).item(),
            'semantic_precision': semantic_precision.cpu().numpy(),
            # 'mean_semantic_precision': semantic_precision.mean(dim=0).item(),
            'mean_semantic_precision': torch.nanmean(semantic_precision.masked_fill(semantic_precision == 0, float('nan')), dim=0).item(),
            'semantic_recall': semantic_recall.cpu().numpy(),
            # 'mean_semantic_recall': semantic_recall.mean(dim=0).item(),
            'mean_semantic_recall': torch.nanmean(semantic_recall.masked_fill(semantic_recall == 0, float('nan')), dim=0).item(),
            'semantic_f1': semantic_f1.cpu().numpy(),
            # 'mean_semantic_f1': semantic_f1.mean(dim=0).item(),
            'mean_semantic_f1': torch.nanmean(semantic_f1.masked_fill(semantic_f1 == 0, float('nan')), dim=0).item(),
            'semantic_confusion_matrix': confmat.cpu().numpy(),
            'classification_iou': classification_iou.cpu().numpy(),
            # 'mean_classification_iou': classification_iou.mean(dim=0).item(),
            'mean_classification_iou': torch.nanmean(classification_iou.masked_fill(classification_iou == 0, float('nan')), dim=0).item(),
            'classification_precision': classification_precision.cpu().numpy(),
            # 'mean_classification_precision': classification_precision.mean(dim=0).item(),
            'mean_classification_precision': torch.nanmean(classification_precision.masked_fill(classification_precision == 0, float('nan')), dim=0).item(),
            'classification_recall': classification_recall.cpu().numpy(),
            # 'mean_classification_recall': classification_recall.mean(dim=0).item(),
            'mean_classification_recall': torch.nanmean(classification_recall.masked_fill(classification_recall == 0, float('nan')), dim=0).item(),
            'classification_f1': classification_f1.cpu().numpy(),
            # 'mean_classification_f1': classification_f1.mean(dim=0).item(),
            'mean_classification_f1': torch.nanmean(classification_f1.masked_fill(classification_f1 == 0, float('nan')), dim=0).item(),
            'centroids_found': remap_info['num_predictions'],
            'centroids_gt': remap_info['num_instances'],
            'centroids_ratio': remap_info['num_predictions'] / remap_info['num_instances'] if remap_info['num_instances'] > 0 else float('nan'),
            'mean_centroid_confidence': centroid_confidences.F.mean().item() if centroid_confidences.F.numel() > 0 else float('nan'),
            'instances_matched': remap_info['num_matches'],
            'tp': instance_tp,
            'fp': instance_fp,
            'fn': instance_fn,
            'instance_iou': iou.item(),
            'instance_precision': instance_precision,
            'instance_recall': instance_recall,
            'instance_f1': instance_f1
        }
    
    def _forward_pass(self, feed_dict: dict, training: bool):
        """
        Execute a complete forward pass including model inference, loss computation,
        and metric calculation. Handles both training and validation modes with
        appropriate gradient computation and mixed precision.
        
        Args:
            feed_dict: Batch data dictionary containing inputs and labels
            training: Whether this is a training pass (affects gradient computation)
            
        Returns:
            Dictionary containing all outputs, losses, and computed metrics
        """
        inputs = feed_dict["inputs"].to(self._device)
        semantic_labels = feed_dict["semantic_labels"].to(self._device)
        semantic_mask = torch.isin(semantic_labels.F, self._foreground_classes)

        classification_labels = feed_dict["classification_labels"].to(self._device)
        classification_labels.C = classification_labels.C[semantic_mask]
        classification_labels.F = classification_labels.F[semantic_mask] - 1

        centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
        centroid_score_labels.C = centroid_score_labels.C[semantic_mask]
        centroid_score_labels.F = centroid_score_labels.F[semantic_mask]

        offset_labels = feed_dict["offset_labels"].to(self._device)
        offset_labels.C = offset_labels.C[semantic_mask]
        offset_labels.F = offset_labels.F[semantic_mask]

        instance_labels = feed_dict["instance_labels"].to(self._device)
        instance_labels.C = instance_labels.C[semantic_mask]
        instance_labels.F = instance_labels.F[semantic_mask] - 1

        #print(f'min semantic label: {semantic_labels.F.min().item()}, max semantic label: {semantic_labels.F.max().item()}')
        #print(f'min classification label: {classification_labels.F.min().item()}, max classification label: {classification_labels.F.max().item()}')
        #print(f'min instance label: {instance_labels.F.min().item()}, max instance label: {instance_labels.F.max().item()}')

        if training:
            self._optimizer.zero_grad()

        with amp.autocast(enabled=True):
            semantic_output, classification_output, centroid_score_output, offset_output, centroid_confidences_output, instance_output = self._model(inputs, semantic_labels)
            loss = self._compute_loss(
                semantic_output=semantic_output,
                semantic_labels=semantic_labels,
                classification_output=classification_output,
                classification_labels=classification_labels,
                centroid_score_output=centroid_score_output,
                centroid_score_labels=centroid_score_labels,
                offset_output=offset_output,
                offset_labels=offset_labels,
                instance_output=instance_output,
                instance_labels=instance_labels,
                centroid_confidences=centroid_confidences_output
            )

        with torch.no_grad():
            stat = self._compute_metrics(
                semantic_output=semantic_output,
                semantic_labels=semantic_labels,
                classification_output=classification_output,
                classification_labels=classification_labels,
                centroid_confidences=centroid_confidences_output,
                instance_output=instance_output,
                instance_labels=instance_labels,
                remap_info=loss['remap_info']
            )

        return {
            'semantic_labels': semantic_labels,
            'semantic_output': semantic_output,
            'classification_labels': classification_labels,
            'classification_output': classification_output,
            'centroid_score_labels': centroid_score_labels,
            'centroid_score_output': centroid_score_output,
            'centroid_confidences_output': centroid_confidences_output,
            'offset_labels': offset_labels,
            'offset_output': offset_output,
            'instance_labels': instance_labels,
            'instance_output': instance_output,
            'loss': loss,
            'stat': stat
        }

    @torch.no_grad()
    def _log_stats(self, loss: Dict, stat: Dict, step: int, prefix: str) -> None:
        """
        Log all training and validation metrics to TensorBoard for monitoring
        and visualization. Organizes metrics into logical groups for easy analysis.
        
        Args:
            loss: Dictionary containing all loss components
            stat: Dictionary containing all computed metrics
            step: Current global step for logging
            prefix: Prefix for metric names (e.g., 'Train', 'Val')
        """
        self._writer.add_scalar(f'{prefix}_Loss/1/Total_loss', loss['total_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/2/Semantic_loss', loss['semantic_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/3/Classification_loss', loss['classification_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/4/Centroid_loss', loss['centroid_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/5/Offset_loss', loss['offset_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/6/Offset_norm_loss', loss['offset_norm_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/7/Offset_dir_loss', loss['offset_dir_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/8/Instance_loss', loss['instance_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/9/Matched_loss', loss['matched_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/10/Unmatched_loss', loss['unmatched_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/11/Bce_loss', loss['bce_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Loss/12/Dice_loss', loss['dice_loss'].item(), step)
        self._writer.add_scalar(f'{prefix}_Semantic/1/Mean_semantic_IoU', stat['mean_semantic_iou'], step)
        self._writer.add_scalar(f'{prefix}_Semantic/2/Mean_semantic_Precision', stat['mean_semantic_precision'], step)
        self._writer.add_scalar(f'{prefix}_Semantic/3/Mean_semantic_Recall', stat['mean_semantic_recall'], step)
        self._writer.add_scalar(f'{prefix}_Semantic/4/Mean_semantic_F1', stat['mean_semantic_f1'], step)
        self._writer.add_scalar(f'{prefix}_Classification/1/Mean_classification_IoU', stat['mean_classification_iou'], step)
        self._writer.add_scalar(f'{prefix}_Classification/2/Mean_classification_Precision', stat['mean_classification_precision'], step)
        self._writer.add_scalar(f'{prefix}_Classification/3/Mean_classification_Recall', stat['mean_classification_recall'], step)
        self._writer.add_scalar(f'{prefix}_Classification/4/Mean_classification_F1', stat['mean_classification_f1'], step)
        self._writer.add_scalar(f'{prefix}_Centroids/1/Centroids_found_ratio', stat['centroids_found'] / stat['centroids_gt'] if stat['centroids_gt'] > 0 else float('nan'), step)
        self._writer.add_scalar(f'{prefix}_Centroids/2/Mean_centroid_confidence', stat['mean_centroid_confidence'], step)
        self._writer.add_scalar(f'{prefix}_Instance/1/Instances_matched_ratio', stat['instances_matched'] / stat['centroids_gt'] if stat['centroids_gt'] > 0 else float('nan'), step)
        self._writer.add_scalar(f'{prefix}_Instance/2/Instance_Precision', stat['instance_precision'], step)
        self._writer.add_scalar(f'{prefix}_Instance/3/Instance_Recall', stat['instance_recall'], step)
        self._writer.add_scalar(f'{prefix}_Instance/4/Instance_F1', stat['instance_f1'], step)
        self._writer.add_scalar(f'{prefix}_Learning_Rate/1/Backbone_LR', self._optimizer.param_groups[0]['lr'], step)
        self._writer.add_scalar(f'{prefix}_Learning_Rate/2/Semantic_LR', self._optimizer.param_groups[1]['lr'], step)
        self._writer.add_scalar(f'{prefix}_Learning_Rate/3/Classification_LR', self._optimizer.param_groups[2]['lr'], step)
        self._writer.add_scalar(f'{prefix}_Learning_Rate/4/Offset_LR', self._optimizer.param_groups[3]['lr'], step)
        self._writer.add_scalar(f'{prefix}_Learning_Rate/5/Centroid_LR', self._optimizer.param_groups[4]['lr'], step)
        self._writer.add_scalar(f'{prefix}_Learning_Rate/6/Instance_LR', self._optimizer.param_groups[5]['lr'], step)

        confmat = torch.from_numpy(stat['semantic_confusion_matrix']).float()
        confmat_normalized = confmat / (confmat.sum(dim=1, keepdim=True) + 1e-10)

        confmat_img = confmat_normalized.unsqueeze(0)
        self._writer.add_image(f'{prefix}_Semantic/Confusion_Matrix', confmat_img, step, dataformats='CHW')

    @torch.no_grad()
    def _log_mean_stats(self, step: int, stats: dict) -> None:
        header = '| Class | IoU | Precision | Recall | F1 |\n|-|-|-|-|-|\n'
        row = ''
        for i, semantic_class in enumerate(self._cfg.dataset.semantic_classes):
            row += f'| {semantic_class.name} | {stats["semantic_iou"][i]:.3f} | {stats["semantic_precision"][i]:.3f} | {stats["semantic_recall"][i]:.3f} | {stats["semantic_f1"][i]:.3f} |\n'
        row += f'| Mean | {stats["semantic_iou"].mean():.3f} | {stats["semantic_precision"].mean():.3f} | {stats["semantic_recall"].mean():.3f} | {stats["semantic_f1"].mean():.3f} |\n'

        self._writer.add_text('Val_Semantic/Mean_Results', header + row, step)

        header = '| Class | IoU | Precision | Recall | F1 |\n|-|-|-|-|-|\n'
        row = ''
        for i, instance_class in enumerate(self._cfg.dataset.instance_classes):
            row += f'| {instance_class.name} | {stats["classification_iou"][i]:.3f} | {stats["classification_precision"][i]:.3f} | {stats["classification_recall"][i]:.3f} | {stats["classification_f1"][i]:.3f} |\n'
        row += f'| Mean | {stats["classification_iou"].mean():.3f} | {stats["classification_precision"].mean():.3f} | {stats["classification_recall"].mean():.3f} | {stats["classification_f1"].mean():.3f} |\n'

        self._writer.add_text('Val_Classification/Mean_Results', header + row, step)

        header = 'IoU | Precision | Recall | F1 |\n|-|-|-|-|\n'
        row = f'{stats["instance_iou"]:.3f} | {stats["instance_precision"]:.3f} | {stats["instance_recall"]:.3f} | {stats["instance_f1"]:.3f} |\n'

        self._writer.add_text('Val_Instance/Mean_Results', header + row, step)

    @torch.no_grad()
    def _log_pointclouds(self, step: int, result: dict) -> None:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_pred = o3d.geometry.PointCloud()

        batch_idx = result['semantic_output'].C.cpu().numpy()[:, 0]
        centroid_batch_idx = result['centroid_confidences_output'].C.cpu().numpy()[:, 0]

        voxels = result['semantic_output'].C.cpu().numpy()[:, 1:]
        centroid_voxels = result['centroid_confidences_output'].C.cpu().numpy()[:, 1:]

        semantic_output = torch.argmax(result['semantic_output'].F.cpu(), dim=1).numpy()
        semantic_labels = result['semantic_labels'].F.cpu().numpy()

        semantic_mask = np.isin(semantic_labels, self._cfg.model.foreground_classes)
        classification_output = np.zeros((voxels.shape[0],), dtype=int)
        classification_output[semantic_mask] = torch.argmax(result['classification_output'].F.cpu(), dim=1).numpy() + 1
        classification_labels = np.zeros((voxels.shape[0],), dtype=int)
        classification_labels[semantic_mask] = result['classification_labels'].F.cpu().numpy()

        centroid_score_output = np.zeros((voxels.shape[0], 1), dtype=float)
        centroid_score_output[semantic_mask] = result['centroid_score_output'].F.cpu().numpy()
        centroid_score_labels = np.zeros((voxels.shape[0], 1), dtype=float)
        centroid_score_labels[semantic_mask] = result['centroid_score_labels'].F.cpu().numpy()

        centroid_confidence_output = result['centroid_confidences_output'].F.cpu().numpy()

        offset_output = np.zeros((voxels.shape[0], 3), dtype=float)
        offset_output[semantic_mask] = result['offset_output'].F.cpu().numpy()
        offset_labels = np.zeros((voxels.shape[0], 3), dtype=float)
        offset_labels[semantic_mask] = result['offset_labels'].F.cpu().numpy()

        instance_output = np.full(voxels.shape[0], fill_value=-1, dtype=int)
        if result['instance_output'].F.size(1) > 0:
            instance_output[semantic_mask] = torch.argmax(result['instance_output'].F.cpu(), dim=1).numpy()

        gt_indices = result['loss']['remap_info']['gt_indices'].cpu().numpy()
        pred_indices = result['loss']['remap_info']['pred_indices'].cpu().numpy()
        num_instances = result['loss']['remap_info']['num_instances']

        lut = np.full(num_instances, fill_value=-1, dtype=int)
        lut[gt_indices] = pred_indices

        start = pred_indices.max() + 1 if pred_indices.size else 0
        unmatched = np.setdiff1d(np.arange(num_instances), gt_indices)
        lut[unmatched] = np.arange(start, start + unmatched.size)
        instance_labels_tmp = lut[result['instance_labels'].F.cpu().numpy()]

        instance_labels = np.full(voxels.shape[0], fill_value=-1, dtype=int)
        instance_labels[semantic_mask] = instance_labels_tmp

        rng = np.random.default_rng(0)
        palette = []
        reserved_colors = np.array([
            [0.2, 0.2, 0.2],  # Ground
            [1.0, 0.0, 0.0],  # Not matched
        ])

        while len(palette) < len(np.unique(instance_labels)):
            color = rng.random(3)
            if np.all(np.linalg.norm(reserved_colors - color, axis=1) > 0.2):
                palette.append(color)

        palette = np.array(palette)

        id2color = {uid: palette[i] for i, uid in enumerate(np.unique(instance_labels)) if i != -1}
        diff = np.setdiff1d(np.unique(instance_output), np.unique(instance_labels))
        id2color.update({uid: tuple(reserved_colors[1]) for uid in diff})
        id2color[-1] = tuple(reserved_colors[0])

        class_colormap = np.array([semantic_cls.color for semantic_cls in self._cfg.dataset.semantic_classes]).astype(np.float32) / 255.0
        classification_colormap = np.vstack([reserved_colors[0], [instance_cls.color for instance_cls in self._cfg.dataset.instance_classes]]).astype(np.float32) / 255.0

        unique_batches = np.unique(batch_idx)
        for idx in unique_batches:
            mask = batch_idx == idx
            cloud_voxels = voxels[mask]

            cloud_semantic_output = semantic_output[mask]
            cloud_semantic_labels = semantic_labels[mask]

            cloud_classification_output = classification_output[mask]
            cloud_classification_labels = classification_labels[mask]

            cloud_centroid_score_output = centroid_score_output[mask]
            cloud_centroid_score_labels = centroid_score_labels[mask]

            cloud_offset_output = offset_output[mask]
            cloud_offset_labels = offset_labels[mask]

            cloud_instance_output = instance_output[mask]
            cloud_instance_labels = instance_labels[mask]

            mask = centroid_batch_idx == idx

            pcd_gt.points = o3d.utility.Vector3dVector(cloud_voxels)
            pcd_gt.translate((30 / self._cfg.dataset.voxel_size, 0, 0))
            pcd_pred.points = o3d.utility.Vector3dVector(cloud_voxels)

            colors = class_colormap[cloud_semantic_labels]
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            colors = class_colormap[cloud_semantic_output]
            pcd_pred.colors = o3d.utility.Vector3dVector(colors)
            self._writer.add_3d("Semantic Segmentation", to_dict_batch([pcd_pred + pcd_gt]), step * len(unique_batches) + idx)
            # self._writer.add_3d("Val_Semantic/GT_pointcloud", to_dict_batch([pcd_gt]), step)

            colors = classification_colormap[cloud_classification_labels]
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            colors = classification_colormap[cloud_classification_output]
            pcd_pred.colors = o3d.utility.Vector3dVector(colors)
            self._writer.add_3d("Classification Segmentation", to_dict_batch([pcd_pred + pcd_gt]), step * len(unique_batches) + idx)

            cmap = plt.get_cmap('viridis')
            cmap_spheres = plt.get_cmap('inferno')
            colors = cmap(cloud_centroid_score_labels[:, 0])[:, :3]
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            colors = cmap(cloud_centroid_score_output[:, 0])[:, :3]
            pcd_pred.colors = o3d.utility.Vector3dVector(colors)

            self._writer.add_3d("Centroid Prediction", to_dict_batch([pcd_pred + pcd_gt]), step * len(unique_batches) + idx)

            voxels_disp_output = cloud_voxels + cloud_offset_output
            voxels_disp_labels = cloud_voxels + cloud_offset_labels

            pcd_gt.points = o3d.utility.Vector3dVector(voxels_disp_labels)
            pcd_gt.translate((30 / self._cfg.dataset.voxel_size, 0, 0))
            pcd_pred.points = o3d.utility.Vector3dVector(voxels_disp_output)

            colors = class_colormap[cloud_semantic_labels]
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            colors = class_colormap[cloud_semantic_output]
            pcd_pred.colors = o3d.utility.Vector3dVector(colors)

            self._writer.add_3d("Offset Prediction", to_dict_batch([pcd_pred + pcd_gt]), step * len(unique_batches) + idx)

            pcd_gt.points = o3d.utility.Vector3dVector(cloud_voxels)
            pcd_gt.translate((30 / self._cfg.dataset.voxel_size, 0, 0))
            pcd_pred.points = o3d.utility.Vector3dVector(cloud_voxels)

            colors = np.array([id2color[i] for i in cloud_instance_labels], dtype=np.float64)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            colors = np.array([id2color[i] for i in cloud_instance_output], dtype=np.float64)
            pcd_pred.colors = o3d.utility.Vector3dVector(colors)

            self._writer.add_3d("Instance Segmentation", to_dict_batch([pcd_pred + pcd_gt]), step * len(unique_batches) + idx)

    def train(self) -> None:
        """
        Execute the complete training procedure including data loading, multi-stage
        training schedule, learning rate scaling, module freezing, and checkpointing.
        
        Supports configurable training schedules where different stages can have
        different learning rates and frozen modules for sophisticated fine-tuning
        strategies. Saves checkpoints after each epoch and final weights at completion.
        """
        dataset = Panoramix3DDataset(self._cfg.dataset, split='train')
        data_loader = DataLoader(
            dataset,
            batch_size=self._cfg.trainer.batch_size,
            collate_fn=sparse_unique_id_collate_fn,
            shuffle=True,
            num_workers=self._cfg.trainer.num_workers,
            pin_memory=True
        )

        for epoch in range(self._cfg.trainer.start_epoch, self._cfg.trainer.num_epochs):
            self._model.train()
            freeze_modules = [item.module for item in self._cfg.trainer.freeze_modules if item.epochs > epoch]

            self._freeze_params(self._model.encoder, 'backbone' in freeze_modules)
            self._freeze_params(self._model.semantic_head, 'semantic' in freeze_modules)
            self._freeze_params(self._model.classification_head, 'classification' in freeze_modules)
            self._freeze_params(self._model.offset_head, 'offset' in freeze_modules)
            self._freeze_params(self._model.centroid_head, 'centroid' in freeze_modules)
            self._freeze_params(self._model.instance_head, 'instance' in freeze_modules)

            self._bn_scheduler.step()

            print(f'\n=== Starting epoch {epoch + 1} / {self._cfg.trainer.num_epochs} ===')
            print(f' * Freeze modules: {freeze_modules}')

            pbar = tqdm(data_loader, desc='[Train]', file=sys.stdout, dynamic_ncols=True)
            for feed_dict in pbar:
                result = self._forward_pass(feed_dict, training=True)
                pbar.set_postfix({
                    'VRAM': f'{torch.cuda.memory_reserved(self._device) / (1024 ** 3):.2f} GB',
                    'Matched': f"{result['stat']['instances_matched']} / {result['stat']['centroids_gt']}",
                    'TP': f"{result['stat']['tp']} / {result['stat']['centroids_gt']}",
                    'Total': result['loss']['total_loss'].item(),
                    'Semantic': result['loss']['semantic_loss'].item(),
                    'Classification': result['loss']['classification_loss'].item(),
                    'Centroid': result['loss']['centroid_loss'].item(),
                    'Offset': result['loss']['offset_loss'].item(),
                    'Instance': result['loss']['instance_loss'].item()
                })

                self._log_stats(result['loss'], result['stat'], epoch * len(data_loader) + pbar.n, prefix='Train')

                self._scaler.scale(result['loss']['total_loss']).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()

            self._save_ckpt(epoch + 1)
            self._lr_scheduler.step()
            print(f"\n✅ Epoch {epoch + 1} / {self._cfg.trainer.num_epochs} finished.")

            self.eval(epoch)

        self._save_weights()
        print(f"\n✅ Training finished. Weights saved to {self._weights_folder}")

    def eval(self, epoch: int = 0) -> None:
        """
        Execute validation procedure with the trained model in evaluation mode.
        Uses optimized inference settings including fused operations and locality-aware
        memory access for efficient validation on large point clouds.
        
        Yields:
            Iterator of validation results containing predictions and ground truth
            for comprehensive evaluation and analysis.
        """
        dataset = Panoramix3DDataset(self._cfg.dataset, split='val')
        data_loader = DataLoader(
            dataset,
            batch_size=self._cfg.trainer.batch_size,
            collate_fn=sparse_unique_id_collate_fn,
            shuffle=True,
            num_workers=self._cfg.trainer.num_workers,
            pin_memory=True
        )
        
        self._model.eval()
        stats = {
            'semantic_iou': [],
            'semantic_precision': [],
            'semantic_recall': [],
            'semantic_f1': [],
            'classification_iou': [],
            'classification_precision': [],
            'classification_recall': [],
            'classification_f1': [],
            'instance_iou': [],
            'instance_precision': [],
            'instance_recall': [],
            'instance_f1': []
        }

        # enable torchsparse 2.0 inference
        # enable fused and locality-aware memory access optimization
        torchsparse.backends.benchmark = True  # type: ignore
        with torch.no_grad():
            pbar = tqdm(data_loader, desc='[Val]', file=sys.stdout, dynamic_ncols=True)
            for feed_dict in pbar:
                result = self._forward_pass(feed_dict, training=False)
                pbar.set_postfix({
                    'VRAM': f'{torch.cuda.memory_reserved(self._device) / (1024 ** 3):.2f} GB',
                    'Matched': f"{result['stat']['instances_matched']} / {result['stat']['centroids_gt']}",
                    'TP': f"{result['stat']['tp']} / {result['stat']['centroids_gt']}",
                    'Total': result['loss']['total_loss'].item(),
                    'Semantic': result['loss']['semantic_loss'].item(),
                    'Classification': result['loss']['classification_loss'].item(),
                    'Centroid': result['loss']['centroid_loss'].item(),
                    'Offset': result['loss']['offset_loss'].item(),
                    'Instance': result['loss']['instance_loss'].item()
                })

                stats['semantic_iou'].append(result['stat']['semantic_iou'])
                stats['semantic_precision'].append(result['stat']['semantic_precision'])
                stats['semantic_recall'].append(result['stat']['semantic_recall'])
                stats['semantic_f1'].append(result['stat']['semantic_f1'])
                stats['classification_iou'].append(result['stat']['classification_iou'])
                stats['classification_precision'].append(result['stat']['classification_precision'])
                stats['classification_recall'].append(result['stat']['classification_recall'])
                stats['classification_f1'].append(result['stat']['classification_f1'])
                stats['instance_iou'].append(result['stat']['instance_iou'])
                stats['instance_precision'].append(result['stat']['instance_precision'])
                stats['instance_recall'].append(result['stat']['instance_recall'])
                stats['instance_f1'].append(result['stat']['instance_f1'])

                self._log_stats(result['loss'], result['stat'], epoch * len(data_loader) + pbar.n, prefix='Val')
                # self._log_pointclouds(epoch * len(data_loader) + pbar.n, result)

        stats = {k: np.nanmean(np.where(np.array(v) != 0.0, np.array(v), np.nan), axis=0) for k, v in stats.items()}
        # stats = {k: np.array(v).mean(axis=0) for k, v in stats.items()}
        self._log_mean_stats(epoch, stats)
