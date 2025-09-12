import itertools
import shutil
import torch
import sys
import os
import time

import open3d as o3d
import torchsparse
import pickle
import numpy as np

from typing import List, Optional, Dict, Iterator, Tuple, Union, Deque
from collections import deque

from torch import nn, Tensor
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex, Precision, Recall, F1Score

from torchsparse import SparseTensor

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .utils import sparse_unique_id_collate_fn
from ..datasets import MixedDataset
from ..models import TreeProjector
from ..modules import FocalLoss, HungarianInstanceLoss
    

class TreeProjectorTrainer:
    def __init__(
            self,
            tree_projector_dir: str,
            dataset_folder: str,
            version_name: str = 'tree_projector_weights.pth',

            voxel_size: float = 0.2,
            feat_keys: List[str] = ['intensity'],
            centroid_sigma: float = 1.0,
            train_pct: float = 0.8,
            data_augmentation_coef: float = 48.0,
            yaw_range: Tuple[float, float] = (0.0, 360.0),
            tilt_range: Tuple[float, float] = (-5.0, 5.0),
            scale_range: Tuple[float, float] = (0.9, 1.1),

            training: bool = True,
            epochs: int = 1,
            start_on_epoch: int = 0,
            batch_size: int = 1,
            semantic_loss_coef: float = 1.0,
            centroid_loss_coef: float = 1.0,
            offset_loss_coef: float = 1.0,
            instance_loss_coef: float = 1.0,

            resnet_blocks: List[Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]] = [
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, (1, 1, 3), (1, 1, 2)),
            ],
            latent_dim: int = 256,
            instance_density: float = 0.01,
            score_thres: float = 0.1,
            centroid_thres: float = 0.2,
            descriptor_dim: int = 16
        ):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tree_projector_dir = Path(tree_projector_dir)
        self._version_name = version_name.removesuffix('.pth')

        self._weights_folder = self._tree_projector_dir / 'weights' / self._version_name
        self._checkpoint_folder = self._weights_folder / 'checkpoints'
        self._logs_folder = self._weights_folder / 'logs'
        self._logs_folder.mkdir(parents=True, exist_ok=True)

        if start_on_epoch == 0:
            shutil.rmtree(self._checkpoint_folder, ignore_errors=True)
            self._checkpoint_folder.mkdir(parents=True, exist_ok = True)

        self._dataset = MixedDataset(
            folder=self._tree_projector_dir / 'datasets' / dataset_folder,
            voxel_size=voxel_size,
            feat_keys=feat_keys,
            centroid_sigma=centroid_sigma,
            train_pct=train_pct,
            data_augmentation=data_augmentation_coef,
            yaw_range=yaw_range,
            tilt_range=tilt_range,
            scale_range=scale_range
        )

        self._train_loader = DataLoader(
            self._dataset.train_dataset,
            batch_size=batch_size,
            collate_fn=sparse_unique_id_collate_fn,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self._val_loader = DataLoader(
            self._dataset.val_dataset,
            batch_size=batch_size,
            collate_fn=sparse_unique_id_collate_fn,
            shuffle=True
        )

        self._semantic_loss_coef = semantic_loss_coef
        self._centroid_loss_coef = centroid_loss_coef
        self._offset_loss_coef = offset_loss_coef
        self._instance_loss_coef = instance_loss_coef

        self._criterion_semantic = nn.CrossEntropyLoss()
        self._criterion_centroid = FocalLoss()
        self._criterion_offset = nn.SmoothL1Loss()
        self._criterion_instance = HungarianInstanceLoss()
        # self._criterion_instance = nn.CosineEmbeddingLoss(margin=0.2, reduction='mean')
        # self._criterion_instance = nn.CrossEntropyLoss(ignore_index=-1)
        # self._criterion_instance = InstanceVariableKLoss()

        self._metric_semantic_iou = MulticlassJaccardIndex(num_classes=self._dataset.num_classes, average='none').to(device=self._device)
        self._metric_semantic_precision = Precision(task='multiclass', num_classes=self._dataset.num_classes, average='none').to(device=self._device)
        self._metric_semantic_recall = Recall(task='multiclass', num_classes=self._dataset.num_classes, average='none').to(device=self._device)
        self._metric_semantic_f1 = F1Score(task='multiclass', num_classes=self._dataset.num_classes, average='none').to(device=self._device)

        self._epochs = epochs
        self._start_on_epoch = start_on_epoch

        self._stats = None
        self._losses = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TreeProjector(
            in_channels=self._dataset.feat_channels,
            num_classes=self._dataset.num_classes,
            resnet_blocks=resnet_blocks,
            latent_dim=latent_dim,
            instance_density=instance_density,
            score_thres=score_thres,
            centroid_thres=centroid_thres,
            descriptor_dim=descriptor_dim
        )

        self._writer = SummaryWriter(log_dir=self._logs_folder, flush_secs=30)

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")

        print('Resnet generates features at the following scales:')
        scales = [1, 1, 1]
        scales_m = [0, 0, 0]
        out_channels_total = 0
        for i, (_, out_channels, _, strides) in enumerate(resnet_blocks):
            if isinstance(strides, int):
                strides = (strides, strides, strides)

            scales = [scale * stride for scale, stride in zip(scales, strides)]
            scales_m = [voxel_size * scale for scale in scales]

            out_channels_total += out_channels
            print(f'\t* ({scales_m[0]:.1f}, {scales_m[1]:.1f}, {scales_m[2]:.1f}) meters -> {out_channels} feats.')

        scales_m = [3 * scale for scale in scales_m]
        print(f'\nMinimum scene size: ({scales_m[0]:.1f}, {scales_m[1]:.1f}, {scales_m[2]:.1f}) meters')
        print(f'Total channels in backbone: {out_channels_total} -> {latent_dim} in latent space.')
        if not training:
            self._load_weights()

        self._model.to(self._device)

    @property
    def dataset(self) -> MixedDataset:
        return self._dataset

    def _load_weights(self) -> None:
        self._model.load_state_dict(torch.load(self._weights_folder / f'{self._version_name}_weights.pth'))

    '''
    @torch.no_grad()
    def _apply_hungarian(self, logits: SparseTensor, labels: SparseTensor) -> Tensor:
        if logits.F.size(1) == 0:
            return labels

        batch_idx = logits.C[:, 0]
        logits = logits.F
        _, K = logits.shape
        device = logits.device
        remapped = torch.full_like(labels.F, fill_value=-1, dtype=torch.long, device=labels.F.device)

        offset = 0
        for b in torch.unique(batch_idx):
            mask = batch_idx == b
            if not mask.any():
                continue

            labels_b = labels.F[mask]

            uniq = torch.unique(labels_b, sorted=True)
            # uniq_fg = uniq[uniq != 0]
            M = uniq.shape[0]

            if (K) == 0 or M == 0:
                continue

            log_p_b = F.log_softmax(logits[mask], dim=1)
            cost = torch.empty((M, K), device=device)
            for m, g in enumerate(uniq):
                cost[m] = -log_p_b[labels_b == g].mean(0)

            row, col = linear_sum_assignment(cost.detach().cpu())
            used_rows = set(row)

            for r, c in zip(row, col):
                remapped[mask & (labels.F == uniq[r])] = c + offset

            offset += M

        return remapped
    '''

    def _compute_loss(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            centroid_score_output: SparseTensor,
            centroid_score_labels: SparseTensor,
            offset_output: SparseTensor,
            offset_labels: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: SparseTensor
        ) -> Tensor:
        
        loss_sem = self._criterion_semantic(semantic_output.F, semantic_labels.F) * self._semantic_loss_coef
        loss_centroid = self._criterion_centroid(centroid_score_output.F, centroid_score_labels.F) * self._centroid_loss_coef
        loss_offset = self._criterion_offset(offset_output.F, offset_labels.F) * self._offset_loss_coef
        loss_inst, remap_info = self._criterion_instance(instance_output.F, instance_labels.F)
        loss_inst = loss_inst * self._instance_loss_coef

        total_loss = loss_sem + loss_centroid + loss_offset + loss_inst

        return {
            'total_loss': total_loss,
            'semantic_loss': loss_sem,
            'centroid_loss': loss_centroid,
            'offset_loss': loss_offset,
            'instance_loss': loss_inst,
            'remap_info': remap_info
        }
    
    @torch.no_grad()
    def _compute_semantic_metrics(
        self,
        semantic_output: SparseTensor,
        semantic_labels: SparseTensor
    ) -> Dict:
        semantic_iou = self._metric_semantic_iou(semantic_output.F, semantic_labels.F)

        semantic_precision = self._metric_semantic_precision(semantic_output.F, semantic_labels.F)
        semantic_recall = self._metric_semantic_recall(semantic_output.F, semantic_labels.F)
        semantic_f1 = self._metric_semantic_f1(semantic_output.F, semantic_labels.F)

        return semantic_iou, semantic_precision, semantic_recall, semantic_f1
    
    @torch.no_grad()
    def _compute_instance_metrics(
        self,
        instance_output: SparseTensor,
        instance_labels: SparseTensor,
        remap_info: Dict,
        iou_thresh: float = 0.5
    ):
        if remap_info['num_instances'] == 0:
            return 0, 0, 0, float('nan'), float('nan'), float('nan')
        
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
        fp = remap_info['num_predictions'] - tp
        fn = remap_info['num_instances'] - tp

        precision = tp / remap_info['num_predictions'] if remap_info['num_predictions'] > 0 else float('nan')
        recall = tp / remap_info['num_instances'] if remap_info['num_instances'] > 0 else float('nan')
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else float('nan')

        return tp, fp, fn, precision, recall, f1_score

    @torch.no_grad()    
    def _compute_metrics(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            centroid_confidences: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: SparseTensor,
            remap_info: Dict,
        ) -> Dict:

        semantic_iou, semantic_precision, semantic_recall, semantic_f1 = self._compute_semantic_metrics(semantic_output, semantic_labels)
        instance_tp, instance_fp, instance_fn, instance_precision, instance_recall, instance_f1 = self._compute_instance_metrics(
            instance_output,
            instance_labels,
            iou_thresh=0.0,
            remap_info=remap_info
        )

        return {
            'semantic_iou': semantic_iou.cpu().numpy(),
            'mean_semantic_iou': semantic_iou.mean(dim=0).item(),
            'semantic_precision': semantic_precision.cpu().numpy(),
            'mean_semantic_precision': semantic_precision.mean(dim=0).item(),
            'semantic_recall': semantic_recall.cpu().numpy(),
            'mean_semantic_recall': semantic_recall.mean(dim=0).item(),
            'semantic_f1': semantic_f1.cpu().numpy(),
            'mean_semantic_f1': semantic_f1.mean(dim=0).item(),
            'centroids_found': remap_info['num_predictions'],
            'centroids_gt': remap_info['num_instances'],
            'centroids_ratio': remap_info['num_predictions'] / remap_info['num_instances'] if remap_info['num_instances'] > 0 else float('nan'),
            'mean_centroid_confidence': centroid_confidences.F.mean().item() if centroid_confidences.F.numel() > 0 else float('nan'),
            'instances_matched': len(torch.unique(instance_output.F.argmax(dim=1))) if instance_output.F.size(1) > 0 else 0,
            'tp': instance_tp,
            'fp': instance_fp,
            'fn': instance_fn,
            'instance_precision': instance_precision,
            'instance_recall': instance_recall,
            'instance_f1': instance_f1
        }
    
    @torch.no_grad()
    def _compute_slope(self, values: deque, x_range: Optional[np.ndarray] = None) -> float:
        if len(values) < 2:
            return 0.0
        
        y = np.array(values)
        x = x_range if x_range is not None else np.arange(len(values))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    def train(self) -> None:
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        scaler = amp.GradScaler(enabled=True)
        start_epoch = 0
        
        window_size = 50
        loss_windows = {
            'total_loss': deque(maxlen=window_size),
            'semantic_loss': deque(maxlen=window_size),
            'centroid_loss': deque(maxlen=window_size),
            'offset_loss': deque(maxlen=window_size),
            'instance_loss': deque(maxlen=window_size)
        }

        if self._start_on_epoch != 0:
            ckpt = torch.load(self._checkpoint_folder / f'{self._version_name}_checkpoint_epoch_{self._start_on_epoch}.pth', map_location=self._device)
            self._model.load_state_dict(ckpt['model_state_dict'])
            self._model.to(device=self._device)

            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])

            start_epoch = ckpt['epoch'] + 1

        self._model.train()
        epoch_iter = range(start_epoch, self._epochs) if self._epochs > 0 else itertools.count(start_epoch)
        for epoch in epoch_iter:
            print(f'Version name: {self._version_name}')
            print(f'\n=== Starting epoch {epoch + 1} ===')
            #if epoch < 3:
            #    print('Training instance correlation with labels instead of predictions by now...\n')
            #else:
            #    print('Training instance correlation with predictions.\n')

            pbar = tqdm(self._train_loader, desc='[Train]', file=sys.stdout, dynamic_ncols=True)
            for feed_dict in pbar:
                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                semantic_mask = semantic_labels.F != 0

                centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
                centroid_score_labels.C = centroid_score_labels.C[semantic_mask]
                centroid_score_labels.F = centroid_score_labels.F[semantic_mask]

                offset_labels = feed_dict["offset_labels"].to(self._device)
                offset_labels.C = offset_labels.C[semantic_mask]
                offset_labels.F = offset_labels.F[semantic_mask]

                instance_labels = feed_dict["instance_labels"].to(self._device)
                instance_labels.C = instance_labels.C[semantic_mask]
                instance_labels.F = instance_labels.F[semantic_mask] - 1

                optimizer.zero_grad()
    
                with amp.autocast(enabled=True):
                    #if epoch < 3:
                    #    semantic_output, centroid_score_output, offset_output, _, instance_output = self._model(inputs, semantic_labels, centroid_score_labels, offset_labels)
                    #else:
                    semantic_output, centroid_score_output, offset_output, centroid_confidences_output, instance_output = self._model(inputs, semantic_labels)
                    loss = self._compute_loss(
                        semantic_output=semantic_output,
                        semantic_labels=semantic_labels,
                        centroid_score_output=centroid_score_output,
                        centroid_score_labels=centroid_score_labels,
                        offset_output=offset_output,
                        offset_labels=offset_labels,
                        instance_output=instance_output,
                        instance_labels=instance_labels
                    )

                with torch.no_grad():
                    stat = self._compute_metrics(
                        semantic_output,
                        semantic_labels,
                        centroid_confidences_output,
                        instance_output,
                        instance_labels,
                        loss['remap_info']
                    )

                    loss_windows['total_loss'].append(loss['total_loss'].item())
                    loss_windows['semantic_loss'].append(loss['semantic_loss'].item())
                    loss_windows['centroid_loss'].append(loss['centroid_loss'].item())
                    loss_windows['offset_loss'].append(loss['offset_loss'].item())
                    loss_windows['instance_loss'].append(loss['instance_loss'].item())

                    slopes = {
                        k: self._compute_slope(v) for k, v in loss_windows.items()
                    }

                    loss_trends = {
                        k: f"{v.item():.4f} {'→' if slopes[k] < 1e-4 and slopes[k] > -1e-4 else ('↑' if slopes[k] > 0 else '↓')}" for k, v in loss.items()
                        for k, v in loss.items() if k != 'remap_info'
                    }

                    pbar.set_postfix({
                        'VRAM': f'{torch.cuda.memory_reserved(self._device) / (1024 ** 3):.2f} GB',
                        'Total': loss_trends['total_loss'],
                        'Semantic': loss_trends['semantic_loss'],
                        'Centroid': loss_trends['centroid_loss'],
                        'Offset': loss_trends['offset_loss'],
                        'Instance': loss_trends['instance_loss']
                    })

                    step = epoch * len(self._train_loader) + pbar.n
                    self._writer.add_scalar('Train_Loss/Total_loss', loss['total_loss'].item(), step)
                    self._writer.add_scalar('Train_Loss/Semantic_loss', loss['semantic_loss'].item(), step)
                    self._writer.add_scalar('Train_Loss/Centroid_loss', loss['centroid_loss'].item(), step)
                    self._writer.add_scalar('Train_Loss/Offset_loss', loss['offset_loss'].item(), step)
                    self._writer.add_scalar('Train_Loss/Instance_loss', loss['instance_loss'].item(), step)
                    self._writer.add_scalar('Train_Semantic/Mean_semantic_IoU', stat['mean_semantic_iou'], step)
                    self._writer.add_scalar('Train_Semantic/Mean_semantic_Precision', stat['mean_semantic_precision'], step)
                    self._writer.add_scalar('Train_Semantic/Mean_semantic_Recall', stat['mean_semantic_recall'], step)
                    self._writer.add_scalar('Train_Semantic/Mean_semantic_F1', stat['mean_semantic_f1'], step)
                    self._writer.add_scalar('Train_Centroids/Centroids_found_ratio', stat['centroids_found'] / stat['centroids_gt'] if stat['centroids_gt'] > 0 else float('nan'), step)
                    self._writer.add_scalar('Train_Centroids/Mean_centroid_confidence', stat['mean_centroid_confidence'], step)
                    self._writer.add_scalar('Train_Instance/Instances_matched_ratio', stat['instances_matched'] / stat['centroids_gt'] if stat['centroids_gt'] > 0 else float('nan'), step)
                    self._writer.add_scalar('Train_Instance/Instance_Precision', stat['instance_precision'], step)
                    self._writer.add_scalar('Train_Instance/Instance_Recall', stat['instance_recall'], step)
                    self._writer.add_scalar('Train_Instance/Instance_F1', stat['instance_f1'], step)

                scaler.scale(loss['total_loss']).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.save({
                'epoch': epoch,
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, self._checkpoint_folder / f'{self._version_name}_checkpoint_epoch_{epoch}.pth')

            print(f"\n✅ Epoch {epoch + 1} finished.")
            if self._epochs > 0:
                continue

            ans = input('Continue with next epoch? (s/N): ').strip().lower()
            if ans != 's':
                print("🔴 Training stopped by the user.")
                break

        torch.save(self._model.state_dict(), self._weights_folder / f'{self._version_name}_weights.pth')

    def eval(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        self._model.eval()

        # enable torchsparse 2.0 inference
        # enable fused and locality-aware memory access optimization
        torchsparse.backends.benchmark = True  # type: ignore
        with torch.no_grad():
            window_size = 50
            loss_windows = {
                'total_loss': deque(maxlen=window_size),
                'semantic_loss': deque(maxlen=window_size),
                'centroid_loss': deque(maxlen=window_size),
                'offset_loss': deque(maxlen=window_size),
                'instance_loss': deque(maxlen=window_size)
            }

            pbar = tqdm(self._train_loader, desc='[Val]', file=sys.stdout, dynamic_ncols=True, bar_format="{l_bar}{bar}{r_bar}\n{postfix}")
            for feed_dict in pbar:
                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                semantic_mask = semantic_labels.F != 0

                centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
                centroid_score_labels.C = centroid_score_labels.C[semantic_mask]
                centroid_score_labels.F = centroid_score_labels.F[semantic_mask]

                offset_labels = feed_dict["offset_labels"].to(self._device)
                offset_labels.C = offset_labels.C[semantic_mask]
                offset_labels.F = offset_labels.F[semantic_mask]

                instance_labels = feed_dict["instance_labels"].to(self._device)
                instance_labels.C = instance_labels.C[semantic_mask]
                instance_labels.F = instance_labels.F[semantic_mask] - 1
    
                with amp.autocast(enabled=True):
                    semantic_output, centroid_score_output, offset_output, centroid_confidences_output, instance_output = self._model(inputs, semantic_labels)
                    loss = self._compute_loss(
                        semantic_output=semantic_output,
                        semantic_labels=semantic_labels,
                        centroid_score_output=centroid_score_output,
                        centroid_score_labels=centroid_score_labels,
                        offset_output=offset_output,
                        offset_labels=offset_labels,
                        instance_output=instance_output,
                        instance_labels=instance_labels
                    )

                stat = self._compute_metrics(
                    semantic_output,
                    semantic_labels,
                    centroid_confidences_output,
                    instance_output,
                    instance_labels,
                    loss['remap_info']
                )

                loss_windows['total_loss'].append(loss['total_loss'].item())
                loss_windows['semantic_loss'].append(loss['semantic_loss'].item())
                loss_windows['centroid_loss'].append(loss['centroid_loss'].item())
                loss_windows['offset_loss'].append(loss['offset_loss'].item())
                loss_windows['instance_loss'].append(loss['instance_loss'].item())

                slopes = {
                    k: self._compute_slope(v) for k, v in loss_windows.items()
                }

                loss_trends = {
                    k: f"{v.item():.4f} {'→' if slopes[k] < 1e-4 and slopes[k] > -1e-4 else ('↑' if slopes[k] > 0 else '↓')}" for k, v in loss.items()
                    for k, v in loss.items() if k != 'remap_info'
                }

                pbar.set_postfix({
                    'Total': loss_trends['total_loss'],
                    'Semantic': loss_trends['semantic_loss'],
                    'Centroid': loss_trends['centroid_loss'],
                    'Offset': loss_trends['offset_loss'],
                    'Instance': loss_trends['instance_loss']
                })

                step = pbar.n
                self._writer.add_scalar('Val_Loss/Total_loss', loss['total_loss'].item(), step)
                self._writer.add_scalar('Val_Loss/Semantic_loss', loss['semantic_loss'].item(), step)
                self._writer.add_scalar('Val_Loss/Centroid_loss', loss['centroid_loss'].item(), step)
                self._writer.add_scalar('Val_Loss/Offset_loss', loss['offset_loss'].item(), step)
                self._writer.add_scalar('Val_Loss/Instance_loss', loss['instance_loss'].item(), step)
                self._writer.add_scalar('Val_Semantic/Mean_semantic_IoU', stat['mean_semantic_iou'], step)
                self._writer.add_scalar('Val_Semantic/Mean_semantic_Precision', stat['mean_semantic_precision'], step)
                self._writer.add_scalar('Val_Semantic/Mean_semantic_Recall', stat['mean_semantic_recall'], step)
                self._writer.add_scalar('Val_Semantic/Mean_semantic_F1', stat['mean_semantic_f1'], step)
                self._writer.add_scalar('Val_Centroids/Centroids_found_ratio', stat['centroids_found'] / stat['centroids_gt'] if stat['centroids_gt'] > 0 else float('nan'), step)
                self._writer.add_scalar('Val_Centroids/Mean_centroid_confidence', stat['mean_centroid_confidence'], step)
                self._writer.add_scalar('Val_Instance/Instances_matched_ratio', stat['instances_matched'] / stat['centroids_gt'] if stat['centroids_gt'] > 0 else float('nan'), step)
                self._writer.add_scalar('Val_Instance/Instance_Precision', stat['instance_precision'], step)
                self._writer.add_scalar('Val_Instance/Instance_Recall', stat['instance_recall'], step)
                self._writer.add_scalar('Val_Instance/Instance_F1', stat['instance_f1'], step)

                yield {
                    'semantic_output': semantic_output,
                    'semantic_labels': semantic_labels,
                    'centroid_score_output': centroid_score_output,
                    'centroid_score_labels': centroid_score_labels,
                    'centroid_confidence_output': centroid_confidences_output,
                    'offset_output': offset_output,
                    'offset_labels': offset_labels,
                    'instance_output': instance_output,
                    'instance_labels': instance_labels,
                    'remap_info': loss['remap_info']
                }
