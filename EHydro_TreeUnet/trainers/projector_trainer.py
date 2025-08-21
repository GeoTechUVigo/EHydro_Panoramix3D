import itertools
import shutil
import torch
import sys

import torchsparse
import pickle
import numpy as np

from typing import List, Optional, Dict, Iterator, Tuple, Union

from torch import nn, Tensor
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex, Precision, Recall, F1Score

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ..datasets import MixedDataset
from ..models import TreeProjector
from ..modules import FocalLoss


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
            centroid_thres: float = 0.1,
            descriptor_dim: int = 16
        ):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tree_projector_dir = Path(tree_projector_dir)
        self._version_name = version_name.removesuffix('.pth')
        self._stat_folder = self._tree_projector_dir / 'stats'
        self._stat_folder.mkdir(parents=True, exist_ok=True)

        self._weights_folder = self._tree_projector_dir / 'weights' / self._version_name
        self._checkpoint_folder = self._weights_folder / 'checkpoints'

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
            batch_size=batch_size, collate_fn=sparse_collate_fn,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self._val_loader = DataLoader(
            self._dataset.val_dataset,
            batch_size=batch_size,
            collate_fn=sparse_collate_fn,
            shuffle=True
        )

        self._semantic_loss_coef = semantic_loss_coef
        self._centroid_loss_coef = centroid_loss_coef
        self._offset_loss_coef = offset_loss_coef
        self._instance_loss_coef = instance_loss_coef

        self._criterion_semantic = nn.CrossEntropyLoss()
        self._criterion_centroid = FocalLoss()
        self._criterion_offset = nn.SmoothL1Loss()
        # self._criterion_instance = nn.CosineEmbeddingLoss(margin=0.2, reduction='mean')
        self._criterion_instance = nn.CrossEntropyLoss()

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
            centroid_thres=centroid_thres,
            descriptor_dim=descriptor_dim
        )

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
    
    @property
    def stats(self) -> Optional[List[Dict]]:
        return self._stats
    
    @property
    def losses(self) -> Optional[List[float]]:
        return self._losses

    def _load_weights(self) -> None:
        self._model.load_state_dict(torch.load(self._weights_folder / f'{self._version_name}_weights.pth'))

    @torch.no_grad()
    def _apply_hungarian(self, logits: SparseTensor, labels: SparseTensor) -> Tensor:
        batch_idx = logits.C[:, 0]
        logits = logits.F
        _, K = logits.shape
        device = logits.device
        remapped = torch.zeros_like(labels.F, dtype=torch.long, device=labels.F.device)

        offset = 1
        for b in torch.unique(batch_idx):
            mask = batch_idx == b
            if not mask.any():
                continue

            labels_b = labels.F[mask]

            uniq = torch.unique(labels_b, sorted=True)
            uniq_fg = uniq[uniq != 0]
            M = uniq_fg.shape[0]

            if (K - 1) == 0 or M == 0:
                continue

            log_p_b = F.log_softmax(logits[mask][:, 1:], dim=-1)
            cost = torch.empty((M, K - 1), device=device)
            for m, g in enumerate(uniq_fg):
                cost[m] = -log_p_b[labels_b == g].mean(0)

            row, col = linear_sum_assignment(cost.detach().cpu())

            for r, c in zip(row, col):
                remapped[mask & (labels.F == uniq_fg[r])] = c + offset

            offset += M

        return remapped

    def _compute_loss(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            centroid_score_output: SparseTensor,
            centroid_score_labels: SparseTensor,
            offset_output: SparseTensor,
            offset_labels: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: Tensor
        ) -> Tensor:
        
        loss_sem = self._criterion_semantic(semantic_output.F, semantic_labels.F)
        loss_centroid = self._criterion_centroid(centroid_score_output.F, centroid_score_labels.F)
        loss_offset = self._criterion_offset(offset_output.F, offset_labels.F)
        loss_inst = self._criterion_instance(instance_output.F, instance_labels)

        '''
        own_mask = (instance_labels.unsqueeze(1) == torch.arange(centroid_descriptors.size(0), device=centroid_descriptors.device).unsqueeze(0))
        sim_matrix_masked = instance_output.clone()
        sim_matrix_masked[own_mask] = -float('inf')
        _, hard_neg_idx = sim_matrix_masked.max(dim=1)

        loss_inst_pos = self._criterion_instance(
            voxel_descriptors,
            centroid_descriptors[instance_labels],
            torch.ones(voxel_descriptors.size(0), device=voxel_descriptors.device)
        )

        loss_inst_neg = self._criterion_instance(
            voxel_descriptors,
            centroid_descriptors[hard_neg_idx],
            -torch.ones(voxel_descriptors.size(0), device=voxel_descriptors.device)
        )

        batches = torch.unique(semantic_output.C[:, 0]).tolist()
        full_centroid_confidences = torch.ones((centroid_confidences.F.size(0) + len(batches), centroid_confidences.F.size(1)), dtype=centroid_confidences.F.dtype, device=centroid_confidences.F.device)
        curr_idx = 1

        for batch in batches:
            batch_mask = centroid_confidences.C[:, 0] == batch

            next_idx = curr_idx + batch_mask.sum().item()
            full_centroid_confidences[curr_idx:next_idx] = centroid_confidences.F[batch_mask]
            curr_idx = next_idx + 1

        conf_pos = full_centroid_confidences[instance_labels]
        conf_neg = full_centroid_confidences[hard_neg_idx]

        loss_inst_pos = (conf_pos * loss_inst_pos).sum() / (conf_pos.sum() + 1e-6)
        loss_inst_neg = (conf_neg * loss_inst_neg).sum() / (conf_neg.sum() + 1e-6)
        loss_inst = loss_inst_pos + loss_inst_neg
        '''

        total_loss = self._semantic_loss_coef * loss_sem + \
                    self._centroid_loss_coef * loss_centroid + \
                    self._offset_loss_coef * loss_offset + \
                    self._instance_loss_coef * loss_inst

        return total_loss, loss_sem, loss_centroid, loss_offset, loss_inst
    
    @torch.no_grad()    
    def _compute_metrics(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: Tensor,
        ) -> Dict:

        iou_semantic = self._metric_semantic_iou(semantic_output.F, semantic_labels.F)

        precision_semantic = self._metric_semantic_precision(semantic_output.F, semantic_labels.F)
        recall_semantic = self._metric_semantic_recall(semantic_output.F, semantic_labels.F)
        f1_semantic = self._metric_semantic_f1(semantic_output.F, semantic_labels.F)
        
        num_instances = instance_output.F.size(1)

        if num_instances == 1:
            miou_instance = torch.tensor([1.0], dtype=iou_semantic.dtype, device=iou_semantic.device)
        else:
            metric_miou_instance = MulticlassJaccardIndex(num_classes=num_instances).to(instance_output.F.device)
            miou_instance = metric_miou_instance(instance_output.F, instance_labels)

        return {
            'iou_semantic': iou_semantic.cpu().numpy(),
            'mean_iou_semantic': iou_semantic.mean(dim=0).item(),
            'precision_semantic': precision_semantic.cpu().numpy(),
            'mean_precision_semantic': precision_semantic.mean(dim=0).item(),
            'recall_semantic': recall_semantic.cpu().numpy(),
            'mean_recall_semantic': recall_semantic.mean(dim=0).item(),
            'f1_semantic': f1_semantic.cpu().numpy(),
            'mean_f1_semantic': f1_semantic.mean(dim=0).item(),
            'mean_iou_instance': miou_instance.item()
        }
    
    def train(self) -> None:
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        scaler = amp.GradScaler(enabled=True)
        start_epoch = 1
        losses = []
        stats = []

        if self._start_on_epoch != 0:
            ckpt = torch.load(self._checkpoint_folder / f'{self._version_name}_checkpoint_epoch_{self._start_on_epoch}.pth', map_location=self._device)
            self._model.load_state_dict(ckpt['model_state_dict'])
            self._model.to(device=self._device)

            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])

            start_epoch = ckpt['epoch'] + 1
            losses = ckpt['losses']
            stats = ckpt['stats']

        self._model.train()
        epoch_iter = range(start_epoch, self._epochs + 1) if self._epochs > 0 else itertools.count(start_epoch)
        for epoch in epoch_iter:
            print(f'Version name: {self._version_name}')
            print(f'\n=== Starting epoch {epoch} ===')
            if epoch < 3:
                print('Training instance correlation with labels instead of predictions by now...\n')
            else:
                print('Training instance correlation with predictions.\n')

            pbar = tqdm(self._train_loader, desc='[Train]', file=sys.stdout)
            for feed_dict in pbar:
                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
                offset_labels = feed_dict["offset_labels"].to(self._device)
                instance_labels = feed_dict["instance_labels"].to(self._device)
                optimizer.zero_grad()
    
                with amp.autocast(enabled=True):
                    if epoch < 3:
                        semantic_output, centroid_score_output, offset_output, _, instance_output = self._model(inputs, centroid_score_labels, offset_labels)
                    else:
                        semantic_output, centroid_score_output, offset_output, _, instance_output = self._model(inputs)

                    instance_labels_remap = self._apply_hungarian(instance_output, instance_labels)

                    loss, loss_sem, loss_centroid, loss_offset, loss_inst = self._compute_loss(
                        semantic_output=semantic_output,
                        semantic_labels=semantic_labels,
                        centroid_score_output=centroid_score_output,
                        centroid_score_labels=centroid_score_labels,
                        offset_output=offset_output,
                        offset_labels=offset_labels,
                        instance_output=instance_output,
                        instance_labels=instance_labels_remap
                    )

                with torch.no_grad():
                    stat = self._compute_metrics(semantic_output, semantic_labels, instance_output, instance_labels_remap)
                    stats.append(stat)
                    losses.append((loss.item(), loss_sem.item(), loss_centroid.item(), loss_offset.item(), loss_inst.item()))
                    instance_output_labels = torch.argmax(instance_output.F, dim=1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'Sem mIoU': f'{stat["mean_iou_semantic"]:.4f}',
                        'centroid loss': f'{loss_centroid.item():.4f}',
                        'offset loss': f'{loss_offset.item():.4f}',
                        'Inst mIoU': f'{stat["mean_iou_instance"]:.4f}',
                        'centroids found': f'{instance_output.F.size(1)} ({len(torch.unique(instance_output_labels))}) / {len(torch.unique(instance_labels_remap))}'
                    })

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            torch.save({
                'epoch': epoch,
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'losses': losses,
                'stats': stats
            }, self._checkpoint_folder / f'{self._version_name}_checkpoint_epoch_{epoch}.pth')

            print(f"\n✅ Epoch {epoch} finished.")
            if self._epochs > 0:
                continue

            ans = input('Continue with next epoch? (s/N): ').strip().lower()
            if ans != 's':
                print("🔴 Training stopped by the user.")
                break

        torch.save(self._model.state_dict(), self._weights_folder / f'{self._version_name}_weights.pth')
        self._stats = stats
        self._losses = losses

        with open(self._stat_folder / 'stats.pkl', 'wb') as f:
            pickle.dump(self._stats, f)
        with open(self._stat_folder / 'losses.pkl', 'wb') as f:
            pickle.dump(self._losses, f)

    def eval(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        self._model.eval()
        losses = []
        stats = []

        # enable torchsparse 2.0 inference
        # enable fused and locality-aware memory access optimization
        torchsparse.backends.benchmark = True  # type: ignore

        with torch.no_grad():
            pbar = tqdm(self._val_loader, desc='[Inference]', file=sys.stdout)
            for feed_dict in pbar:
                semantic_labels_cpu = feed_dict["semantic_labels"].F.numpy()
                centroid_score_labels_cpu = feed_dict["centroid_score_labels"].F.numpy()
                offset_labels_cpu = feed_dict["offset_labels"].F.numpy()
                instance_labels_cpu = feed_dict["instance_labels"].F.numpy()

                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
                offset_labels = feed_dict["offset_labels"].to(self._device)
                instance_labels = feed_dict["instance_labels"].to(self._device)
    
                with amp.autocast(enabled=True):
                    semantic_output, centroid_score_output, offset_output, centroid_confidence_output, instance_output = self._model(inputs)
                    instance_labels_remap = self._apply_hungarian(instance_output, instance_labels)

                    loss, loss_sem, loss_centroid, loss_offset, loss_inst = self._compute_loss(
                        semantic_output=semantic_output,
                        semantic_labels=semantic_labels,
                        centroid_score_output=centroid_score_output,
                        centroid_score_labels=centroid_score_labels,
                        offset_output=offset_output,
                        offset_labels=offset_labels,
                        instance_output=instance_output,
                        instance_labels=instance_labels_remap
                    )

                stat = self._compute_metrics(semantic_output, semantic_labels, instance_output, instance_labels_remap)
                losses.append((loss.item(), loss_sem.item(), loss_centroid.item(), loss_offset.item(), loss_inst.item()))
                stats.append(stat)

                voxels = semantic_output.C.cpu().numpy()
                semantic_output = torch.argmax(semantic_output.F.cpu(), dim=1).numpy()
                centroid_score_output = centroid_score_output.F.cpu().numpy()
                centroid_voxels = centroid_confidence_output.C.cpu().numpy()
                centroid_confidence_output = centroid_confidence_output.F.cpu().numpy()
                instance_output = torch.argmax(instance_output.F, dim=1).cpu().numpy()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'Sem mIoU': f'{stat["mean_iou_semantic"]:.4f}',
                    'centroid loss': f'{loss_centroid.item():.4f}',
                    'offset loss': f'{loss_offset.item():.4f}',
                    'Inst mIoU': f'{stat["mean_iou_instance"]:.4f}',
                    'centroids found': f'({len(np.unique(instance_output))}) / {len(torch.unique(instance_labels_remap))}'
                })

                yield voxels, semantic_output, semantic_labels_cpu, centroid_score_output, centroid_score_labels_cpu, offset_output, offset_labels_cpu, instance_output, instance_labels_cpu, centroid_voxels, centroid_confidence_output
        
        self._stats = stats
        self._losses = losses

        Path('./stats').mkdir(parents=False, exist_ok = True)
        with open('./stats/stats.pkl', 'wb') as f:
            pickle.dump(self._stats, f)
        with open('./stats/losses.pkl', 'wb') as f:
            pickle.dump(self._losses, f)
