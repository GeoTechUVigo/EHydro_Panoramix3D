import itertools
import shutil
import torch

import torchsparse
import pickle
import numpy as np

from typing import List, Optional, Dict, Iterator, Tuple

from torch import nn, Tensor
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
            dataset_folder: str,
            voxel_size: float = 0.2,
            train_pct: float = 0.8,
            data_augmentation_coef: float = 48.0,
            epochs: int = 1,
            feat_keys: List[str] = ['intensity'],
            channels: List[int] = [16, 32, 64, 128],
            latent_dim: int = 256,
            instance_density: float = 0.01,
            centroid_thres: float = 0.1,
            descriptor_dim: int = 16,
            batch_size: int = 1,
            training: bool = True,
            semantic_loss_coef: float = 1.0,
            centroid_loss_coef: float = 1.0,
            instance_loss_coef: float = 1.0,
            output_dir: str = '.',
            version_name: str = 'tree_projector_weights.pth',
            start_on_epoch: int = 0
        ):

        self._dataset = MixedDataset(folder=dataset_folder, voxel_size=voxel_size, train_pct=train_pct, data_augmentation=data_augmentation_coef, feat_keys=feat_keys)
        self._train_loader = DataLoader(self._dataset.train_dataset, batch_size=batch_size, collate_fn=sparse_collate_fn, shuffle=True, num_workers=4, pin_memory=True)
        self._val_loader = DataLoader(self._dataset.val_dataset, batch_size=batch_size, collate_fn=sparse_collate_fn, shuffle=True)

        self._semantic_loss_coef = semantic_loss_coef
        self._centroid_loss_coef = centroid_loss_coef
        self._instance_loss_coef = instance_loss_coef

        self._criterion_semantic = nn.CrossEntropyLoss()
        self._criterion_centroid = FocalLoss()
        self._criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')

        self._version_name = version_name.removesuffix('.pth')
        self._epochs = epochs
        self._start_on_epoch = start_on_epoch

        self._output_dir = Path(output_dir)
        self._stat_folder = self._output_dir / 'stats'
        self._stat_folder.mkdir(parents=True, exist_ok=True)

        self._weights_folder = self._output_dir / 'weights' / self._version_name
        self._checkpoint_folder = self._weights_folder / 'checkpoints'

        if start_on_epoch == 0:
            shutil.rmtree(self._checkpoint_folder, ignore_errors=True)
            self._checkpoint_folder.mkdir(parents=True, exist_ok = True)

        self._stats = None
        self._losses = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TreeProjector(
            in_channels=self._dataset.feat_channels,
            num_classes=self._dataset.num_classes,
            channels=channels,
            latent_dim=latent_dim,
            instance_density=instance_density,
            centroid_thres=centroid_thres,
            descriptor_dim=descriptor_dim
        )

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")

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
    def _apply_hungarian(self, logits: SparseTensor, labels: Tensor) -> Tensor:
        batch_idx = logits.C[:, 0]
        logits = logits.F
        N, K = logits.shape
        device = logits.device
        remapped = torch.zeros_like(labels, dtype=torch.long)

        for b in torch.unique(batch_idx):
            mask = batch_idx == b
            if not mask.any():
                continue

            log_p_b = F.log_softmax(logits[mask][:, 1:], dim=-1)
            labels_b = labels[mask]

            uniq = torch.unique(labels_b, sorted=True)
            uniq = uniq[uniq != 0]
            M = len(uniq)
            if (K - 1) == 0 or M == 0:
                continue

            cost = torch.empty((M, K - 1), device=device)
            for m, g in enumerate(uniq):
                cost[m] = -log_p_b[labels_b == g].mean(0)

            row, col = linear_sum_assignment(cost.detach().cpu())

            for r, c in zip(row, col):
                remapped[mask & (labels == uniq[r])] = c + 1

        return remapped

    def _compute_loss(
            self,
            semantic_output: SparseTensor,
            semantic_labels: SparseTensor,
            centroid_score_output: SparseTensor,
            centroid_score_labels: SparseTensor,
            instance_output: SparseTensor,
            instance_labels: Tensor
        ) -> Tensor:
        
        loss_sem = self._criterion_semantic(semantic_output.F, semantic_labels.F)
        loss_centroid = self._criterion_centroid(centroid_score_output.F, centroid_score_labels.F)

        N, K = instance_output.F.shape
        targets = torch.zeros((N, K), dtype=torch.float32, device=instance_output.F.device)
        targets[torch.arange(N), instance_labels] = 1.0

        prob = torch.sigmoid(instance_output.F)
        loss_bce = self._criterion_bce(instance_output.F, targets)
        loss_dice = (1 - (2 * ( prob * targets).sum(0) + 1e-4) / (prob.sum(0) + targets.sum(0) + 1e-4)).mean()
        
        loss_inst = loss_bce + loss_dice

        return self._semantic_loss_coef * loss_sem + self._centroid_loss_coef * loss_centroid + self._instance_loss_coef * loss_inst

    def _mask_iou(mask_pred: Tensor, mask_gt: Tensor) -> float:
        mask_pred = mask_pred.bool()
        mask_gt = mask_gt.bool()

        intersection = (mask_pred & mask_gt).sum().float()
        union = (mask_pred | mask_gt).sum().float()

        if union == 0:
            return 0.0
        
        return (intersection / union).item()

    @torch.no_grad()    
    def _compute_metrics(
            self,
            semantic_output: Tensor,
            semantic_labels: Tensor,
            instance_output: Tensor,
            instance_labels: Tensor,
            ignore_index: int = None
        ) -> Dict:
        if ignore_index is not None:
            mask = semantic_labels != ignore_index
            semantic_output, semantic_labels = semantic_output[mask], semantic_labels[mask]

        semantic_output = torch.argmax(semantic_output, dim=1)

        C = self._dataset.num_classes
        conf = torch.zeros((C, C), dtype=torch.long, device=semantic_output.device)
        idx = C * semantic_labels + semantic_output
        conf += torch.bincount(idx, minlength=C**2).reshape(C, C)

        TP = conf.diag()
        FP = conf.sum(0) - TP
        FN = conf.sum(1) - TP

        precision = TP.float() / (TP + FP).clamp(min=1)
        recall    = TP.float() / (TP + FN).clamp(min=1)
        f1        = 2 * precision * recall / (precision + recall).clamp(min=1e-6)

        iou = TP.float() / (TP + FP + FN).clamp(min=1)
        miou = iou.mean()

        macroP, macroR, macroF = precision.mean(), recall.mean(), f1.mean()

        microTP = TP.sum()
        microP = microTP.float() / (microTP + FP.sum()).clamp(min=1)
        microR = microTP.float() / (microTP + FN.sum()).clamp(min=1)
        microF = 2 * microP * microR / (microP + microR).clamp(min=1e-6)

        if instance_output.shape[1] <= 1:
            ap_per_class = torch.zeros(C, device=instance_output.device)
            map_val      = torch.tensor(0.0, device=instance_output.device)
        else:
            probs  = torch.sigmoid(instance_output[:, 1:])
            gt_ids = instance_labels.squeeze()
            mask_fg = gt_ids != 0
            probs   = probs[mask_fg]
            gt_ids  = gt_ids[mask_fg]

            C = instance_output.shape[1] - 1
            ap_per_class = torch.zeros(C, device=instance_output.device)
            valid_cls    = torch.zeros(C, dtype=torch.bool, device=instance_output.device)

            for c in range(C):
                gt_mask = (gt_ids == c)
                if not gt_mask.any():
                    continue

                valid_cls[c] = True
                scores = probs[:, c]

                scores_sorted, order = scores.sort(descending=True)
                gt_sorted = gt_mask[order]

                tp = gt_sorted.float()
                fp = 1.0 - tp

                cum_tp = torch.cumsum(tp, dim=0)
                cum_fp = torch.cumsum(fp, dim=0)

                recalls    = cum_tp / gt_mask.sum()
                precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

                precisions_rev = torch.flip(precisions, dims=[0])
                precisions_mon = torch.cummax(precisions_rev, dim=0)[0]
                precisions_mon = torch.flip(precisions_mon, dims=[0])

                ap = torch.trapz(precisions_mon, recalls)
                ap_per_class[c] = ap

            map_val = ap_per_class[valid_cls].mean() if valid_cls.any() else torch.tensor(0.0, device=instance_output.device)


        out_dict = {
            "confusion":             conf.cpu().numpy(),
            "iou_per_class":         iou.cpu().numpy(),
            "miou":                  miou.cpu().numpy(),
            "precision_per_class":   precision.cpu().numpy(),
            "recall_per_class":      recall.cpu().numpy(),
            "f1_per_class":          f1.cpu().numpy(),
            "precision_macro":       macroP.cpu().numpy(),
            "recall_macro":          macroR.cpu().numpy(),
            "f1_macro":              macroF.cpu().numpy(),
            "precision_micro":       microP.cpu().numpy(),
            "recall_micro":          microR.cpu().numpy(),
            "f1_micro":              microF.cpu().numpy(),
            "map":                   map_val.cpu().numpy(),
        }

        return out_dict
    
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
            print(f'\n=== Starting epoch {epoch} ===')
            if epoch == 1:
                print('Training instance correlation with labels instead of predictions by now...\n')
            if epoch == 3:
                print('Training instance correlation with predictions.\n')

            pbar = tqdm(self._train_loader, desc='[Train]')
            for feed_dict in pbar:
                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
                instance_labels = feed_dict["instance_labels"].to(self._device)
                optimizer.zero_grad()
    
                with amp.autocast(enabled=True):
                    if epoch == 1:
                        semantic_output, centroid_score_output, centroid_confidence_output, instance_output = self._model(inputs, centroid_score_labels)
                    else:
                        semantic_output, centroid_score_output, centroid_confidence_output, instance_output = self._model(inputs)
                    
                    instance_labels_remap = self._apply_hungarian(instance_output, instance_labels.F)
                    loss = self._compute_loss(semantic_output, semantic_labels, centroid_score_output, centroid_score_labels, instance_output, instance_labels_remap)
                    stat = self._compute_metrics(semantic_output.F, semantic_labels.F, instance_output.F, instance_labels_remap)

                with torch.no_grad():
                    stats.append(stat)
                    losses.append(loss.item())
                    instance_output_labels = torch.argmax(instance_output.F, dim=1)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mIoU': f'{stat["miou"]:.4f}',
                        'mAP': f'{stat["map"]:.4f}',
                        'centroids found': f'{centroid_confidence_output.F.size(0)} ({len(torch.unique(instance_output_labels))}) / {len(torch.unique(instance_labels.F))}'
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
            pbar = tqdm(self._val_loader, desc='[Inference]')
            for feed_dict in pbar:
                semantic_labels_cpu = feed_dict["semantic_labels"].F.numpy()
                centroid_score_labels_cpu = feed_dict["centroid_score_labels"].F.numpy()
                instance_labels_cpu = feed_dict["instance_labels"].F.numpy()

                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                centroid_score_labels = feed_dict["centroid_score_labels"].to(self._device)
                instance_labels = feed_dict["instance_labels"].to(self._device)
    
                with amp.autocast(enabled=True):
                    semantic_output, centroid_score_output, centroid_confidence_output, instance_output = self._model(inputs)
                    instance_labels_remap = self._apply_hungarian(instance_output, instance_labels.F)
                    loss = self._compute_loss(semantic_output, semantic_labels, centroid_score_output, centroid_score_labels, instance_output, instance_labels_remap)
                    stat = self._compute_metrics(semantic_output.F, semantic_labels.F, instance_output.F, instance_labels_remap)

                losses.append(loss.item())
                stats.append(stat)

                voxels = semantic_output.C.cpu().numpy()
                semantic_output = torch.argmax(semantic_output.F.cpu(), dim=1).numpy()
                centroid_score_output = centroid_score_output.F.cpu().numpy()
                centroid_voxels = centroid_confidence_output.C.cpu().numpy()
                centroid_confidence_output = centroid_confidence_output.F.cpu().numpy()
                instance_output = torch.argmax(instance_output.F.cpu(), dim=1).numpy()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mIoU': f'{stat["miou"]:.4f}',
                    'mAP': f'{stat["map"]:.4f}',
                    'centroids found': f'{centroid_confidence_output.shape[0]} ({len(np.unique(instance_output))}) / {len(np.unique(instance_labels_cpu))}'
                })

                yield voxels, semantic_output, semantic_labels_cpu, centroid_score_output, centroid_score_labels_cpu, instance_output, instance_labels_cpu, centroid_voxels, centroid_confidence_output
        
        self._stats = stats
        self._losses = losses

        Path('./stats').mkdir(parents=False, exist_ok = True)
        with open('./stats/stats.pkl', 'wb') as f:
            pickle.dump(self._stats, f)
        with open('./stats/losses.pkl', 'wb') as f:
            pickle.dump(self._losses, f)
