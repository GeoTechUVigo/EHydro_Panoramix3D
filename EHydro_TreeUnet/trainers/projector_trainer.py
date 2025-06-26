import torch
import torchsparse
import pickle
import numpy as np

from torch import nn
from torch.cuda import amp
from torch.nn import functional as tF
from torch.utils.data import DataLoader

from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ..datasets import MixedDataset
from ..models import TreeProjector


class TreeProjectorTrainer:
    def __init__(self, dataset_folder, voxel_size = 0.2, train_pct = 0.8, data_augmentation_coef = 1.0, feat_keys = ['intensity'], max_instances = 64, channels = [16, 32, 64, 128], latent_dim = 256, batch_size = 1, training = True, semantic_loss_coef = 1.0, instance_loss_coef = 1.0):
        F.set_kmap_mode("hashmap")

        self._stats = None
        self._losses = None
        self._dataset = MixedDataset(folder=dataset_folder, voxel_size=voxel_size, train_pct=train_pct, data_augmentation=data_augmentation_coef, feat_keys=feat_keys)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TreeProjector(self._dataset.feat_channels, self._dataset.num_classes, max_instances, channels = channels, latent_dim = latent_dim)
        # self._model = TreeUNet(self._dataset.feat_channels, self._dataset.num_classes, base_channels=16, depth=4)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self._model = nn.DataParallel(self._model)
        
        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")

        self._train_loader = DataLoader(self._dataset.train_dataset, batch_size=batch_size, collate_fn=sparse_collate_fn, shuffle=True)
        self._val_loader = DataLoader(self._dataset.val_dataset, batch_size=batch_size, collate_fn=sparse_collate_fn, shuffle=True)

        self._criterion_semantic = nn.CrossEntropyLoss()
        self._criterion_instance = nn.CrossEntropyLoss()

        self._semantic_loss_coef = semantic_loss_coef
        self._instance_loss_coef = instance_loss_coef

        if not training:
            self._load_weights()

        self._model.to(self._device)

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def stats(self):
        return self._stats
    
    @property
    def losses(self):
        return self._losses

    def _load_weights(self):
        self._model.load_state_dict(torch.load('./weights/tree_unet_weights.pth'))

    @torch.no_grad()
    def _apply_hungarian(self, logits, labels):
        batch_idx = logits.C[:, 0]
        logits = logits.F
        K = logits.size(1)
        device = logits.device
        remapped = torch.full_like(labels, -1)

        for b in torch.unique(batch_idx):
            mask = batch_idx == b
            if not mask.any():
                continue

            log_p_b = tF.log_softmax(logits[mask], dim=-1)
            labels_b = labels[mask]

            uniq = torch.unique(labels_b, sorted=True)
            M = len(uniq)

            cost = torch.empty((M, K), device=device)
            for m, g in enumerate(uniq):
                cost[m] = -log_p_b[labels_b == g].mean(0)

            row, col = linear_sum_assignment(cost.detach().cpu())

            for r, c in zip(row, col):
                remapped[mask & (labels == uniq[r])] = c

        return remapped
    
    def _criterion_instance_1d(self, instance_output, instance_labels, delta_v=0.1, delta_d=2.0, alpha=1.0, beta=1.0, gamma=1e-3):
        device = instance_output.device
        uniq = instance_labels.unique()
        if len(uniq) == 0:
            return instance_output.new_tensor(0.)

        mu = []
        L_var = instance_output.new_tensor(0.)
        for k in uniq:
            mask = instance_labels == k
            e_k  = instance_output[mask]
            mu_k = e_k.mean()
            mu.append(mu_k)
            L_var += ((torch.relu(torch.abs(e_k - mu_k) - delta_v)) ** 2).mean()
        L_var /= len(uniq)

        mu = torch.stack(mu)
        if len(mu) > 1:
            pdist = torch.abs(mu.unsqueeze(0) - mu.unsqueeze(1))
            L_dist = (torch.relu(delta_d - pdist - torch.eye(len(mu), device=device) * 1e5) ** 2).sum() / (len(mu)*(len(mu)-1))
        else:
            L_dist = instance_output.new_tensor(0.)

        L_reg = torch.abs(mu).mean()
        return alpha * L_var + beta * L_dist + gamma * L_reg
    
    def _compute_loss(self, semantic_output, semantic_labels, instance_output = 0, instance_labels = 0):
        loss_sem = self._criterion_semantic(semantic_output.F, semantic_labels.F)
        loss_inst = self._criterion_instance(instance_output.F, self._apply_hungarian(instance_output, instance_labels.F))
        # loss_inst = self._criterion_instance(instance_output, instance_labels)
        # loss_inst = 0

        return self._semantic_loss_coef * loss_sem + self._instance_loss_coef * loss_inst
    
    @torch.no_grad()    
    def _compute_metrics(self, pred_labels, gt_labels, num_classes, ignore_index = None):
        if ignore_index is not None:
            mask = gt_labels != ignore_index
            pred_labels, gt_labels = pred_labels[mask], gt_labels[mask]

        pred_labels = torch.argmax(pred_labels, dim=1)

        C = num_classes
        conf = torch.zeros((C, C), dtype=torch.long, device=pred_labels.device)
        idx = C * gt_labels + pred_labels
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

        return {
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
        }
    
    def train(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        scaler = amp.GradScaler(enabled=True)
        losses = []
        stats = []

        pbar = tqdm(self._train_loader, desc='[Train]')
        for feed_dict in pbar:
            inputs = feed_dict["inputs"].to(self._device)
            semantic_labels = feed_dict["semantic_labels"].to(self._device)
            instance_labels = feed_dict["instance_labels"].to(self._device)

            with amp.autocast(enabled=True):
                semantic_output, instance_output = self._model(inputs)
                # semantic_output = self._model(inputs)
                # loss = self._compute_loss(semantic_output, semantic_labels)
                loss = self._compute_loss(semantic_output, semantic_labels, instance_output, instance_labels)
                stat = self._compute_metrics(semantic_output.F, semantic_labels.F, num_classes=self._dataset.num_classes)

            stats.append(stat)
            losses.append(loss.item())
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{stat["miou"]:.4f}'
            })

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            del inputs, semantic_output, semantic_labels

        Path('./weights').mkdir(parents=False, exist_ok = True)
        torch.save(self._model.state_dict(), './weights/tree_projector_weights.pth')
        self._stats = stats
        self._losses = losses

        Path('./stats').mkdir(parents=False, exist_ok = True)
        with open('./stats/stats.pkl', 'wb') as f:
            pickle.dump(self._stats, f)
        with open('./stats/losses.pkl', 'wb') as f:
            pickle.dump(self._losses, f)

    def eval(self):
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
                instance_labels_cpu = feed_dict["instance_labels"].F.numpy()
                # coords = feed_dict["coords"].numpy()
                # inverse_map = feed_dict["inverse_map"].numpy()

                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                instance_labels = feed_dict["instance_labels"].to(self._device)

                with amp.autocast(enabled=True):
                    semantic_output, instance_output = self._model(inputs)
                    # semantic_output = self._model(inputs)
                    # loss = self._compute_loss(semantic_output, semantic_labels)
                    loss = self._compute_loss(semantic_output, semantic_labels, instance_output, instance_labels)
                    stat = self._compute_metrics(semantic_output.F, semantic_labels.F, num_classes=self._dataset.num_classes)

                losses.append(loss.item())
                stats.append(stat)

                voxels = semantic_output.C.cpu().numpy()
                semantic_output = torch.argmax(semantic_output.F.cpu(), dim=1).numpy()
                instance_output = torch.argmax(instance_output.F.cpu(), dim=1).numpy()
                # instance_output = np.zeros(semantic_output.shape)
                # instance_output = instance_output.F.cpu().reshape(-1, 1).numpy()
                # print(instance_output)
                # instance_output = DBSCAN(eps=1.0, min_samples=10, metric='euclidean').fit(instance_output).labels_
                # print(instance_output)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mIoU': f'{stat["miou"]:.4f}'
                })

                yield voxels, semantic_output, instance_output, semantic_labels_cpu, instance_labels_cpu #, coords, inverse_map
        
        self._stats = stats
        self._losses = losses

        with open('./stats/stats.pkl', 'wb') as f:
            pickle.dump(self._stats, f)
        with open('./stats/losses.pkl', 'wb') as f:
            pickle.dump(self._losses, f)
