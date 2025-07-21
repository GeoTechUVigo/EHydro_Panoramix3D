import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32  = False

import torchsparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.cuda import amp
from torch.nn import functional as tF
from torch.utils.data import DataLoader

from torchsparse.utils.collate import sparse_collate_fn

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ..datasets import MixedDataset
from ..models import TreeProjector


class TreeProjectorTrainer:
    def __init__(
            self,
            dataset_folder,
            voxel_size = 0.2,
            train_pct = 0.8,
            data_augmentation_coef = 1.0,
            feat_keys = ['intensity'],
            max_instances = 64,
            channels = [16, 32, 64, 128],
            latent_dim = 256,
            batch_size = 1,
            training = True,
            semantic_loss_coef = 1.0,
            offset_loss_coef = 1.0,
            instance_loss_coef = 1.0,
            cosine_loss_coef = 1.0,
            weights_file = 'tree_projector_weights.pth'
        ):

        print(f'tf32 enabled: {torch.backends.cuda.matmul.allow_tf32}')

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
        self._criterion_offset_magnitude = nn.L1Loss()
        self._criterion_offset_cosine = nn.CosineSimilarity()

        self._semantic_loss_coef = semantic_loss_coef
        self._offset_loss_coef = offset_loss_coef
        self._instance_loss_coef = instance_loss_coef
        self._cosine_loss_coef = cosine_loss_coef

        self._weights_file = weights_file

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
        self._model.load_state_dict(torch.load(Path('./weights') / self._weights_file))

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
    
    def _compute_loss(self, semantic_output, semantic_labels, instance_output = 0, instance_labels = 0, offset_output = 0, offset_labels = 0):
        loss_sem = self._criterion_semantic(semantic_output.F, semantic_labels.F)
        loss_inst = self._criterion_instance(instance_output.F, self._apply_hungarian(instance_output, instance_labels.F))

        with torch.no_grad():
            semantic_prediction = torch.argmax(semantic_output.F, dim=1)
            mask_pred = (semantic_prediction == 0)
            mask_gt = (semantic_labels.F == 0)

        idx_pred = mask_pred.nonzero(as_tuple=False).squeeze(1)
        idx_gt = mask_gt.nonzero(as_tuple=False).squeeze(1)

        offset_labels.F[idx_pred] = 0
        offset_labels.F[idx_gt] = 0
        
        loss_offset_magnitude = self._criterion_offset_magnitude(offset_output.F, offset_labels.F)
        loss_offset_cos = (1.0 - self._criterion_offset_cosine(offset_output.F, offset_labels.F)).mean()

        loss_offset = loss_offset_magnitude + self._cosine_loss_coef * loss_offset_cos

        return self._semantic_loss_coef * loss_sem + self._offset_loss_coef * loss_offset + self._instance_loss_coef * loss_inst
    
    def _mask_iou(mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> float:
        mask_pred = mask_pred.bool()
        mask_gt = mask_gt.bool()

        intersection = (mask_pred & mask_gt).sum().float()
        union = (mask_pred | mask_gt).sum().float()

        if union == 0:
            return 0.0
        
        return (intersection / union).item()

    @torch.no_grad()    
    def _compute_metrics(self, semantic_output, semantic_labels, instance_output = None, instance_labels = None, ignore_index = None):
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
        }

        if instance_output is None or instance_labels is None:
            return out_dict
        
        device = semantic_output.device
        ap_per_class = torch.zeros(C, dtype=torch.float32, device=device)
        valid = torch.zeros(C, dtype=torch.bool, device=device)

        gts_by_class = {c: [] for c in range(C)}
        for gt in instance_labels:
            gts_by_class[gt["label"]].append(gt["mask"].to(device))

        preds_by_class = {c: [] for c in range(C)}
        for pred in instance_output:
            preds_by_class[pred["label"]].append(
                (pred["score"], pred["mask"].to(device))
            )

        for c in range(C):
            gts = gts_by_class[c]
            preds = preds_by_class[c]
            if len(gts) == 0:
                continue

            valid[c] = True
            preds.sort(key=lambda x: x[0], reverse=True)
            scores = torch.tensor([p[0] for p in preds], device=device)
            tp = torch.zeros(len(preds), dtype=torch.float32, device=device)
            fp = torch.zeros(len(preds), dtype=torch.float32, device=device)
            matched = torch.zeros(len(gts), dtype=torch.bool, device=device)

            for i, (_, pmask) in enumerate(preds):
                best_iou = 0.0
                best_j = -1
                for j, gt_mask in enumerate(gts):
                    if matched[j]:
                        continue
                    iou_val = self._mask_iou(pmask, gt_mask)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_j = j

                if best_iou >= 0.5:
                    tp[i] = 1.0
                    matched[best_j] = True
                else:
                    fp[i] = 1.0

            cum_tp = torch.cumsum(tp, dim=0)
            cum_fp = torch.cumsum(fp, dim=0)
            recalls = cum_tp / max(len(gts), 1)
            precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

            precisions_flip = torch.flip(precisions, dims=[0])
            precisions_monotone = torch.cummax(precisions_flip, dim=0)[0]
            precisions = torch.flip(precisions_monotone, dims=[0])

            ap = torch.trapz(precisions, recalls)
            ap_per_class[c] = ap

        if valid.any():
            map_val = ap_per_class[valid].mean()
        else:
            map_val = torch.tensor(0.0, device=device)

        out_dict.update({
            "ap_per_class":          ap_per_class.cpu().numpy(),
            "map":                   map_val.cpu().numpy(),
        })

        return out_dict
    
    def train(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        scaler = amp.GradScaler(enabled=True)
        losses = []
        stats = []

        pbar = tqdm(self._train_loader, desc='[Train]')
        for feed_dict in pbar:
            inputs = feed_dict["inputs"].to(self._device)
            semantic_labels = feed_dict["semantic_labels"].to(self._device)
            offset_labels = feed_dict["offset_labels"].to(self._device)
            instance_labels = feed_dict["instance_labels"].to(self._device)
            optimizer.zero_grad()
 
            with amp.autocast(enabled=True):
                semantic_output, instance_output, offset_output = self._model(inputs)
                # semantic_output = self._model(inputs)
                # loss = self._compute_loss(semantic_output, semantic_labels)
                loss = self._compute_loss(semantic_output, semantic_labels, instance_output, instance_labels, offset_output, offset_labels)
                stat = self._compute_metrics(semantic_output.F, semantic_labels.F)

            stats.append(stat)
            losses.append(loss.item())
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{stat["miou"]:.4f}'
            })

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            del inputs, semantic_output, semantic_labels

        Path('./weights').mkdir(parents=False, exist_ok = True)
        torch.save(self._model.state_dict(), Path('./weights') / self._weights_file)
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
                offset_labels_cpu = feed_dict["offset_labels"].F.numpy()
                instance_labels_cpu = feed_dict["instance_labels"].F.numpy()
                # coords = feed_dict["coords"].numpy()
                # inverse_map = feed_dict["inverse_map"].numpy()

                inputs = feed_dict["inputs"].to(self._device)
                semantic_labels = feed_dict["semantic_labels"].to(self._device)
                offset_labels = feed_dict["offset_labels"].to(self._device)
                instance_labels = feed_dict["instance_labels"].to(self._device)

                with amp.autocast(enabled=True):
                    semantic_output, instance_output, offset_output = self._model(inputs)
                    # semantic_output = self._model(inputs)
                    # loss = self._compute_loss(semantic_output, semantic_labels)
                    loss = self._compute_loss(semantic_output, semantic_labels, instance_output, instance_labels, offset_output, offset_labels)
                    stat = self._compute_metrics(semantic_output.F, semantic_labels.F)

                losses.append(loss.item())
                stats.append(stat)

                voxels = semantic_output.C.cpu().numpy()
                semantic_output = torch.argmax(semantic_output.F.cpu(), dim=1).numpy()
                instance_output = torch.argmax(instance_output.F.cpu(), dim=1).numpy()
                offset_output = offset_output.F.cpu().numpy()
                # offset_output = torch.argmax(offset_output.F.cpu(), dim=1).numpy()
                # instance_output = np.zeros(semantic_output.shape)
                # instance_output = instance_output.F.cpu().reshape(-1, 1).numpy()
                # plt.hist(instance_output, bins=256, edgecolor='black')  # puedes ajustar 'bins' según necesites
                # plt.title('Histograma del array')
                # plt.xlabel('Valor')
                # plt.ylabel('Frecuencia')
                # plt.grid(True)
                # plt.show()
                # instance_output = DBSCAN(eps=1.0, min_samples=10, metric='euclidean').fit(instance_output).labels_
                # print(np.unique(instance_output))

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mIoU': f'{stat["miou"]:.4f}'
                })

                yield voxels, semantic_output, instance_output, offset_output, semantic_labels_cpu, instance_labels_cpu, offset_labels_cpu #, coords, inverse_map
        
        self._stats = stats
        self._losses = losses

        with open('./stats/stats.pkl', 'wb') as f:
            pickle.dump(self._stats, f)
        with open('./stats/losses.pkl', 'wb') as f:
            pickle.dump(self._losses, f)
