import torch

from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class FocalLoss(nn.Module):
    '''
    Copied from CenterNet (https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py)
    '''

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        '''
        Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
        '''

        pred = pred.clamp(min=1e-6, max=1 - 1e-6)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class BCEPlusDice(nn.Module):
    def __init__(self):
        super(BCEPlusDice, self).__init__()
        self._criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        N, K = pred.shape
        targets = torch.zeros((N, K), dtype=torch.float32, device=pred.device)
        targets[torch.arange(N), gt] = 1.0

        prob = torch.sigmoid(pred)
        loss_bce = self._criterion_bce(pred, targets)
        loss_dice = (1 - (2 * ( prob * targets).sum(0) + 1e-4) / (prob.sum(0) + targets.sum(0) + 1e-4)).mean()
        
        return loss_bce + loss_dice


class InstanceVariableKLoss(nn.Module):
    def __init__(
        self,
        use_focal=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        dice_weight=1.0,
        bce_or_focal_weight=1.0,
        bg_weight=1.0,
        ghost_weight=0.2,
        count_weight=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_w = dice_weight
        self.bce_w = bce_or_focal_weight
        self.bg_w = bg_weight
        self.ghost_w = ghost_weight
        self.count_w = count_weight
        self.eps = eps

        self._bce_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    @torch.no_grad()
    def _hungarian(self, cost_mat):
        K_pred, M_gt = cost_mat.shape
        if K_pred == 0 or M_gt == 0:
            return [], list(range(K_pred)), list(range(M_gt))

        if torch.is_tensor(cost_mat):
            cm = cost_mat.detach().cpu().numpy()
        else:
            cm = cost_mat

        row_ind, col_ind = linear_sum_assignment(cm)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
        used_pred = set(r for r in row_ind)
        used_gt = set(c for c in col_ind)
        pred_unmatched = [i for i in range(K_pred) if i not in used_pred]
        gt_unmatched = [j for j in range(M_gt) if j not in used_gt]

        return pairs, pred_unmatched, gt_unmatched
    
    def dice_coeff_soft(self, pred_probs, target_mask, eps=1e-6):
        num = 2.0 * torch.sum(pred_probs * target_mask)
        den = torch.sum(pred_probs) + torch.sum(target_mask) + eps
        return (num + eps) / (den)

    def dice_loss_soft(self, pred_probs, target_mask, eps=1e-6):
        return 1.0 - self.dice_coeff_soft(pred_probs, target_mask, eps=eps)
    
    def focal_loss_binary(self, logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (alpha * (1 - pt).pow(gamma)) * bce
        if reduction == 'mean':
            return focal.mean()
        elif reduction == 'sum':
            return focal.sum()
        else:
            return focal

    def forward(self, logits, gt_instance_ids):
        V, Kp1 = logits.shape
        K = Kp1 - 1

        valid_mask = (gt_instance_ids >= 0)
        if valid_mask.sum() == 0:
            zero = logits.sum() * 0.0
            return zero, {
                'loss_mask': 0.0, 'loss_bg': 0.0, 'loss_ghost': 0.0, 'loss_count': 0.0, 'num_gt': 0, 'num_pred': K
            }

        logits_v = logits[valid_mask]
        gt_ids_v = gt_instance_ids[valid_mask]

        probs = F.softmax(logits_v, dim=1)
        probs_bg = probs[:, 0]
        probs_inst = probs[:, 1:]

        gt_pos = gt_ids_v > 0
        unique_ids = torch.unique(gt_ids_v[gt_pos])
        M = len(unique_ids)

        if M == 0:
            loss_bg = self._bce_criterion(logits_v[:, 0], (gt_ids_v == 0).float())
            if K > 0:
                mass = probs_inst.mean(dim=0)
                loss_ghost = mass.mean()
            else:
                loss_ghost = torch.tensor(0.0, device=logits.device)

            loss_total = self.bg_w * loss_bg + self.ghost_w * loss_ghost
            return loss_total, {
                'loss_mask': 0.0, 'loss_bg': float(loss_bg.detach()), 'loss_ghost': float(loss_ghost.detach()),
                'loss_count': 0.0, 'num_gt': 0, 'num_pred': K
            }

        gt_masks = []
        for gid in unique_ids:
            gt_masks.append((gt_ids_v == gid).float())
        gt_masks = torch.stack(gt_masks, dim=0)

        if K > 0:
            pred_masks = probs_inst.transpose(0, 1).contiguous()
        else:
            pred_masks = probs_inst.new_zeros((0, probs_inst.shape[0]))

        if K > 0:
            with torch.no_grad():
                cost = torch.zeros((K, M), device=logits.device)
                for k in range(K):
                    pk = pred_masks[k]
                    for j in range(M):
                        gj = gt_masks[j]
                        num = 2.0 * torch.sum(pk * gj)
                        den = torch.sum(pk) + torch.sum(gj) + self.eps
                        dice = (num + self.eps) / (den)
                        cost[k, j] = 1.0 - dice

                pairs, pred_unmatched, gt_unmatched = self._hungarian(cost.detach().cpu())
        else:
            pairs, pred_unmatched, gt_unmatched = [], [], list(range(M))

        loss_mask_terms = []
        for (k, j) in pairs:
            pk_probs = pred_masks[k]
            pk_logits = logits_v[:, 1 + k]
            gj = gt_masks[j]

            ldice = self.dice_loss_soft(pk_probs, gj, eps=self.eps)
            if self.use_focal:
                lbce = self.focal_loss_binary(pk_logits, gj, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='mean')
            else:
                lbce = self._bce_criterion(pk_logits, gj)
            loss_mask_terms.append((self.dice_w * ldice + self.bce_w * lbce) / M)

        loss_mask = torch.stack(loss_mask_terms).mean() if len(loss_mask_terms) > 0 else torch.tensor(0.0, device=logits.device)

        gt_bg = (gt_ids_v == 0).float()
        loss_bg = self._bce_criterion(logits_v[:, 0], gt_bg)

        if K > 0 and len(pred_unmatched) > 0:
            mass = pred_masks[pred_unmatched].mean(dim=1)
            loss_ghost = mass.mean()
        else:
            loss_ghost = torch.tensor(0.0, device=logits.device)

        if K > 0:
            s_k = pred_masks.mean(dim=1)
            M_pred_soft = s_k.sum()
        else:
            M_pred_soft = torch.tensor(0.0, device=logits.device)

        M_gt = torch.tensor(float(M), device=logits.device)
        loss_count = (M_pred_soft - M_gt).pow(2)

        loss_total = (
            loss_mask
            + self.bg_w * loss_bg
            + self.ghost_w * loss_ghost
            + self.count_w * loss_count
        )

        stats = {
            'loss_mask': float(loss_mask.detach()),
            'loss_bg': float(loss_bg.detach()),
            'loss_ghost': float(loss_ghost.detach()),
            'loss_count': float(loss_count.detach()),
            'num_gt': int(M),
            'num_pred': int(K),
            'matched': int(len(pairs))
        }
        
        return loss_total, stats
    