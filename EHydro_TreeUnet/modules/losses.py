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

        pred = pred.float().clamp(min=1e-6, max=1 - 1e-6)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log1p(-pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class HungarianInstanceLoss(nn.Module):
    def __init__(self, lambda_bce: float = 5.0, lambda_dice: float = 5.0, eps: float = 1e-6, normalize_by_num_gt: bool = True):
        super().__init__()
        self.lambda_bce = float(lambda_bce)
        self.lambda_dice = float(lambda_dice)
        self.eps = float(eps)
        self.normalize_by_num_gt = bool(normalize_by_num_gt)

    def _bce_matrix(self, logits: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # logits: (N, M), G: (I, N)
        N = logits.shape[0]
        ls_pos = F.logsigmoid(logits)    # (N, M) = log σ(x)
        ls_neg = F.logsigmoid(-logits)   # (N, M) = log (1-σ(x))
        # BCE(i,m) = mean_n [ -g * logσ(x) - (1-g)*log(1-σ(x)) ]
        denom = N + self.eps
        cost = -((G / denom) @ ls_pos) - (((1.0 - G) / denom) @ ls_neg)
        return cost   # (I, M)

    def _dice_matrix(self, logits: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        # logits: (N, M), G: (I, N)
        p = torch.sigmoid(logits)                 # (N, M)
        inter = 2.0 * (G @ p)                     # (I, M)
        sum_p = p.sum(0, keepdim=True)            # (1, M)
        sum_g = G.sum(1, keepdim=True)            # (I, 1)
        return 1.0 - inter / (sum_p + sum_g + self.eps)  # (I, M)

    def forward(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor):
        device = pred_logits.device
        dtype = pred_logits.dtype
        N, M = pred_logits.shape

        labels = gt_labels.view(-1).to(device=device, dtype=torch.long)
        uniq = torch.unique(labels)

        if uniq.numel() == 0:
            gt_masks = pred_logits.new_zeros((0, N), dtype=dtype)
        else:
            gt_masks = (labels.unsqueeze(0) == uniq.unsqueeze(1)).to(dtype=dtype)

        I = gt_masks.shape[0]

        bce = self._bce_matrix(pred_logits, gt_masks)
        dice = self._dice_matrix(pred_logits, gt_masks)
        cost = bce + dice

        if I == 0 or M == 0:
            return pred_logits.new_tensor(0.0), {
                'gt_indices': torch.empty(0, dtype=torch.long, device=device),
                'pred_indices': torch.empty(0, dtype=torch.long, device=device),
                'num_instances': I,
                'num_predictions': M
            }

        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
        
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        gi = torch.as_tensor(row, device=device, dtype=torch.long)
        pj = torch.as_tensor(col, device=device, dtype=torch.long)

        bce_pairs = bce[gi, pj] if gi.numel() else pred_logits.new_zeros(0, dtype=dtype, device=device)
        dice_pairs = dice[gi, pj] if gi.numel() else pred_logits.new_zeros(0, dtype=dtype, device=device)
        mask = self.lambda_bce * bce_pairs + self.lambda_dice * dice_pairs

        denom = float(I) if self.normalize_by_num_gt else max(gi.numel(), 1)
        loss_mask = mask.sum() / (denom + self.eps)

        remap_info = {
            'gt_indices': gi,
            'pred_indices': pj,
            'num_instances': I,
            'num_predictions': M
        }

        return loss_mask, remap_info
