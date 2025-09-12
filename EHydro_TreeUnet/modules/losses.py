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

        pred = pred.float().clamp(min=1e-3, max=1 - 1e-3)
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
    def __init__(self, lambda_focal: float = 1.0, lambda_dice: float = 1.0, alpha_focal: float = 0.25, gamma_focal: float = 2.0):
        super(HungarianInstanceLoss, self).__init__()

        self._lambda_focal = lambda_focal
        self._lambda_dice = lambda_dice

        self._alpha_focal = alpha_focal
        self._gamma_focal = gamma_focal
    
    def _focal_matrix(self, logits: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        N = logits.shape[0]
        p = torch.sigmoid(logits)

        log_p = F.logsigmoid(logits)
        log_1mp = F.logsigmoid(-logits)

        pos_term = -self._alpha_focal * ((1 - p) ** self._gamma_focal) * log_p
        neg_term = -(1.0 - self._alpha_focal) * (p ** self._gamma_focal) * log_1mp

        return (G / N) @ pos_term + ((1.0 - G) / N) @ neg_term

    def _dice_matrix(self, logits: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        inter = 2.0 * (G @ p)
        sum_p = p.sum(0, keepdim=True)
        sum_g = G.sum(1, keepdim=True)
        return 1.0 - inter / (sum_p + sum_g)

    def forward(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor):
        pred_logits = pred_logits.clamp(min=-10.0, max=10.0)
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
        focal = self._focal_matrix(pred_logits, gt_masks) * self._lambda_focal
        dice = self._dice_matrix(pred_logits, gt_masks) * self._lambda_dice
        cost = focal + dice

        if I == 0 or M == 0:
            return pred_logits.new_tensor(0.0), {
                'gt_indices': torch.empty(0, dtype=torch.long, device=device),
                'pred_indices': torch.empty(0, dtype=torch.long, device=device),
                'num_instances': I,
                'num_predictions': M
            }
        
        with torch.no_grad():
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            gi = torch.as_tensor(row, device=device, dtype=torch.long)
            pj = torch.as_tensor(col, device=device, dtype=torch.long)

        focal_pairs = focal[gi, pj] if gi.numel() else pred_logits.new_zeros(0, dtype=dtype, device=device)
        dice_pairs = dice[gi, pj] if gi.numel() else pred_logits.new_zeros(0, dtype=dtype, device=device)
        mask = focal_pairs + dice_pairs
        loss_mask = mask.sum() / I

        remap_info = {
            'gt_indices': gi,
            'pred_indices': pj,
            'num_instances': I,
            'num_predictions': M
        }

        return loss_mask, remap_info
