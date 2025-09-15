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
    def __init__(self, lambda_matched: float = 1.0, lambda_unmatched: float = 1.0, lambda_focal: float = 1.0, lambda_dice: float = 1.0, alpha_focal: float = 0.25, gamma_focal: float = 2.0):
        super().__init__()

        self._lambda_matched = lambda_matched
        self._lambda_unmatched = lambda_unmatched

        self._lambda_focal = lambda_focal
        self._lambda_dice = lambda_dice

        self._alpha_focal = alpha_focal
        self._gamma_focal = gamma_focal

    
    def _bce_matrix(self, logits: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        hw = logits.shape[1]
        G = G.T

        pos = F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, G) + torch.einsum(
            "nc,mc->nm", neg, (1 - G)
        )

        return (loss / hw).T
    
    def _focal_matrix(self, logits: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        x = logits.float()
        Gf = G.float()

        p = torch.sigmoid(x)
        log_p   = F.logsigmoid(x)
        log_1mp = F.logsigmoid(-x)

        pos_fl = ((1.0 - p) ** self._gamma_focal) * (-log_p)
        neg_fl = (p ** self._gamma_focal) * (-log_1mp)

        w_pos = self._alpha_focal * Gf
        w_neg = (1.0 - self._alpha_focal) * (1.0 - Gf)

        num = (w_pos @ pos_fl) + (w_neg @ neg_fl)
        den = w_pos.sum(dim=1, keepdim=True) + w_neg.sum(dim=1, keepdim=True)
        cost = num / (den + 1e-6)

        return cost.to(logits.dtype)

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
        if I == 0:
            total_loss = F.softplus(pred_logits).mean()
            loss = {
                'total_loss': total_loss,
                'matched_loss': pred_logits.new_tensor(0.0),
                'unmatched_loss': total_loss,
                'focal_loss': pred_logits.new_tensor(0.0),
                'dice_loss': pred_logits.new_tensor(0.0)
            }
            remap_info = {
                'gt_indices': torch.empty(0, dtype=torch.long, device=device),
                'pred_indices': torch.empty(0, dtype=torch.long, device=device),
                'num_instances': I,
                'num_predictions': M
            }
            return loss, remap_info

        if M == 0:
            total_loss = pred_logits.new_tensor(0.0)
            loss = {
                'total_loss': total_loss,
                'matched_loss': pred_logits.new_tensor(0.0),
                'unmatched_loss': pred_logits.new_tensor(0.0),
                'focal_loss': pred_logits.new_tensor(0.0),
                'dice_loss': pred_logits.new_tensor(0.0)
            }
            remap_info = {
                'gt_indices': torch.empty(0, dtype=torch.long, device=device),
                'pred_indices': torch.empty(0, dtype=torch.long, device=device),
                'num_instances': I,
                'num_predictions': M
            }
            return loss, remap_info
        
        focal = self._bce_matrix(pred_logits, gt_masks) * self._lambda_focal
        dice = self._dice_matrix(pred_logits, gt_masks) * self._lambda_dice

        # print('focal nans: ', torch.isnan(focal).sum().item(), ' dice nans: ', torch.isnan(dice).sum().item())
        cost = focal + dice
        
        with torch.no_grad():
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            gi = torch.as_tensor(row, device=device, dtype=torch.long)
            pj = torch.as_tensor(col, device=device, dtype=torch.long)

        focal_pairs = focal[gi, pj] if gi.numel() else pred_logits.new_zeros(0, dtype=dtype, device=device)
        dice_pairs = dice[gi, pj] if gi.numel() else pred_logits.new_zeros(0, dtype=dtype, device=device)
        focal_loss = focal_pairs.sum() / I
        dice_loss = dice_pairs.sum() / I
        matched_loss = self._lambda_matched * (focal_loss + dice_loss)

        unmatched_mask = torch.ones(M, dtype=torch.bool, device=device)
        if pj.numel():
            unmatched_mask[pj] = False

        unmatched_logits = pred_logits[:, unmatched_mask]
        if unmatched_logits.numel() == 0:
            unmatched_loss = pred_logits.new_tensor(0.0)
        else:
            unmatched_loss = self._lambda_unmatched * F.softplus(unmatched_logits).mean()

        total_loss = matched_loss + unmatched_loss

        # print(f'Matched Loss: {matched_loss.item():.4f}, Unmatched Loss: {unmatched_loss.item():.4f}')
        # print(f'focal: {focal.mean().item():.4f}, dice: {dice.mean().item():.4f}')

        loss = {
            'total_loss': total_loss,
            'matched_loss': matched_loss,
            'unmatched_loss': unmatched_loss,
            'focal_loss': focal_loss.mean(),
            'dice_loss': dice_loss.mean()
        }

        remap_info = {
            'gt_indices': gi,
            'pred_indices': pj,
            'num_instances': I,
            'num_predictions': M
        }

        return loss, remap_info
