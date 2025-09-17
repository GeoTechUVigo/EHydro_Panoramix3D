import torch

from torch import nn
from torch.nn import functional as F
from . import HungarianMatcher


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if inputs.numel() == 0:
        return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
    
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    if inputs.numel() == 0:
        return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
    
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


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
    def __init__(self, lambda_matched: float = 1.0, lambda_unmatched: float = 5.0, lambda_bce: float = 2.0, lambda_dice: float = 1.0):
        super().__init__()

        self._lambda_matched = lambda_matched
        self._lambda_unmatched = lambda_unmatched

        self._lambda_bce = lambda_bce
        self._lambda_dice = lambda_dice

        self._matcher = HungarianMatcher(lambda_bce=lambda_bce, lambda_dice=lambda_dice)

    def forward(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor, centroid_batches: torch.Tensor):
        batch_indices = torch.unique(pred_logits.C[:, 0])
        uniq, inv = torch.unique(gt_labels.F, return_inverse=True)
        uniq_batches = torch.zeros(uniq.size(0), device=gt_labels.F.device, dtype=torch.int32)
        uniq_batches.scatter_(0, inv, gt_labels.C[:, 0])

        gt_masks = (gt_labels.F.unsqueeze(0) == uniq.unsqueeze(1)).to(dtype=pred_logits.F.dtype)

        num_matches = torch.unique(torch.argmax(pred_logits.F, dim=1)).size(0) if pred_logits.F.size(1) > 0 else 0
        losses = torch.zeros(5, device=pred_logits.F.device, dtype=pred_logits.F.dtype)
        global_remap_info = {
            'gt_indices': torch.empty(0, dtype=torch.int64, device=pred_logits.F.device),
            'pred_indices': torch.empty(0, dtype=torch.int64, device=pred_logits.F.device),
            'num_instances': uniq.size(0),
            'num_predictions': pred_logits.F.size(1),
            'num_matches': num_matches
        }

        for batch_idx in batch_indices:
            voxel_mask = pred_logits.C[:, 0] == batch_idx
            pred_mask = centroid_batches == batch_idx
            gt_mask = uniq_batches == batch_idx

            if not pred_mask.any():
                continue

            batch_logits = pred_logits.F.T[pred_mask, :][:, voxel_mask]
            batch_gt_masks = gt_masks[gt_mask, :][:, voxel_mask]

            #print(f'max logits: {batch_logits.max().item():.4f}, min logits: {batch_logits.min().item():.4f}')

            remap_info = self._matcher(batch_logits, batch_gt_masks)
            bce_loss = self._lambda_bce * sigmoid_ce_loss_jit(
                batch_logits[remap_info['pred_indices']],
                batch_gt_masks[remap_info['gt_indices']],
                num_masks=batch_gt_masks.size(0)
            )
            dice_loss = self._lambda_dice * dice_loss_jit(
                batch_logits[remap_info['pred_indices']],
                batch_gt_masks[remap_info['gt_indices']],
                num_masks=batch_gt_masks.size(0)
            )

            matched_loss = self._lambda_matched * (bce_loss + dice_loss)
            unmatched_mask = torch.ones(batch_logits.size(0), dtype=torch.bool, device=batch_logits.device)
            if remap_info['pred_indices'].numel():
                unmatched_mask[remap_info['pred_indices']] = False

            unmatched_logits = batch_logits[unmatched_mask]
            unmatched_loss = self._lambda_unmatched * sigmoid_ce_loss_jit(unmatched_logits, torch.zeros_like(unmatched_logits), num_masks=unmatched_logits.size(0))
            total_loss = matched_loss + unmatched_loss

            losses[0] += total_loss
            losses[1] += matched_loss
            losses[2] += unmatched_loss
            losses[3] += bce_loss
            losses[4] += dice_loss

            pred_global = torch.nonzero(pred_mask, as_tuple=False).squeeze(1)
            gt_global = torch.nonzero(gt_mask, as_tuple=False).squeeze(1)

            global_remap_info['gt_indices'] = torch.cat([global_remap_info['gt_indices'], gt_global[remap_info['gt_indices']]], dim=0)
            global_remap_info['pred_indices'] = torch.cat([global_remap_info['pred_indices'], pred_global[remap_info['pred_indices']]], dim=0)

        losses = losses / batch_indices.size(0)
        loss = {
            'total_loss': losses[0],
            'matched_loss': losses[1],
            'unmatched_loss': losses[2],
            'bce_loss': losses[3],
            'dice_loss': losses[4]
        }

        return loss, global_remap_info