import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
from torchsparse import SparseTensor

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


class CenterNetFocalLoss(nn.Module):
    '''
    Copied from CenterNet (https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py)
    '''

    def __init__(self, lambda_pos: float = 1.0):
        super(CenterNetFocalLoss, self).__init__()
        self._lambda_pos = lambda_pos

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
        pos_loss = self._lambda_pos * pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss
    

class OffsetLoss(nn.Module):
    def __init__(self):
        super(OffsetLoss, self).__init__()

    def forward(self, pred_offsets, gt_offsets, total_instance_points):
        """Computes the L1 norm between prediction and ground truth and
        also computes cosine similarity between both vectors.
        see https://arxiv.org/pdf/2004.01658.pdf equations 2 and 3
        """
        pt_diff = pred_offsets - gt_offsets
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)
        offset_norm_loss = torch.sum(pt_dist) / (total_instance_points + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pred_offsets_norm = torch.norm(pred_offsets, p=2, dim=1)
        pred_offsets_ = pred_offsets / (pred_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = 1 - (gt_offsets_ * pred_offsets_).sum(-1)  # (N) // Modified to avoid negative values
        offset_dir_loss = torch.sum(direction_diff) / (total_instance_points + 1e-6)

        return offset_norm_loss, offset_dir_loss
    

class HungarianInstanceLoss(nn.Module):
    """
    This class computes a per-batch assignment-aware instance loss by combining
    matched and unmatched components. It internally uses a HungarianMatcher to
    align predicted instance logits (one column per predicted center) with
    ground-truth instance masks (derived from voxel-wise instance ids), and
    aggregates BCE and DICE losses for matched pairs plus a focal loss penalty
    for unmatched predictions.

    Args:
        lambda_matched: Relative weight for the loss over matched pairs
            (scales the sum of BCE and DICE on matched assignments).
        lambda_unmatched: Relative weight for the unmatched prediction penalty
            (applied to predictions that were not assigned to any GT).
        lambda_bce: Relative weight of the BCE term used both in the matcher
            cost and the matched loss.
        lambda_dice: Relative weight of the DICE term used both in the matcher
            cost and the matched loss.

    Example:
        >>> criterion = HungarianInstanceLoss(
        ...     lambda_matched=1.0,
        ...     lambda_unmatched=7.0,
        ...     lambda_bce=2.0,
        ...     lambda_dice=1.0,
        ... )
        >>> # pred_logits: Sparse-like tensor with .C (coords) and .F (logits)
        >>> # gt_labels:   Sparse-like tensor with .C (coords) and .F (instance ids)
        >>> # centroid_batches: Long/Int tensor with the batch id per prediction column
        >>> loss_dict, remap = criterion(pred_logits, gt_labels, centroid_batches)
        >>> print(loss_dict.keys())
        dict_keys(['total_loss', 'matched_loss', 'unmatched_loss', 'bce_loss', 'dice_loss'])
        >>> print(remap.keys())
        dict_keys(['gt_indices', 'pred_indices', 'num_instances', 'num_predictions', 'num_matches'])
    """
    def __init__(self, lambda_matched: float = 1.0, lambda_unmatched: float = 1.0, lambda_bce: float = 1.0, lambda_dice: float = 1.0):
        super().__init__()

        self._lambda_matched = lambda_matched
        self._lambda_unmatched = lambda_unmatched

        self._lambda_bce = lambda_bce
        self._lambda_dice = lambda_dice

        self._matcher = HungarianMatcher(lambda_bce=lambda_bce, lambda_dice=lambda_dice)

    def forward(self, pred_logits: SparseTensor, gt_labels: SparseTensor, centroid_batches: Tensor):
        """
        Args:
            pred_logits: A sparse-like structure with attributes:
                - C: Long tensor of shape [N_voxels, 4] (or compatible), where
                  the first column encodes the batch index per voxel.
                - F: Float tensor of shape [N_voxels, N_pred], containing the
                  per-voxel logits for each predicted center (one column per prediction).
            gt_labels: A sparse-like structure with attributes:
                - C: Long tensor of shape [N_voxels, 4] (or compatible), where
                  the first column encodes the batch index per voxel.
                - F: Long/Int tensor of shape [N_voxels], containing the
                  instance id per voxel (same id across all voxels of a GT instance).
            centroid_batches: 1-D tensor of dtype int/long with length N_pred,
                mapping each prediction (column in pred_logits.F) to its batch id.

        Returns:
            A tuple (loss_dict, remap_info) where:
                - loss_dict is a dict with keys:
                    'total_loss':   scalar tensor (sum of matched and unmatched terms)
                    'matched_loss': scalar tensor (weighted BCE + DICE over matched pairs)
                    'unmatched_loss': scalar tensor (weighted focal loss over unmatched predictions)
                    'bce_loss':     scalar tensor (BCE over matched pairs before lambda_matched)
                    'dice_loss':    scalar tensor (DICE over matched pairs before lambda_matched)
                - remap_info is a dict with keys:
                    'gt_indices':   1-D Long tensor of selected GT indices (global within batch)
                    'pred_indices': 1-D Long tensor of matched prediction indices (global within batch)
                    'num_instances': int (total number of unique GT instances across the batch)
                    'num_predictions': int (total number of prediction columns)
                    'num_matches':  int (sum of unique assigned predictions per batch)

        Notes:
            - Matching is performed independently per batch index (from the
              voxel/prediction batch ids) to avoid cross-batch assignments.
            - The unmatched term penalizes predictions that remain unassigned
              after Hungarian matching, using a sigmoid focal loss against zeros.
        """
        pred_logits.F = pred_logits.F

        batch_indices = torch.unique(pred_logits.C[:, 0])
        uniq, inv = torch.unique(gt_labels.F, return_inverse=True)
        uniq_batches = torch.zeros(uniq.size(0), device=gt_labels.F.device, dtype=torch.int32)
        uniq_batches.scatter_(0, inv, gt_labels.C[:, 0])

        gt_masks = (gt_labels.F.unsqueeze(0) == uniq.unsqueeze(1)).to(dtype=pred_logits.F.dtype)

        losses = torch.zeros(5, device=pred_logits.F.device, dtype=pred_logits.F.dtype)
        global_remap_info = {
            'gt_indices': torch.empty(0, dtype=torch.int64, device=pred_logits.F.device),
            'pred_indices': torch.empty(0, dtype=torch.int64, device=pred_logits.F.device),
            'num_instances': uniq.size(0),
            'num_predictions': pred_logits.F.size(1),
            'num_matches': 0
        }

        for batch_idx in batch_indices:
            voxel_mask = pred_logits.C[:, 0] == batch_idx
            pred_mask = centroid_batches == batch_idx
            gt_mask = uniq_batches == batch_idx

            if not pred_mask.any():
                continue

            batch_logits = pred_logits.F.T[pred_mask, :][:, voxel_mask]
            batch_gt_masks = gt_masks[gt_mask, :][:, voxel_mask]

            num_matches = torch.unique(torch.argmax(batch_logits, dim=0)).size(0)
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
            unmatched_loss = (self._lambda_unmatched * sigmoid_focal_loss(
                unmatched_logits,
                torch.zeros_like(unmatched_logits),
                alpha=0.25,
                gamma=2.0,
                reduction='mean'
            )) if unmatched_logits.numel() > 0 else torch.tensor(0.0, device=pred_logits.F.device, dtype=pred_logits.F.dtype)

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
            global_remap_info['num_matches'] += num_matches

        losses = losses / batch_indices.size(0)
        loss = {
            'total_loss': losses[0],
            'matched_loss': losses[1],
            'unmatched_loss': losses[2],
            'bce_loss': losses[3],
            'dice_loss': losses[4]
        }

        return loss, global_remap_info
    