import torch
import torch.nn.functional as F

from torch import nn
from torch.cuda.amp import autocast

from scipy.optimize import linear_sum_assignment


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    Args:
        lambda_bce: relative weight of the BCE loss between centers
        lambda_dice: relative weight of the DICE loss between centers

    Example:
        >>> matcher = HungarianMatcher()
        >>> pred_logits = torch.randn(5, 100)  # [num_predictions, num_voxels]
        >>> target_masks = torch.randint(0, 2, (3, 100)).float()  # [num_gt, num_voxels]
        >>> remap_info = matcher(pred_logits, target_masks)
        >>> print(remap_info['gt_indices'].shape)
        torch.Size([3])
        >>> print(remap_info['pred_indices'].shape)
        torch.Size([3])
    """
    def __init__(self, lambda_bce: float = 1.0, lambda_dice: float = 1.0):
        super().__init__()
        self._lambda_bce = lambda_bce
        self._lambda_dice = lambda_dice

        assert lambda_bce != 0 or lambda_dice != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            pred: A float tensor of shape [num_predictions, num_voxels].
                The predictions for each example.
            target: A float tensor of shape [num_gt, num_voxels]. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        Returns:
            A dict containing:
                'gt_indices': A tensor of shape [num_gt] containing the indices of the matched ground truth elements
                'pred_indices': A tensor of shape [num_gt] containing the indices of the matched prediction elements
        '''
        with autocast(enabled=False):
            pred = pred.float()
            target = target.float()
            bce_cost = self._lambda_bce * batch_sigmoid_ce_loss_jit(pred, target)
            dice_cost = self._lambda_dice * batch_dice_loss_jit(pred, target)

        cost = bce_cost + dice_cost
        row, col = linear_sum_assignment(cost.cpu())

        return {
            'gt_indices': torch.as_tensor(col, dtype=torch.int64, device=pred.device),
            'pred_indices': torch.as_tensor(row, dtype=torch.int64, device=pred.device)
        }
