import torch

from torch import nn


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

        pred = pred.clamp(min=1e-6)
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
