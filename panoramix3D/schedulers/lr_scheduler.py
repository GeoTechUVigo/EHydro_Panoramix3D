from torch.optim.lr_scheduler import ExponentialLR

from ..config import ExponentialLRConfig


def build_lr_scheduler(cfg, optimizer, current_epoch=0):
    if isinstance(cfg, ExponentialLRConfig):
        return ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
            last_epoch=current_epoch - 1
        )
    else:
        raise ValueError(f"Unsupported LR scheduler type: {cfg.type}")
    