from typing import Any
from torch import nn
from torchsparse import nn as spnn


from ..config import StepDecayBNConfig


BATCH_NORM_MODULES: Any = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    spnn.BatchNorm
)


def set_momentum(momentum: float):
    def fn(module):
        if isinstance(module, BATCH_NORM_MODULES):
            module.momentum = momentum

    return fn


class StepDecayBNScheduler:
    def __init__(self, model: nn.Module, momentum: float, decay: float, decay_step: int, clip: float, current_epoch: int):
        if not isinstance(model, nn.Module):
            raise ValueError("model should be an instance of nn.Module")

        self.model = model
        self.momentum = momentum
        self.decay = decay
        self.decay_step = decay_step
        self.clip = clip
        self.current_epoch = current_epoch

    def step(self):
        new_momentum = max(
            self.momentum * (self.decay ** int(self.current_epoch // self.decay_step)),
            self.clip
        )

        if new_momentum == self.momentum:
            return
        
        self.model.apply(set_momentum(new_momentum))
        self.momentum = new_momentum
        self.current_epoch += 1


def build_bn_scheduler(cfg, model, current_epoch: int = 0):
    if isinstance(cfg, StepDecayBNConfig):
        return StepDecayBNScheduler(
            model,
            momentum=cfg.bn_momentum,
            decay=cfg.bn_decay,
            decay_step=cfg.decay_step,
            clip=cfg.bn_clip,
            current_epoch=current_epoch
        )
    else:
        raise ValueError(f"Unsupported BN scheduler type: {cfg.type}")
    