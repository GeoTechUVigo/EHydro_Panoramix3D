from pydantic import Field, field_validator
from typing import Literal

from . import StrictModel, _load_yaml_with_includes


class TrainScheduleConfig(StrictModel):
    epochs: int = Field(..., ge=0, description="Number of epochs to train for (int >= 0).")
    lr_scale: float = Field(..., gt=0, description="Learning rate scale factor (float > 0).")
    freeze_modules: list[Literal['backbone', 'semantic', 'offset', 'centroid', 'instance']] = Field(..., description="List of modules to freeze (list of strings).")

    @field_validator('freeze_modules')
    @classmethod
    def _no_dupes_or_all(cls, v: list[Literal['backbone', 'semantic', 'offset', 'centroid', 'instance']]):
        if len(v) != len(set(v)):
            raise ValueError('freeze_modules must not contain duplicate entries.')
        if len(v) >= 5:
            raise ValueError('freeze_modules must not contain all entries.')
        return v


class LossCoeffsConfig(StrictModel):
    semantic_loss_coeff: float = Field(2.0, gt=0, description="Semantic segmentation loss coefficient (float > 0).")
    specie_loss_coeff: float = Field(2.0, gt=0, description="Specie classification loss coefficient (float > 0).")
    centroid_loss_coeff: float = Field(1.0, gt=0, description="Centroid detection loss coefficient (float > 0).")
    offset_loss_coeff: float = Field(1.0, gt=0, description="Offset prediction loss coefficient (float > 0).")
    offset_smooth_l1_beta_loss_coef: float = Field(5.0, gt=0, description="Offset Smooth L1 loss beta coefficient (float > 0).")
    instance_loss_coeff: float = Field(1.0, gt=0, description="Instance assignment loss coefficient (float > 0).")
    instance_matched_loss_coeff: float = Field(1.0, gt=0, description="Instance matched loss coefficient (float > 0).")
    instance_unmatched_loss_coeff: float = Field(7.0, gt=0, description="Instance unmatched loss coefficient (float > 0).")
    instance_bce_loss_coeff: float = Field(2.0, gt=0, description="Instance BCE loss coefficient (float > 0).")
    instance_dice_loss_coeff: float = Field(1.0, gt=0, description="Instance Dice loss coefficient (float > 0).")


class LearningRatesConfig(StrictModel):
    backbone: float = Field(2e-3, gt=0, description="Backbone learning rate (float > 0).")
    semantic: float = Field(1e-3, gt=0, description="Semantic head learning rate (float > 0).")
    offset: float = Field(2e-2, gt=0, description="Offset head learning rate (float > 0).")
    centroid: float = Field(1e-3, gt=0, description="Centroid head learning rate (float > 0).")
    instance: float = Field(2e-2, gt=0, description="Instance head learning rate (float > 0).")


class TrainerConfig(StrictModel):
    root: str = Field(..., description="Root directory for training outputs and checkpoints.")
    version_name: str = Field(..., description="Version name for the training run.")
    mode: Literal['train', 'val', 'test'] = Field('train', description="Training mode: 'train', 'val', or 'test'.")
    batch_size: int = Field(8, gt=0, description="Batch size for training (int > 0).")
    train_schedule: list[TrainScheduleConfig] = Field(..., description="Training schedule configuration.")
    start_epoch: int = Field(0, ge=0, description="Starting epoch number (int >= 0).")
    num_workers: int = Field(4, ge=0, description="Number of data loader workers (int >= 0).")
    weight_decay: float = Field(0.04, ge=0, description="Weight decay for optimization (float >= 0).")
    loss_coeffs: LossCoeffsConfig = Field(default_factory=LossCoeffsConfig, description="Loss function coefficients configuration.")
    learning_rates: LearningRatesConfig = Field(default_factory=LearningRatesConfig, description="Learning rates for different model components.")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainerConfig":
        config_dict = _load_yaml_with_includes(path)
        return cls.model_validate(config_dict)
