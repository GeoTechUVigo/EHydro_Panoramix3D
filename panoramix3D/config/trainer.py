from pydantic import Field, field_validator
from typing import Literal, Union, Annotated

from . import StrictModel, _load_yaml_with_includes


class ExponentialLRConfig(StrictModel):
    type: Literal['ExponentialLR']
    gamma: float = Field(0.95, gt=0, lt=1, description="Exponential decay factor (0 < gamma < 1).")


class DummyLRConfig(StrictModel):
    type: Literal['DummyLR']


LRScheduleConfig = Annotated[
    Union[ExponentialLRConfig, DummyLRConfig],
    Field(discriminator='type')
]


class StepDecayBNConfig(StrictModel):
    type: Literal['StepDecay']
    bn_momentum: float = Field(0.1, gt=0, lt=1, description="Initial batch norm momentum (0 < bn_momentum < 1).")
    bn_decay: float = Field(0.5, gt=0, lt=1, description="Batch norm momentum decay factor (0 < bn_decay < 1).")
    decay_step: int = Field(4, gt=0, description="Number of epochs between each decay (int > 0).")
    bn_clip: float = Field(1e-2, gt=0, lt=1, description="Minimum batch norm momentum (0 < bn_clip < 1).")


class DummyBNConfig(StrictModel):
    type: Literal['DummyBN']


BNScheduleConfig = Annotated[
    Union[StepDecayBNConfig, DummyBNConfig],
    Field(discriminator='type')
]


class FreezeModulesConfig(StrictModel):
    module: Literal['backbone', 'semantic', 'classification', 'offset', 'centroid', 'instance'] = Field(..., description="Module to freeze.")
    epochs: int = Field(..., ge=0, description="Number of epochs to freeze the module (int >= 0).")


class LossCoeffsConfig(StrictModel):
    semantic_loss_coeff: float = Field(2.0, gt=0, description="Semantic segmentation loss coefficient (float > 0).")
    classification_loss_coeff: float = Field(2.0, gt=0, description="Classification loss coefficient (float > 0).")
    centroid_loss_coeff: float = Field(1.0, gt=0, description="Centroid detection loss coefficient (float > 0).")
    offset_loss_coeff: float = Field(1.0, gt=0, description="Offset prediction loss coefficient (float > 0).")
    offset_norm_loss_coeff: float = Field(1.0, gt=0, description="Offset norm loss coefficient (float > 0).")
    offset_dir_loss_coeff: float = Field(1.0, gt=0, description="Offset direction loss coefficient (float > 0).")
    instance_loss_coeff: float = Field(1.0, gt=0, description="Instance assignment loss coefficient (float > 0).")
    instance_matched_loss_coeff: float = Field(1.0, gt=0, description="Instance matched loss coefficient (float > 0).")
    instance_unmatched_loss_coeff: float = Field(7.0, gt=0, description="Instance unmatched loss coefficient (float > 0).")
    instance_bce_loss_coeff: float = Field(2.0, gt=0, description="Instance BCE loss coefficient (float > 0).")
    instance_dice_loss_coeff: float = Field(1.0, gt=0, description="Instance Dice loss coefficient (float > 0).")


class WeightDecaysConfig(StrictModel):
    backbone: float = Field(1e-4, ge=0, description="Backbone weight decay (float >= 0).")
    semantic: float = Field(1e-4, ge=0, description="Semantic head weight decay (float >= 0).")
    classification: float = Field(1e-4, ge=0, description="Classification head weight decay (float >= 0).")
    offset: float = Field(1e-4, ge=0, description="Offset head weight decay (float >= 0).")
    centroid: float = Field(1e-4, ge=0, description="Centroid head weight decay (float >= 0).")
    instance: float = Field(1e-4, ge=0, description="Instance head weight decay (float >= 0).")


class LearningRatesConfig(StrictModel):
    backbone: float = Field(2e-3, gt=0, description="Backbone learning rate (float > 0).")
    semantic: float = Field(1e-3, gt=0, description="Semantic head learning rate (float > 0).")
    classification: float = Field(1e-3, gt=0, description="Classification head learning rate (float > 0).")
    offset: float = Field(2e-2, gt=0, description="Offset head learning rate (float > 0).")
    centroid: float = Field(1e-3, gt=0, description="Centroid head learning rate (float > 0).")
    instance: float = Field(2e-2, gt=0, description="Instance head learning rate (float > 0).")


class TrainerConfig(StrictModel):
    root: str = Field(..., description="Root directory for training outputs and checkpoints.")
    version_name: str = Field(..., description="Version name for the training run.")
    mode: Literal['train', 'val', 'test'] = Field('train', description="Training mode: 'train', 'val', or 'test'.")
    batch_size: int = Field(8, gt=0, description="Batch size for training (int > 0).")
    start_epoch: int = Field(0, ge=0, description="Starting epoch number (int >= 0).")
    num_epochs: int = Field(50, gt=0, description="Total number of training epochs (int > 0).")
    num_workers: int = Field(4, ge=0, description="Number of data loader workers (int >= 0).")
    loss_coeffs: LossCoeffsConfig = Field(default_factory=LossCoeffsConfig, description="Loss function coefficients configuration.")
    freeze_modules: list[FreezeModulesConfig] = Field([], description="List of modules to freeze with their respective epochs.")
    weight_decays: WeightDecaysConfig = Field(default_factory=WeightDecaysConfig, description="Weight decays for different model components.")
    learning_rates: LearningRatesConfig = Field(default_factory=LearningRatesConfig, description="Learning rates for different model components.")
    lr_scheduler: LRScheduleConfig = Field(..., description="Learning rate schedule configuration.")
    bn_scheduler: BNScheduleConfig = Field(..., description="Batch normalization schedule configuration.")
    class_weights: Literal['none', 'sqrt', 'log'] = Field('none', description="Class weights strategy for loss functions ('none', 'sqrt', or 'log').")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainerConfig":
        config_dict = _load_yaml_with_includes(path)
        return cls.model_validate(config_dict)
