from pydantic import Field, model_validator
from typing import Optional

from . import StrictModel, ModelConfig, DatasetConfig, TrainerConfig, _load_yaml_with_includes


class AppConfig(StrictModel):
    """
    Global application configuration that orchestrates all component configurations
    for the Panoramix3D multi-task 3D point cloud processing pipeline.
    
    This configuration class serves as the main entry point for configuring the entire
    application, including model architecture, dataset handling, and training parameters.
    It provides centralized configuration management with cross-component validation
    and consistency checks.

    Args:
        name: Application instance name for identification and logging.
        description: Optional description of the configuration purpose or experiment.
        model: ModelConfig object defining the neural network architecture including
            backbone, semantic head, centroid head, and instance head configurations.
        dataset: DatasetConfig object specifying dataset paths, preprocessing parameters,
            augmentation settings, and semantic class definitions.
        trainer: TrainerConfig object containing training hyperparameters, loss coefficients,
            learning rates, and optimization settings.

    Example:
        >>> from panoramix3D.config import AppConfig
        >>> cfg = AppConfig.from_yaml('app_config.yaml')
        >>> print(f"Training {cfg.name} with {cfg.model.semantic_head.num_classes} classes")
        >>> model = Panoramix3D(cfg.model)
        >>> dataset = Panoramix3DDataset(cfg.dataset, split='train')
        Training tree_detection_v1 with 3 classes

    Notes:
        - Provides cross-component validation between model and dataset configurations
        - Enables centralized parameter management across the entire pipeline
        - Supports YAML-based configuration loading for reproducible experiments
        - Validates consistency between dataset classes and model output dimensions
    """
    name: str = Field(..., description="Application configuration name for identification.")
    description: Optional[str] = Field(None, description="Optional description of the configuration or experiment.")
    model: ModelConfig = Field(..., description="Model architecture configuration.")
    dataset: DatasetConfig = Field(..., description="Dataset and preprocessing configuration.")
    trainer: TrainerConfig = Field(..., description="Training and optimization configuration.")

    @model_validator(mode='after')
    def _validate_consistency(self):
        dataset_num_classes = len(self.dataset.semantic_classes)
        model_num_classes = self.model.semantic_head.num_classes
        
        if model_num_classes != dataset_num_classes:
            raise ValueError(
                f"Model semantic head expects {model_num_classes} classes, "
                f"but dataset defines {dataset_num_classes} classes. "
                f"Please ensure model.semantic_head.num_classes matches len(dataset.classes)."
            )
        
        return self
    
    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        config_dict = _load_yaml_with_includes(path)
        return cls.model_validate(config_dict)
