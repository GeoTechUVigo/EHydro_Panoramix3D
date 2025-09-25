from pydantic import Field, field_validator, model_validator
from typing import List, Tuple

from . import StrictModel, _load_yaml_with_includes


class SpecieClassConfig(StrictModel):
    name: str = Field(..., description="Name of the specie class.")
    id: int = Field(..., ge=0, description="ID of the specie class (must be non-negative).")
    color: List[int] = Field(..., description="RGB color of the specie class.")

    @field_validator('color')
    @classmethod
    def _rgb_color(cls, v: Tuple[int, int, int]):
        if len(v) != 3 or any(c < 0 or c > 255 for c in v):
            raise ValueError('Color must be a list of three integers between 0 and 255.')
        return v


class SemanticClassConfig(StrictModel):
    name: str = Field(..., description="Name of the semantic class.")
    id: int = Field(..., ge=0, description="ID of the semantic class (must be non-negative).")
    color: List[int] = Field(..., description="RGB color of the semantic class.")

    @field_validator('color')
    @classmethod
    def _rgb_color(cls, v: Tuple[int, int, int]):
        if len(v) != 3 or any(c < 0 or c > 255 for c in v):
            raise ValueError('Color must be a list of three integers between 0 and 255.')
        return v
    

class AugmentationConfig(StrictModel):
    coef: float = Field(1.0, gt=0, description="Data augmentation coefficient (float > 0).")
    yaw_range: Tuple[float, float] = Field(
        (0.0, 0.0), description="Range of yaw rotation in degrees."
    )
    tilt_range: Tuple[float, float] = Field(
        (0.0, 0.0), description="Range of tilt rotation in degrees."
    )
    scale_range: Tuple[float, float] = Field(
        (1.0, 1.0), description="Range of scaling factor."
    )

    @field_validator('yaw_range', 'tilt_range', 'scale_range')
    @classmethod
    def _valid_range(cls, v: Tuple[float, float]):
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError('Range must be a tuple of two values (min, max) with min <= max.')
        return v
    
    @field_validator('scale_range')
    @classmethod
    def _scale_positive(cls, v: Tuple[float, float]):
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError('Scale range values must be positive.')
        return v
    

class SplitConfig(StrictModel):
    data_augmentation: AugmentationConfig = Field(
        default_factory=AugmentationConfig,
        description="Data augmentation settings for the split."
    )

class SplitsConfig(StrictModel):
    train: SplitConfig = Field(..., description="Configuration for the training split.")
    val: SplitConfig = Field(..., description="Configuration for the validation split.")
    test: SplitConfig = Field(..., description="Configuration for the test split.")


class DatasetConfig(StrictModel):
    name: str = Field(..., description="Name of the dataset.")
    root: str = Field(..., description="Path to the dataset folder.")
    
    voxel_size: float = Field(0.3, gt=0, description="Voxel size for downsampling (float > 0).")
    centroid_sigma_min: float = Field(1.0, gt=0, description="Minimum centroid sigma (float > 0).")
    centroid_sigma_max: float = Field(4.0, gt=0, description="Maximum centroid sigma (float > 0).")
    centroid_sigma_divisor: float = Field(18.0, gt=0, description="Centroid sigma divisor (float > 0).")
    min_tree_voxels: int = Field(125, ge=1, description="Minimum number of voxels for a tree instance (int >= 1).")
    feat_keys: List[str] = Field(['intensity'], description="List of feature keys.")
    semantic_classes: List[SemanticClassConfig] = Field(..., min_length=2, description="List of semantic classes (Min 2).")
    specie_classes: List[SpecieClassConfig] = Field(..., min_length=2, description="List of specie classes.")
    splits: SplitsConfig = Field(..., description="Dataset splits configuration.")

    @field_validator("semantic_classes")
    @classmethod
    def _semantic_classes_unique(cls, v: List[SemanticClassConfig]) -> List[SemanticClassConfig]:
        ids = [c.id for c in v]
        names = [c.name for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("IDs de semantic_classes deben ser únicos.")
        if len(names) != len(set(names)):
            raise ValueError("Nombres de semantic_classes deben ser únicos.")
        return v
    
    @field_validator("specie_classes")
    @classmethod
    def _specie_classes_unique(cls, v: List[SpecieClassConfig]) -> List[SpecieClassConfig]:
        ids = [c.id for c in v]
        names = [c.name for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("IDs de specie_classes deben ser únicos.")
        if len(names) != len(set(names)):
            raise ValueError("Nombres de specie_classes deben ser únicos.")
        return v

    @model_validator(mode="after")
    def _sigma_order(self):
        if self.centroid_sigma_max <= self.centroid_sigma_min:
            raise ValueError("centroid_sigma_max debe ser > centroid_sigma_min.")
        return self
    
    @model_validator(mode='after')
    def _validate_consistency(self):
        # Validate that dataset class IDs are sequential starting from 0
        dataset_num_classes = len(self.semantic_classes)
        expected_ids = set(range(dataset_num_classes))
        actual_ids = {cls.id for cls in self.semantic_classes}
        
        if actual_ids != expected_ids:
            raise ValueError(
                f"Dataset semantic_class IDs must be sequential starting from 0. "
                f"Expected: {sorted(expected_ids)}, Got: {sorted(actual_ids)}"
            )
        
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        config_dict = _load_yaml_with_includes(path)
        return cls.model_validate(config_dict)
