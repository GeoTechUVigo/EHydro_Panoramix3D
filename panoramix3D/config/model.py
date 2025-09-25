from pydantic import Field, field_validator
from typing import List, Optional, Tuple, Union

from . import StrictModel, MutableModel, _load_yaml_with_includes


layer_spec = Tuple[int, int, Union[int, Tuple[int, int, int]], Union[int, Tuple[int, int, int]]]
class BackboneConfig(StrictModel):
    resnet_blocks: List[layer_spec] = Field([
            (3, 16, 3, 1),
            (3, 32, 3, 2),
            (3, 64, 3, 2),
            (3, 128, 3, 2),
            (1, 128, (1, 1, 3), (1, 1, 2)),
        ], description="ResNet block specifications."
    )
    in_channels: int = Field(1, gt=0, description="Number of input channels.")

    @field_validator('resnet_blocks')
    @classmethod
    def _validate_resnet_blocks(cls, v: List[layer_spec]):
        if not v:
            raise ValueError('resnet_blocks must contain at least one block specification.')
        for block in v:
            if len(block) != 4:
                raise ValueError('Each block specification must be a tuple of four elements (num_blocks, out_channels, kernel, stride).')
            if not all(isinstance(x, int) and x > 0 for x in block[:2]):
                raise ValueError('The number of blocks and out channels of each block must be positive integers.')
            if not (isinstance(block[2], int) and block[2] > 0) and not (isinstance(block[2], tuple) and all(isinstance(x, int) and x > 0 for x in block[2])):
                raise ValueError('The kernel size of each block must be a positive integer or a tuple of positive integers.')
            if not (isinstance(block[3], int) and block[3] > 0) and not (isinstance(block[3], tuple) and all(isinstance(x, int) and x > 0 for x in block[3])):
                raise ValueError('The stride of each block must be a positive integer or a tuple of positive integers.')

        return v
    

class SemanticHeadConfig(StrictModel):
    num_classes: int = Field(3, gt=0, description="Number of semantic classes (int > 0).")


class SpecieHeadConfig(StrictModel):
    num_classes: int = Field(2, gt=0, description="Number of specie classes (int > 0).")


class CentroidHeadConfig(StrictModel):
    instance_density: float = Field(0.01, gt=0, description="Instance density (float > 0).")
    score_thres: float = Field(0.2, gt=0, description="Score threshold (float > 0).")
    centroid_thres: float = Field(0.3, gt=0, description="Centroid threshold (float > 0).")
    max_instances_per_scene: int = Field(64, gt=0, description="Maximum instances per scene (int > 0).")


class InstanceHeadConfig(StrictModel):
    descriptor_dim: int = Field(16, gt=0, description="Descriptor dimension (int > 0).")


class ModelConfig(MutableModel):
    weights_file: Optional[str] = Field(None, description="Path to pre-trained weights file.")
    backbone: BackboneConfig = Field(default_factory=BackboneConfig, description="Backbone network configuration.")
    semantic_head: SemanticHeadConfig = Field(default_factory=SemanticHeadConfig, description="Semantic head configuration.")
    specie_head: SpecieHeadConfig = Field(default_factory=SpecieHeadConfig, description="Specie head configuration.")
    centroid_head: CentroidHeadConfig = Field(default_factory=CentroidHeadConfig, description="Centroid head configuration.")
    instance_head: InstanceHeadConfig = Field(default_factory=InstanceHeadConfig, description="Instance head configuration.")

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        config_dict = _load_yaml_with_includes(path)
        return cls.model_validate(config_dict)
