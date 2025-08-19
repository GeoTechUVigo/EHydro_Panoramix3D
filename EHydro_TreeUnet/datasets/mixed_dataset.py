import numpy as np

from pathlib import Path
from typing import Tuple, List

from .dataset import Dataset


class MixedDataset:
    def __init__(
            self,
            folder: Path,
            voxel_size: float = 0.2,
            feat_keys: List[str] = ['intensity'],
            centroid_sigma: float = 1.0,
            train_pct: float = 0.8,
            data_augmentation: float = 1.0,
            yaw_range: Tuple[float, float] = (0.0, 360.0),
            tilt_range: Tuple[float, float] = (-5.0, 5.0),
            scale_range: Tuple[float, float] = (0.9, 1.1)
        ) -> None:
        self._folder = folder
        self._extensions = ('.laz', '.las')

        self._feat_channels = len(feat_keys)
        self._num_classes = 4
        self._class_names = ['Terrain', 'Low Vegetation', 'Stem', 'Canopy']
        # self._class_names = ['Terrain', 'Stem', 'Canopy']
        self._class_colormap = np.array([
            [128, 128, 128], # clase 0 - Terrain - gris
            [147, 255, 138], # clase 1 - Low vegetation - verde claro
            [255, 165, 0],   # clase 2 - Stem - naranja
            [0, 128, 0],     # clase 3 - Canopy - verde oscuro
        ], dtype=np.uint8)

        files = sorted(
            [f for f in self._folder.rglob("*") if f.is_file() and f.suffix.lower() in self._extensions],
            key=lambda f: f.name
        )

        train_idx = int(train_pct * len(files))
        self._train_dataset = Dataset(
            files[:train_idx],
            voxel_size=voxel_size,
            feat_keys=feat_keys,
            centroid_sigma=centroid_sigma,
            data_augmentation=data_augmentation,
            yaw_range=yaw_range,
            tilt_range=tilt_range,
            scale_range=scale_range
        )

        self._val_dataset = Dataset(
            files[train_idx:],
            voxel_size=voxel_size,
            feat_keys=feat_keys,
            centroid_sigma=centroid_sigma
        )

    @property
    def feat_channels(self) -> int:
        return self._feat_channels
    
    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_names(self) -> List[str]:
        return self._class_names
    
    @property
    def class_colormap(self) -> np.ndarray:
        return self._class_colormap
    
    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset
    
    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset
    