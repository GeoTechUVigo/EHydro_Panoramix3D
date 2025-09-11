import torch
import numpy as np

import laspy

from typing import Tuple, List, Dict
from pathlib import Path

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class Dataset:
    def __init__(self,
            files: List[Path],
            voxel_size: float,
            feat_keys: List[str] =['intensity'],
            centroid_sigma: float = 1.0,
            data_augmentation: float = 1.0,
            yaw_range: Tuple[float, float] = (0.0, 360.0),
            tilt_range: Tuple[float, float] = (-5.0, 5.0),
            scale_range: Tuple[float, float] = (0.8, 1.2)
        ) -> None:

        self._rng = np.random.default_rng()
        self._files = files
        self._feat_keys = feat_keys

        self._voxel_size = voxel_size
        self._centroid_sigma = centroid_sigma
        self._len = int(len(self._files) * data_augmentation)
        
        self._yaw_range = yaw_range
        self._tilt_range = tilt_range
        self._scale = scale_range
        
    def __getitem__(self, idx: int) -> Dict:
        if isinstance(idx, slice):
            return [self._preprocess(i) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("Index out of range")
            return self._preprocess(idx)
        else:
            raise TypeError("Index must be a slice or an integer")
        
    def __len__(self) -> int:
        return self._len
    
    def _load_file(self, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ext = path.suffix.lower()

        coords = ...
        feats = ...
        semantic_labels = ...
        
        if ext in ('.las, .laz'):
            file = laspy.read(path)

            coords = file.xyz
            select_columns = []
            if 'intensity' in self._feat_keys:
                select_columns.append(np.array(file.norm_intensity)[:, None])

            feats = np.hstack(select_columns)

            semantic_labels = np.array(file.semantic_pred)
            instance_labels = np.array(file.instance_pred)
        else:
            raise ValueError(f'Unsopported file extension: {ext}!')

        return coords, feats, semantic_labels, instance_labels
    
    def _agument_data(self, coords: np.ndarray) -> np.ndarray:
        yaw = np.deg2rad(self._rng.uniform(*self._yaw_range))
        pitch = np.deg2rad(self._rng.uniform(*self._tilt_range))
        roll = np.deg2rad(self._rng.uniform(*self._tilt_range))
        scale = self._rng.uniform(*self._scale)

        if self._rng.random() > 0.5:
            coords[:, 0] *= -1
        if self._rng.random() > 0.5:
            coords[:, 1] *= -1

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        rotation_mtx = np.array([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                                 [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                                 [ -sp ,            cp*sr ,            cp*cr ]],
                                dtype=coords.dtype)

        return (coords @ rotation_mtx.T) * scale
    
    def _get_instance_offsets(self, voxels: np.ndarray, semantic_labels: np.ndarray, instance_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        offsets = np.zeros((len(voxels), 3), dtype=np.float32)
        for instance_id in np.unique(instance_labels):
            if instance_id == 0:
                continue

            idx = np.where(instance_labels == instance_id)[0]
            pts = voxels[idx]

            ctr = pts.mean(axis=0)
            d2 = np.sum((pts - ctr) ** 2, axis=1)
            ctr_idx = np.argmin(d2)
            ctr_voxel = pts[ctr_idx]

            offsets[idx, :] = ctr_voxel - pts

        # xy = voxels[:, :2]
        # x_min, y_min = xy.min(axis=0)
        # x_max, y_max = xy.max(axis=0)

        # diag = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

        return offsets  # / diag
    
    def _get_centroid_scores(self, voxels: np.ndarray, instance_labels: np.ndarray) -> np.ndarray:
        heat_map = np.zeros((len(voxels), 1), dtype=np.float32)

        for instance_id in np.unique(instance_labels):
            if instance_id == 0:
                continue

            idx = np.where(instance_labels == instance_id)[0]
            pts = voxels[idx]

            ctr = pts.mean(axis=0)
            d2 = np.sum((pts - ctr) ** 2, axis=1)
            ctr_idx = np.argmin(d2)
            ctr_voxel = pts[ctr_idx]

            d2 = np.sum((pts - ctr_voxel) ** 2, axis=1)
            mask = d2 < (3 * self._centroid_sigma) ** 2
            heat_map[idx[mask], 0] = np.exp(-d2[mask] / (2 * (self._centroid_sigma ** 2)))

        return heat_map
    
    def _preprocess(self, idx: int) -> Dict:
        coords, feat, semantic_labels, instance_labels = self._load_file(self._files[idx % len(self._files)])
        if idx >= len(self._files):
            coords = self._agument_data(coords)

        coords -= np.min(coords, axis=0, keepdims=True)

        voxels, indices = sparse_quantize(coords, self._voxel_size, return_index=True)
        feat = feat[indices]
        semantic_labels = semantic_labels[indices]
        instance_labels = instance_labels[indices]
        
        _, instance_labels = np.unique(instance_labels, return_inverse=True)

        centroid_score_labels = self._get_centroid_scores(voxels, instance_labels)
        offset_labels = self._get_instance_offsets(voxels, semantic_labels, instance_labels)

        voxels = torch.tensor(voxels, dtype=torch.int)
        feat = torch.tensor(feat.astype(np.float32), dtype=torch.float)

        semantic_labels = torch.tensor(semantic_labels, dtype=torch.long)
        centroid_score_labels = torch.tensor(centroid_score_labels, dtype=torch.float)
        offset_labels = torch.tensor(offset_labels, dtype=torch.float)
        instance_labels = torch.tensor(instance_labels, dtype=torch.long)

        inputs = SparseTensor(coords=voxels, feats=feat)
        
        semantic_labels = SparseTensor(coords=voxels, feats=semantic_labels)
        centroid_score_labels = SparseTensor(coords=voxels, feats=centroid_score_labels)
        offset_labels = SparseTensor(coords=voxels, feats=offset_labels)
        instance_labels = SparseTensor(coords=voxels, feats=instance_labels)

        return {
            "inputs": inputs,
            "semantic_labels": semantic_labels,
            "centroid_score_labels": centroid_score_labels,
            "offset_labels": offset_labels,
            "instance_labels": instance_labels
        }
    