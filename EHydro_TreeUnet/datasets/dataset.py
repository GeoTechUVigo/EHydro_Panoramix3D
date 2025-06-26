import torch
import numpy as np

import laspy
from plyfile import PlyData

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class Dataset:
    def __init__(self, files, voxel_size: float, data_augmentation: float = 1.0, yaw_range = (0, 360), tilt_range = (-5, 5), scale = (0.9, 1.1), feat_keys=['intensity', 'x_norm', 'y_norm', 'z_norm', 'wavelength']) -> None:
        self._rng = np.random.default_rng()
        self._files = files
        self._feat_keys = feat_keys

        self._voxel_size = voxel_size
        self._len = int(len(self._files) * data_augmentation)
        
        self._yaw_range = yaw_range
        self._tilt_range = tilt_range
        self._scale = scale
        
    def __getitem__(self, idx):
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
        
    def __len__(self):
        return self._len
    
    def _load_file(self, path):
        ext = path.suffix.lower()

        coords = ...
        feats = ...
        semantic_labels = ...
        
        if ext in ('.las, .laz'):
            file = laspy.read(path)

            coords = np.vstack((file.x, file.y, file.z)).transpose()
            intensity = np.array(file.I_norm)[:, None]

            coords_min = coords.min(axis=0)
            coords_norm = (coords - coords_min) / (coords.max(axis=0) - coords_min)

            wavelength = np.array(file.wavelength)[:, None]

            select_columns = []
            if 'intensity' in self._feat_keys:
                select_columns.append(intensity)
            if 'x_norm' in self._feat_keys:
                select_columns.append(coords_norm[:, [0]])
            if 'y_norm' in self._feat_keys:
                select_columns.append(coords_norm[:, [1]])
            if 'z_norm' in self._feat_keys:
                select_columns.append(coords_norm[:, [2]])
            if 'wavelength' in self._feat_keys:
                select_columns.append(wavelength)

            feats = np.hstack(select_columns)

            semantic_labels = np.array(file.classification)
            instance_labels = np.array(file.treeID)
        else:
            raise ValueError(f'Unsopported file extension: {ext}!')

        return coords, feats, semantic_labels, instance_labels
    
    def _agument_data(self, coords):
        yaw = np.deg2rad(self._rng.uniform(*self._yaw_range))
        pitch = np.deg2rad(self._rng.uniform(*self._tilt_range))
        roll = np.deg2rad(self._rng.uniform(*self._tilt_range))
        scale = self._rng.uniform(*self._scale)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        rotation_mtx = np.array([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
                                 [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
                                 [ -sp ,            cp*sr ,            cp*cr ]],
                                dtype=coords.dtype)

        return (coords @ rotation_mtx.T) * scale
        
    def _preprocess(self, idx: int):
        coords, feat, semantic_labels, instance_labels = self._load_file(self._files[idx % len(self._files)])
        if idx >= len(self._files):
            coords = self._agument_data(coords)

        coords -= np.min(coords, axis=0, keepdims=True)

        voxels, indices, inverse_map = sparse_quantize(coords, self._voxel_size, return_index=True, return_inverse=True)
        feat = feat[indices]
        semantic_labels = semantic_labels[indices]
        instance_labels = instance_labels[indices]

        voxels = torch.tensor(voxels, dtype=torch.int)
        feat = torch.tensor(feat.astype(np.float32), dtype=torch.float)
        semantic_labels = torch.tensor(semantic_labels, dtype=torch.long)
        instance_labels = torch.tensor(instance_labels, dtype=torch.long)

        inputs = SparseTensor(coords=voxels, feats=feat)
        semantic_labels = SparseTensor(coords=voxels, feats=semantic_labels)
        instance_labels = SparseTensor(coords=voxels, feats=instance_labels)

        return {"inputs": inputs, "semantic_labels": semantic_labels, "instance_labels": instance_labels}