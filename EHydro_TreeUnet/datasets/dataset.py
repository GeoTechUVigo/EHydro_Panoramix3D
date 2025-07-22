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
    
    def _get_instance_offsets(self, voxels, semantic_labels, instance_labels):
        unique_labels, inverse_indices = np.unique(instance_labels, return_inverse=True)
        n_instances = len(unique_labels)
        # print(f'Numero de instancias: {n_instances}')

        sums = np.zeros((n_instances, 3), dtype=np.float64)
        counts = np.zeros(n_instances, dtype=np.int64)

        np.add.at(sums, inverse_indices, voxels)
        np.add.at(counts, inverse_indices, 1)

        centroids = sums / counts[:, None]
        centroids_per_point = centroids[inverse_indices]

        offsets = centroids_per_point - voxels
        offsets[semantic_labels == 0] = 0

        mag = np.linalg.norm(offsets, axis=1)
        mask = mag > 0
        mag = mag[:, None]

        dir = np.zeros_like(offsets)
        dir[mask] = offsets[mask] / mag[mask]

        return dir, np.log1p(mag)
    
    def _preprocess(self, idx: int):
        coords, feat, semantic_labels, instance_labels = self._load_file(self._files[idx % len(self._files)])
        if idx >= len(self._files):
            coords = self._agument_data(coords)

        coords -= np.min(coords, axis=0, keepdims=True)

        voxels, indices = sparse_quantize(coords, self._voxel_size, return_index=True)
        feat = feat[indices]
        semantic_labels = semantic_labels[indices]
        instance_labels = instance_labels[indices]
        offset_dir, offset_mag = self._get_instance_offsets(voxels, semantic_labels, instance_labels)

        voxels = torch.tensor(voxels, dtype=torch.int)
        feat = torch.tensor(feat.astype(np.float32), dtype=torch.float)

        semantic_labels = torch.tensor(semantic_labels, dtype=torch.long)
        offset_dir_labels = torch.tensor(offset_dir, dtype=torch.float)
        offset_mag_labels = torch.tensor(offset_mag, dtype=torch.float)

        inputs = SparseTensor(coords=voxels, feats=feat)
        
        semantic_labels = SparseTensor(coords=voxels, feats=semantic_labels)
        offset_dir_labels = SparseTensor(coords=voxels, feats=offset_dir_labels)
        offset_mag_labels = SparseTensor(coords=voxels, feats=offset_mag_labels)

        return {"inputs": inputs, "semantic_labels": semantic_labels, "offset_dir_labels": offset_dir_labels, "offset_mag_labels": offset_mag_labels}
    