import torch
import numpy as np

import laspy

from torch.utils.data import Dataset
from torchsparse import SparseTensor

from typing import Literal, Tuple, List, Dict
from pathlib import Path

from torchsparse.utils.quantize import sparse_quantize

from ..config import DatasetConfig, SplitConfig


class Panoramix3DDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing 3D point cloud data for multi-task
    learning including semantic segmentation, instance detection, and offset prediction.
    
    This dataset handles LAS/LAZ point cloud files with semantic and instance annotations,
    applying voxelization, data augmentation, and generating supervision signals for
    centroid detection and offset prediction. It supports train/validation/test splits
    with configurable augmentation strategies.

    The dataset automatically generates three types of supervision:
    1. Semantic labels: Per-voxel class labels from ground truth
    2. Centroid scores: Gaussian heatmaps centered on instance centers
    3. Offset vectors: Displacement vectors pointing from voxels to instance centers

    Args:
        cfg: DatasetConfig object containing all dataset parameters:
            - cfg.name: Dataset name identifier
            - cfg.root: Path to the dataset root directory  
            - cfg.voxel_size: Voxel size for downsampling (default: 0.3)
            - cfg.feat_keys: List of feature keys to extract (default: ['intensity'])
            - cfg.min_tree_voxels: Minimum voxels per instance (default: 125)
            - cfg.centroid_sigma_min: Minimum Gaussian sigma (default: 1.0)
            - cfg.centroid_sigma_max: Maximum Gaussian sigma (default: 4.0)
            - cfg.centroid_sigma_divisor: Sigma computation divisor (default: 18.0)
            - cfg.classes: List of SemanticClassConfig with id, name, and color
            - cfg.splits: SplitsConfig containing train/val/test configurations
              Each split has data_augmentation settings (coef, yaw_range, tilt_range, scale_range)
        split: Dataset split to load ('train', 'val', or 'test').

    Example:
        >>> from panoramix3D.config import DatasetConfig
        >>> cfg = DatasetConfig.from_yaml('dataset_config.yaml')
        >>> dataset = Panoramix3DDataset(cfg, split='train')
        >>> sample = dataset[0]
        >>> print(f"Input shape: {sample['inputs'].F.shape}")
        >>> print(f"Semantic labels: {sample['semantic_labels'].F.shape}")
        Input shape: torch.Size([1024, 1])
        Semantic labels: torch.Size([1024])

    Notes:
        - Files are loaded from cfg.root/split/ directory structure
        - Data augmentation coefficient (cfg.splits.{split}.data_augmentation.coef) controls virtual dataset size expansion
        - Instance filtering removes instances smaller than cfg.min_tree_voxels
        - All outputs are SparseTensors with consistent coordinate indexing
    """
    def __init__(self, cfg: DatasetConfig, split: Literal['train', 'val', 'test'] = 'train'):
        super().__init__()
        self._extensions = ('.laz', '.las')
        self._rng = np.random.default_rng()

        self._cfg = cfg
        self._split = split
        self._split_cfg: SplitConfig = getattr(self._cfg.splits, split)
        self._files = self._scan_files()
        self._len = int(len(self._files) * self._split_cfg.data_augmentation.coef)

    def _scan_files(self) -> List[Path]:
        """
        Scan the dataset directory for valid point cloud files.
        
        This method recursively searches the split-specific subdirectory
        for files with supported extensions (.las, .laz) and returns
        a sorted list for consistent ordering across runs.

        Returns:
            List of Path objects pointing to valid point cloud files,
            sorted alphabetically by filename.

        Raises:
            FileNotFoundError: If dataset root or split folder doesn't exist.

        Notes:
            - Only files with extensions in self._extensions are included
            - Files are sorted by name for reproducible ordering
            - Recursive search allows nested directory structures
        """
        root_folder = Path(self._cfg.root)
        split_folder = root_folder / self._split
        if not root_folder.exists() or not root_folder.is_dir():
            raise FileNotFoundError(f'Dataset root does not exist or is not a directory: {root_folder}')
        if not split_folder.exists() or not split_folder.is_dir():
            raise FileNotFoundError(f'Split folder does not exist or is not a directory: {split_folder}')
        
        return sorted(
            [f for f in split_folder.rglob("*") if f.is_file() and f.suffix.lower() in self._extensions],
            key=lambda f: f.name
        )
    
    def _load_file(self, path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load point cloud data and annotations from a single file.
        
        This method reads LAS/LAZ files using laspy and extracts coordinates,
        features, semantic labels, and instance labels according to the
        dataset configuration.

        Args:
            path: Path to the point cloud file to load.

        Returns:
            A tuple (coords, feats, semantic_labels, instance_labels) where:
                - coords: Float array [N_points, 3] with XYZ coordinates
                - feats: Float array [N_points, N_features] with point features
                - semantic_labels: Integer array [N_points] with class labels
                - instance_labels: Integer array [N_points] with instance IDs

        Raises:
            ValueError: If required feature keys are missing from the file
                       or if the file extension is unsupported.

        Notes:
            - Feature extraction uses cfg.feat_keys (e.g., 'intensity')
            - Assumes 'semantic_gt' and 'instance_gt' fields exist in LAS files
            - Currently only supports LAS/LAZ format
        """
        ext = path.suffix.lower()

        coords = ...
        feats = ...
        semantic_labels = ...
        
        if ext in ('.las', '.laz'):
            file = laspy.read(path)

            feats = []
            coords = file.xyz
            for column in self._cfg.feat_keys:
                if column not in file.point_format.dimension_names:
                    raise ValueError(f'Feature key "{column}" not found in file: {path}!')
                
                feats.append(np.array(getattr(file, column))[:, None])

            feats = np.hstack(feats)

            semantic_labels = np.array(file.semantic_gt)
            instance_labels = np.array(file.instance_gt)
            classification_labels = np.array(file.species_gt)
        else:
            raise ValueError(f'Unsopported file extension: {ext}!')

        return coords, feats, semantic_labels, instance_labels, classification_labels
    
    def _augment_data(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply geometric data augmentation to point cloud coordinates.
        
        This method applies random transformations including rotations around
        all three axes, scaling, and axis flipping according to the split
        configuration ranges.

        Args:
            coords: Input coordinates array of shape [N_points, 3].

        Returns:
            Augmented coordinates array of shape [N_points, 3] after applying
            random rotations, scaling, and flipping transformations.

        Notes:
            - Rotation angles are sampled from configured yaw and tilt ranges
            - Scale factor is sampled from configured scale range
            - X and Y axis flipping are applied with 50% probability each
            - All transformations use the instance's random number generator
            - Rotation matrix combines yaw, pitch, and roll rotations
        """
        yaw = np.deg2rad(self._rng.uniform(*self._split_cfg.data_augmentation.yaw_range))
        pitch = np.deg2rad(self._rng.uniform(*self._split_cfg.data_augmentation.tilt_range))
        roll = np.deg2rad(self._rng.uniform(*self._split_cfg.data_augmentation.tilt_range))
        scale = self._rng.uniform(*self._split_cfg.data_augmentation.scale_range)

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
    
    def _get_instance_offsets(self, voxels: np.ndarray, instance_labels: np.ndarray) -> np.ndarray:
        """
        Compute offset vectors pointing from each voxel to its instance center.
        
        For each instance, this method finds the centroid voxel (closest to
        the geometric center) and computes displacement vectors from all
        instance voxels to this center voxel.

        Args:
            voxels: Voxel coordinates array of shape [N_voxels, 3].
            instance_labels: Instance ID array of shape [N_voxels] where
                0 indicates background and positive values indicate instances.

        Returns:
            Offset vectors array of shape [N_voxels, 3] where each element
            contains the displacement vector to the instance center.
            Background voxels (label 0) have zero offset vectors.

        Notes:
            - Instance centers are defined as the voxel closest to geometric centroid
            - Background voxels (instance_id == 0) are skipped
            - Offset computation: center_voxel - current_voxel
            - Used for training the offset prediction head
        """
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

        return offsets
    
    def _get_centroid_scores(self, voxels: np.ndarray, instance_labels: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmaps for centroid detection supervision.
        
        This method creates soft target heatmaps where each instance contributes
        a Gaussian distribution centered on its center voxel. The Gaussian spread
        is adaptive based on instance extent, providing scale-aware supervision.

        Args:
            voxels: Voxel coordinates array of shape [N_voxels, 3].
            instance_labels: Instance ID array of shape [N_voxels] where
                0 indicates background and positive values indicate instances.

        Returns:
            Heatmap array of shape [N_voxels, 1] with confidence scores
            between 0 and 1. Peak values of 1.0 occur at instance centers,
            with Gaussian fall-off based on distance and instance size.

        Notes:
            - Sigma values are computed as extent / centroid_sigma_divisor
            - Sigma is clamped between centroid_sigma_min and centroid_sigma_max
            - Gaussians are truncated beyond 3-sigma (norm2 < 9.0)
            - Multiple instances can contribute to the same voxel (max taken)
            - Background voxels (instance_id == 0) remain at 0.0
        """
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

            extent = pts.max(axis=0) - pts.min(axis=0) + 1.0
            sigma = np.clip(extent / self._cfg.centroid_sigma_divisor, self._cfg.centroid_sigma_min, self._cfg.centroid_sigma_max)

            diff = pts - ctr_voxel
            norm2 = np.sum((diff / sigma) ** 2, axis=1)
            mask = norm2 < 9.0

            values = np.exp(-0.5 * norm2[mask])
            cur = heat_map[idx[mask], 0]
            np.maximum(cur, values, out=cur)
            heat_map[idx[mask], 0] = cur

        return heat_map
    
    def _preprocess(self, idx: int) -> Dict:
        """
        Complete preprocessing pipeline for a single sample.
        
        This method orchestrates the full data loading and preprocessing pipeline:
        file loading, optional augmentation, voxelization, instance filtering,
        supervision signal generation, and tensor conversion.

        Args:
            idx: Sample index. If idx >= len(files), augmentation is applied
                using modulo indexing for virtual dataset expansion.

        Returns:
            Dictionary containing preprocessed data with keys:
                - 'inputs': SparseTensor with voxel coordinates and features
                - 'semantic_labels': SparseTensor with per-voxel class labels
                - 'centroid_score_labels': SparseTensor with centroid heatmaps
                - 'offset_labels': SparseTensor with offset vectors
                - 'instance_labels': SparseTensor with instance IDs

        Notes:
            - Coordinates are normalized to start from origin
            - Voxelization removes duplicate points at the same grid cell
            - Instance filtering removes instances smaller than min_tree_voxels
            - Instance IDs are compacted to consecutive integers starting from 0
            - All outputs share the same voxel coordinate space
        """
        coords, feat, semantic_labels, instance_labels, classification_labels = self._load_file(self._files[idx % len(self._files)])
        current_file = self._files[idx % len(self._files)]
        if idx >= len(self._files):
            coords = self._augment_data(coords)

        coords -= np.min(coords, axis=0, keepdims=True)

        voxels, indices = sparse_quantize(coords, self._cfg.voxel_size, return_index=True)
        feat = feat[indices]
        semantic_labels = semantic_labels[indices]
        instance_labels = instance_labels[indices]
        classification_labels = classification_labels[indices]

        sizes = np.bincount(instance_labels)[instance_labels]
        size_mask = (sizes >= self._cfg.min_tree_voxels) # & ((instance_labels != 0) | (semantic_labels == 0)) 
        voxels = voxels[size_mask]
        feat = feat[size_mask]
        semantic_labels = semantic_labels[size_mask]
        instance_labels = instance_labels[size_mask]
        classification_labels = classification_labels[size_mask]
        
        _, instance_labels = np.unique(instance_labels, return_inverse=True)

        centroid_score_labels = self._get_centroid_scores(voxels, instance_labels)
        offset_labels = self._get_instance_offsets(voxels, instance_labels)

        voxels = torch.tensor(voxels, dtype=torch.int)
        feat = torch.tensor(feat.astype(np.float32), dtype=torch.float)

        semantic_labels = torch.tensor(semantic_labels, dtype=torch.long)
        centroid_score_labels = torch.tensor(centroid_score_labels, dtype=torch.float)
        offset_labels = torch.tensor(offset_labels, dtype=torch.float)
        instance_labels = torch.tensor(instance_labels, dtype=torch.long)
        classification_labels = torch.tensor(classification_labels, dtype=torch.long)

        inputs = SparseTensor(coords=voxels, feats=feat)
        
        semantic_labels = SparseTensor(coords=voxels, feats=semantic_labels)
        centroid_score_labels = SparseTensor(coords=voxels, feats=centroid_score_labels)
        offset_labels = SparseTensor(coords=voxels, feats=offset_labels)
        instance_labels = SparseTensor(coords=voxels, feats=instance_labels)
        classification_labels = SparseTensor(coords=voxels, feats=classification_labels)

        return {
            "inputs": inputs,
            "semantic_labels": semantic_labels,
            "centroid_score_labels": centroid_score_labels,
            "offset_labels": offset_labels,
            "instance_labels": instance_labels,
            "classification_labels": classification_labels
        }
    
    def __len__(self) -> int:
        return self._len
    
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
