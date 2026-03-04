import numpy as np
import re

from pathlib import Path
from .panoramix3D_dataset import Panoramix3DDataset


class S3DISDataset(Panoramix3DDataset):
    def __init__(self, cfg, split = 'train'):
        super().__init__(cfg, split)

        self._extensions = ('.npy')


    def _scan_files(self):
        """
        Scan the dataset directory for point cloud files corresponding to the specified split.
        
        This method traverses the dataset directory structure to identify all files with supported
        extensions (e.g., .npy) that belong to the current split (train/val/test). It collects
        their paths into a list for subsequent loading.

        Returns:
            A list of file paths for all point cloud files in the specified split.
        """

        root_folder = Path(self._cfg.root)
        split_folder = root_folder / self._split
        if not root_folder.exists() or not root_folder.is_dir():
            raise FileNotFoundError(f'Dataset root does not exist or is not a directory: {root_folder}')
        if not split_folder.exists() or not split_folder.is_dir():
            raise FileNotFoundError(f'Split folder does not exist or is not a directory: {split_folder}')
        
        area_pattern = re.compile(r"Area_\d+")
        return sorted(
            [f for f in split_folder.rglob("*") if f.is_dir() and not area_pattern.fullmatch(f.name)],
            key=lambda f: f.name
        )


    def _load_file(self, path):
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

        coords = np.load(path / 'coord.npy')
        colors = np.load(path / 'color.npy')

        feats = []
        for column in self._cfg.feat_keys:
            if column == 'red':
                feats.append(colors[:, 0:1])
            elif column == 'green':
                feats.append(colors[:, 1:2])
            elif column == 'blue':
                feats.append(colors[:, 2:3])
            else:
                raise ValueError(f'Unsupported feature key "{column}" in config for file: {path}!')
            
        feats = np.hstack(feats)
        semantic_labels = np.load(path / 'segment.npy').squeeze(-1)
        instance_labels = np.load(path / 'instance.npy').squeeze(-1) + 1

        # Remap instance labels to consecutive indices starting from 1
        unique_instances = np.unique(instance_labels)
        unique_instances = unique_instances[unique_instances > 0]  # Exclude instance 0 (background)
        instance_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_instances, start=1)}
        instance_mapping[0] = 0  # Keep 0 as background
        
        instance_labels_remap = np.zeros_like(instance_labels)
        for old_id, new_id in instance_mapping.items():
            instance_labels_remap[instance_labels == old_id] = new_id
        
        # Set instance labels to 0 for stuff classes (not in foreground_classes)
        if self._cfg.foreground_classes:
            foreground_mask = np.isin(semantic_labels, self._cfg.foreground_classes)
            instance_labels_remap[~foreground_mask] = 0
        
        classification_labels = np.ones_like(instance_labels_remap)

        return coords, feats, semantic_labels, instance_labels_remap, classification_labels