import numpy as np
from plyfile import PlyData

from .panoramix3D_dataset import Panoramix3DDataset


class ScanNetDataset(Panoramix3DDataset):
    def __init__(self, cfg, split = 'train'):
        super().__init__(cfg, split)

        self._extensions = ('.ply')


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
        file = PlyData.read(path)['vertex'].data

        feats = []
        coords = np.vstack([
            file['x'],
            file['y'],
            file['z']
        ]).T
        for column in self._cfg.feat_keys:
            if column not in file.dtype.names:
                raise ValueError(f'Feature key "{column}" not found in file: {path}!')
            
            feats.append(np.array(getattr(file, column))[:, None])

        feats = np.hstack(feats)

        semantic_labels = np.array(file.semantic_gt)
        instance_labels = np.array(file.instance_gt)
        classification_labels = np.array(file.species_gt)

        return coords, feats, semantic_labels, instance_labels, classification_labels