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

        If a scene extends more than 15 m along X or Y, it is recursively halved along that
        axis until every tile is ≤ 15 m wide/tall.  Each resulting tile is stored as a tuple
        (path, x_low, x_high, y_low, y_high) so that _load_file can crop to it.

        Returns:
            A list of (path, x_low, x_high, y_low, y_high) tuples, one per tile.
        """

        root_folder = Path(self._cfg.root)
        split_folder = root_folder / self._split
        if not root_folder.exists() or not root_folder.is_dir():
            raise FileNotFoundError(f'Dataset root does not exist or is not a directory: {root_folder}')
        if not split_folder.exists() or not split_folder.is_dir():
            raise FileNotFoundError(f'Split folder does not exist or is not a directory: {split_folder}')

        area_pattern = re.compile(r"Area_\d+")
        dirs = sorted(
            [f for f in split_folder.rglob("*") if f.is_dir() and not area_pattern.fullmatch(f.name)],
            key=lambda f: f.name
        )

        entries = []
        for d in dirs:
            coords = np.load(d / 'coord.npy')
            x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
            y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

            # number of tiles per axis: keep halving until each tile is ≤ 15 m
            nx = 1
            while (x_max - x_min) / nx > 15.0:
                nx *= 2
            ny = 1
            while (y_max - y_min) / ny > 15.0:
                ny *= 2

            x_step = (x_max - x_min) / nx
            y_step = (y_max - y_min) / ny

            for ix in range(nx):
                xl = x_min + ix * x_step
                # last tile: extend slightly beyond x_max to capture boundary points
                xh = x_min + (ix + 1) * x_step if ix < nx - 1 else x_max + 1.0
                for iy in range(ny):
                    yl = y_min + iy * y_step
                    yh = y_min + (iy + 1) * y_step if iy < ny - 1 else y_max + 1.0
                    entries.append((d, xl, xh, yl, yh))

        return entries


    def _load_file(self, path):
        """
        Load point cloud data and annotations from a single file (or tile).

        Args:
            path: Either a plain Path to a scene directory, or a tuple
                  (Path, x_low, x_high, y_low, y_high) produced by _scan_files
                  when the scene was tiled because it exceeded 15 m in X or Y.

        Returns:
            A tuple (coords, feats, semantic_labels, instance_labels, classification_labels).
        """

        if isinstance(path, tuple):
            d, x_low, x_high, y_low, y_high = path
        else:
            d = path
            x_low = x_high = y_low = y_high = None

        coords = np.load(d / 'coord.npy')
        colors = np.load(d / 'color.npy')

        # crop to tile bounds when the scene was split
        if x_low is not None:
            tile_mask = (
                (coords[:, 0] >= x_low) & (coords[:, 0] < x_high) &
                (coords[:, 1] >= y_low) & (coords[:, 1] < y_high)
            )
            coords = coords[tile_mask]
            colors = colors[tile_mask]

        feats = []
        for column in self._cfg.feat_keys:
            if column == 'red':
                feats.append(colors[:, 0:1])
            elif column == 'green':
                feats.append(colors[:, 1:2])
            elif column == 'blue':
                feats.append(colors[:, 2:3])
            else:
                raise ValueError(f'Unsupported feature key "{column}" in config for file: {d}!')

        feats = np.hstack(feats)

        semantic_labels = np.load(d / 'segment.npy').squeeze(-1)
        instance_labels = np.load(d / 'instance.npy').squeeze(-1) + 1

        if x_low is not None:
            semantic_labels = semantic_labels[tile_mask]
            instance_labels = instance_labels[tile_mask]

        instance_labels[~np.isin(semantic_labels, self._cfg.foreground_classes)] = 0
        _, instance_labels = np.unique(instance_labels, return_inverse=True)
        classification_labels = np.ones_like(instance_labels)
        classification_labels[instance_labels == 0] = 0

        return coords, feats, semantic_labels, instance_labels, classification_labels