import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from panoramix3D.config import DatasetConfig
from panoramix3D.datasets import EHydroDataset, ScanNetDataset, S3DISDataset


class Visualizer:
    def __init__(self, cfg, split='train'):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid split: {split}. Must be one of "train", "val", or "test".')
        
        self._cfg = DatasetConfig.from_yaml(cfg)
        if self._cfg.name == 'EHydroDataset':
            self._dataset_class = EHydroDataset
        elif self._cfg.name == 'ScanNetDataset':
            self._dataset_class = ScanNetDataset
        elif self._cfg.name == 'S3DISDataset':
            self._dataset_class = S3DISDataset
        else:
            raise ValueError(f'Unsupported dataset: {self._cfg.name}!')
        
        self._dataset = self._dataset_class(self._cfg, split=split)
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._vis.create_window(window_name='Panoramix3D Dataset Visualizer', width=800, height=600)
        self._vis.register_key_callback(256, lambda vis: vis.close())

        self._vis.register_key_callback(ord('Q'), self._prev_scene)
        self._vis.register_key_callback(ord('E'), self._next_scene)

        self._vis.register_key_callback(ord('Z'), self._show_rgb)
        self._vis.register_key_callback(ord('A'), self._show_semantics)
        self._vis.register_key_callback(ord('S'), self._show_centroids)
        self._vis.register_key_callback(ord('D'), self._show_offsets)
        self._vis.register_key_callback(ord('F'), self._show_instances)
        self._vis.register_key_callback(ord('G'), self._show_classification)

        self._current_idx = 0
        self._current_scene = self._dataset[self._current_idx]

    def _update_geometry(self, vis, geometry):
        view_ctl = vis.get_view_control()
        params = view_ctl.convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        vis.add_geometry(geometry)
        view_ctl.convert_from_pinhole_camera_parameters(params)
        vis.update_renderer()

    def _prev_scene(self, vis):
        if self._current_idx <= 0:
            return
        
        self._current_idx -= 1
        self._current_scene = self._dataset[self._current_idx]
        self._show_rgb(vis)

    def _next_scene(self, vis):
        if self._current_idx >= len(self._dataset) -1:
            return
        
        self._current_idx += 1
        self._current_scene = self._dataset[self._current_idx]
        self._show_rgb(vis)

    def _show_rgb(self, vis):
        inputs = self._current_scene['inputs']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(inputs.F.cpu().numpy() / 255.0)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._update_geometry(vis, voxel_grid)

    def _show_semantics(self, vis):
        inputs = self._current_scene['inputs']
        semantic_labels = self._current_scene['semantic_labels']

        class_colormap = np.array([semantic_cls.color for semantic_cls in self._cfg.semantic_classes]).astype(np.float32) / 255.0
        colors = class_colormap[semantic_labels.F.cpu().numpy()]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._update_geometry(vis, voxel_grid)

    def _show_centroids(self, vis):
        inputs = self._current_scene['inputs']
        centroid_score_labels = self._current_scene['centroid_score_labels']

        scores = centroid_score_labels.F.cpu().numpy().squeeze(-1)
        rgb = inputs.F.cpu().numpy() / 255.0
        jet_colors = plt.get_cmap('jet')(scores)[:, :3]
        mask = scores == 0
        colors = np.where(mask[:, None], rgb, jet_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._update_geometry(vis, voxel_grid)

    def _show_offsets(self, vis):
        inputs = self._current_scene['inputs']
        semantic_labels = self._current_scene['semantic_labels']
        offset_labels = self._current_scene['offset_labels']

        class_colormap = np.array([semantic_cls.color for semantic_cls in self._cfg.semantic_classes]).astype(np.float32) / 255.0
        colors = class_colormap[semantic_labels.F.cpu().numpy()]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy() + offset_labels.F.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._update_geometry(vis, voxel_grid)

    def _show_instances(self, vis):
        inputs = self._current_scene['inputs']
        instance_labels = self._current_scene['instance_labels']

        unique_instances = np.unique(instance_labels.F.cpu().numpy())
        np.random.seed(0)
        instance_colormap = np.random.rand(len(unique_instances) + 1, 3)
        colors = instance_colormap[instance_labels.F.cpu().numpy()]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._update_geometry(vis, voxel_grid)

    def _show_classification(self, vis):
        inputs = self._current_scene['inputs']
        classification_labels = self._current_scene['classification_labels']

        class_colormap = np.array([instance_cls.color for instance_cls in self._cfg.instance_classes]).astype(np.float32) / 255.0
        colors = class_colormap[classification_labels.F.cpu().numpy()]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._update_geometry(vis, voxel_grid)

    def run(self):
        inputs = self._current_scene['inputs']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inputs.C.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(inputs.F.cpu().numpy() / 255.0)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._vis.add_geometry(voxel_grid)

        self._vis.run()
        self._vis.destroy_window()