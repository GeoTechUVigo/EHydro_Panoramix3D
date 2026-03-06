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

        self._print_dataset_info()

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

    def _print_dataset_info(self):
        # Print class distribution for all scenes in the dataset
        print("Dataset Class Distribution:")
        total_points = 0
        class_counts = np.zeros(len(self._cfg.semantic_classes), dtype=int)
        sqrt_areas = []
        for i in range(len(self._dataset._files)):
            scene = self._dataset[i]
            semantic_labels = scene['semantic_labels']
            unique, counts = np.unique(semantic_labels.F.cpu().numpy(), return_counts=True)
            total_points += len(semantic_labels.F.cpu().numpy())
            for cls_id, count in zip(unique, counts):
                class_counts[cls_id] += count

            C = scene['inputs'].C.cpu().numpy()  # [N, 3] integer voxel coords
            x_extent = (C[:, 0].max() - C[:, 0].min()) * self._cfg.voxel_size
            y_extent = (C[:, 1].max() - C[:, 1].min()) * self._cfg.voxel_size
            sqrt_areas.append(np.sqrt(x_extent * y_extent))

        for cls_id, count in enumerate(class_counts):
            cls_name = self._cfg.semantic_classes[cls_id].name
            percentage = (count / total_points) * 100
            print(f"  {cls_name} (ID: {cls_id}): {count} points ({percentage:.2f}%)")

        sqrt_areas = np.array(sqrt_areas)
        print(f"\nScene size distribution (√2D area) — {len(sqrt_areas)} scenes:")
        print(f"  min:    {sqrt_areas.min():.1f} m")
        print(f"  max:    {sqrt_areas.max():.1f} m")
        print(f"  mean:   {sqrt_areas.mean():.1f} m")
        print(f"  median: {np.median(sqrt_areas):.1f} m")

        large = []
        for i in range(len(sqrt_areas)):
            if sqrt_areas[i] > 15.0:
                entry = self._dataset._files[i]
                path = entry[0] if isinstance(entry, tuple) else entry
                large.append((path, sqrt_areas[i]))
        print(f"\nScenes with √(2D area) > 10 m: {len(large)}")
        for path, area in sorted(large, key=lambda x: -x[1]):
            print(f"  {area:.1f} m  —  {path}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sqrt_areas, bins=30, edgecolor='black')
        ax.set_xlabel('√(2D area) [m]')
        ax.set_ylabel('Scenes')
        ax.set_title('Scene size distribution — train split')
        ax.axvline(sqrt_areas.mean(), color='red', linestyle='--', label=f'mean = {sqrt_areas.mean():.1f} m')
        ax.axvline(np.median(sqrt_areas), color='orange', linestyle='--', label=f'median = {np.median(sqrt_areas):.1f} m')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _get_colors(self, features):
        if set(['red', 'green', 'blue']).issubset(self._cfg.feat_keys):
            red_idx = self._cfg.feat_keys.index('red')
            green_idx = self._cfg.feat_keys.index('green')
            blue_idx = self._cfg.feat_keys.index('blue')
            rgb_values = features[:, [red_idx, green_idx, blue_idx]]

            max_value = rgb_values.max()
            if max_value > 255:
                colors = rgb_values / 65535.0
            else:
                colors = rgb_values / 255.0
        elif 'intensity' in self._cfg.feat_keys:
            intensity_idx = self._cfg.feat_keys.index('intensity')
            intensity = features[:, intensity_idx]
            max_intensity = intensity.max()
            if max_intensity > 255:
                colors = plt.get_cmap('gray')(intensity / 65535.0)[:, :3]
            else:
                colors = plt.get_cmap('gray')(intensity / 255.0)[:, :3]
        else:
            colors = np.zeros((features.shape[0], 3))

        return o3d.utility.Vector3dVector(colors)

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
        pcd.colors = self._get_colors(inputs.F.cpu().numpy())

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

        unique, counts = np.unique(semantic_labels.F.cpu().numpy(), return_counts=True)
        print("Semantic Class Distribution:")
        for cls_id, count in zip(unique, counts):
            cls_name = self._cfg.semantic_classes[cls_id].name
            percentage = (count / len(semantic_labels.F.cpu().numpy())) * 100
            print(f"  {cls_name} (ID: {cls_id}): {count} points ({percentage:.2f}%)")

    def _show_centroids(self, vis):
        inputs = self._current_scene['inputs']
        centroid_score_labels = self._current_scene['centroid_score_labels']

        scores = centroid_score_labels.F.cpu().numpy().squeeze(-1)
        rgb = self._get_colors(inputs.F.cpu().numpy())
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
        pcd.colors = o3d.utility.Vector3dVector(self._get_colors(inputs.F.cpu().numpy()))

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)
        self._vis.add_geometry(voxel_grid)

        self._vis.run()
        self._vis.destroy_window()