import laspy
import numpy as np
import shutil

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

from utils import load_point_clouds, load_las_file, get_las_files, chunkerize_clean, chunkerize_four


class DatasetProcessor:
    def __init__(self):
        self.split_pct = [0.6, 0.2, 0.2]
        
        # Augmentation parameters
        self._augmented_per_cloud = 15  # Number of augmented clouds per ground
        self._min_trees_per_cloud = 8  # Min trees per augmented cloud
        self._max_trees_per_cloud = 30  # Max trees per augmented cloud
        self._min_distance = 1.5  # Min distance between trees (meters)
        self.border_margin = 2.0  # Min distance from borders (meters)
        
        datasets_folder = Path.home() / 'Panoramix3D_data' / 'datasets'
        
        self.paths = {
            'for_instance_in': datasets_folder / 'FORinstance_dataset/raw',
            'for_instance_out': datasets_folder / 'FORinstance_dataset/processed',
            'for_instance_big_in': datasets_folder / 'FORinstance_big_dataset/raw',
            'for_instance_big_out': datasets_folder / 'FORinstance_big_dataset/processed',
            'mixed': datasets_folder / 'MixedDataset'
        }

        self.class_mappings = {
            'mixed': {
                'terrain': 0,
                'stem': 1,
                'canopy': 2
            },
            'for_instance': {
                'unclassified': 0,
                'low_vegetation': 1,
                'terrain': 2,
                'out_points': 3,
                'stem': 4,
                'live_branches': 5,
                'woody_branches': 6
            }
        }

        # Clean and setup directories
        for dir_path in [self.paths['for_instance_out'], self.paths['for_instance_big_out'], self.paths['mixed']]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"Cleaned: {dir_path}")
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'val', 'test', 'train_augmented']:
            (self.paths['for_instance_out'] / split).mkdir(exist_ok=True)
            
        for split in ['train', 'val', 'test']:
            (self.paths['mixed'] / split).mkdir(exist_ok=True)

    def _split_files_or_chunks(self, items: List, description: str = "items") -> Tuple[List, List, List]:
        """Single split function for both files and chunks."""
        items = sorted(items, key=lambda x: str(x) if hasattr(x, '__str__') else str(x[1]))
        n_items = len(items)
        
        train_count = round(n_items * self.split_pct[0])
        val_count = round(n_items * self.split_pct[1])
        test_count = n_items - train_count - val_count
        
        if test_count < 0:
            val_count += test_count
            test_count = 0
        
        print(f"Splitting {n_items} {description}: {train_count} train, {val_count} val, {test_count} test")
        
        return items[:train_count], items[train_count:train_count + val_count], items[train_count + val_count:]

    def _normalize_intensity_and_save(self, input_path: Path, output_path: Path):
        """Load file, normalize intensity, and save to new location."""
        file = laspy.read(input_path)
        
        # Normalize coordinates to origin
        min_coords = np.array([file.x.min(), file.y.min(), file.z.min()], dtype=np.int64)
        mins_world = min_coords * file.header.scales + file.header.offsets
        file.header.offsets -= mins_world
        
        # Add extra dimensions if they don't exist
        existing_dims = [dim.name for dim in file.point_format.extra_dimensions]
        
        extra_dims = []
        if "norm_intensity" not in existing_dims:
            extra_dims.append(laspy.ExtraBytesParams(name="norm_intensity", type=np.float32))
        if "semantic_gt" not in existing_dims:
            extra_dims.append(laspy.ExtraBytesParams(name="semantic_gt", type=np.int16))
        if "instance_gt" not in existing_dims:
            extra_dims.append(laspy.ExtraBytesParams(name="instance_gt", type=np.int16))
        
        if extra_dims:
            file.add_extra_dims(extra_dims)
        
        # Normalize intensity
        intensity = np.array(file.intensity)
        if len(intensity) > 0:
            min_intensity = np.min(intensity)
            max_intensity = np.max(intensity)
            if max_intensity > min_intensity:
                file.norm_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)
            else:
                file.norm_intensity = np.zeros_like(intensity, dtype=np.float32)
        
        # Save to output location
        file.write(output_path)

    def _copy_files_to_splits(self, files: List[Path], base_path: Path):
        """Copy files to train/val/test subdirectories with intensity normalization."""
        train_files, val_files, test_files = self._split_files_or_chunks(files, "files")
        
        for file_list, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for counter, file_path in enumerate(tqdm(file_list, desc=f"Copying {split_name}")):
                stem, suffix = file_path.stem, file_path.suffix
                unique_name = f"{stem}_{counter:03d}{suffix}"
                output_path = base_path / split_name / unique_name
                self._normalize_intensity_and_save(file_path, output_path)

    def augment_data(self, point_clouds):
        """Generate augmented training data by placing random trees on ground."""
        import random
        from scipy.spatial.distance import cdist
        
        # 1. Accumulate all ground points from all clouds
        ground_clouds = []
        for file in point_clouds:
            # Ground points: treeID == 0 AND terrain/low_vegetation classification
            terrain_classes = [
                self.class_mappings['for_instance']['terrain'],
                self.class_mappings['for_instance']['low_vegetation']
            ]
            ground_mask = (file.treeID == 0) & np.isin(file.classification, terrain_classes)
            if np.any(ground_mask):
                ground_points = file.points[ground_mask]
                ground_clouds.append(ground_points)
        
        # 2. Accumulate all trees from all clouds - keep it simple
        tree_data = []  # Store (points, center_x, center_y, min_z) for each tree
        for file in point_clouds:
            unique_tree_ids = np.unique(file.treeID)
            for tree_id in unique_tree_ids:
                if tree_id != 0:  # Skip ground
                    tree_mask = file.treeID == tree_id
                    tree_points = file.points[tree_mask]
                    
                    # Calculate centering values but don't modify the points yet
                    center_x = np.mean(tree_points.x)
                    center_y = np.mean(tree_points.y)
                    min_z = np.min(tree_points.z)
                    
                    tree_data.append((tree_points, center_x, center_y, min_z))
        
        print(f"Found {len(ground_clouds)} ground patches and {len(tree_data)} trees for augmentation")
        
        if len(tree_data) == 0:
            print("No trees found for augmentation")
            return
        
        augmented_counter = 0
        
        # 3. For each ground, generate M augmented clouds
        for ground_idx, ground_points in enumerate(tqdm(ground_clouds, desc="Augmenting grounds")):
            for m in range(self._augmented_per_cloud):
                # Get ground boundaries
                x_min, x_max = np.min(ground_points.x), np.max(ground_points.x)
                y_min, y_max = np.min(ground_points.y), np.max(ground_points.y)
                
                # Filter ground points away from borders
                border_mask = ((ground_points.x > x_min + self.border_margin) & 
                              (ground_points.x < x_max - self.border_margin) &
                              (ground_points.y > y_min + self.border_margin) & 
                              (ground_points.y < y_max - self.border_margin))
                
                valid_ground_points = ground_points[border_mask]
                if len(valid_ground_points) == 0:
                    continue
                
                # 4. Choose number of trees and placement points
                n_trees = random.randint(self._min_trees_per_cloud, self._max_trees_per_cloud)
                placement_points = []
                
                for _ in range(n_trees * 10):  # Try multiple times to place trees
                    if len(placement_points) >= n_trees:
                        break
                    
                    # Random point from valid ground
                    candidate_idx = random.randint(0, len(valid_ground_points) - 1)
                    candidate_point = valid_ground_points[candidate_idx]
                    candidate_pos = np.array([valid_ground_points.x[candidate_idx], valid_ground_points.y[candidate_idx]])
                    
                    # Check distance to existing placements
                    if len(placement_points) == 0:
                        placement_points.append((candidate_pos, valid_ground_points.z[candidate_idx]))
                    else:
                        existing_positions = np.array([p[0] for p in placement_points])
                        distances = cdist([candidate_pos], existing_positions)[0]
                        if np.min(distances) >= self._min_distance:
                            placement_points.append((candidate_pos, valid_ground_points.z[candidate_idx]))
                
                if len(placement_points) == 0:
                    continue
                
                # 5. Place trees and create augmented cloud
                # Start by collecting all point data
                all_points_data = [(ground_points, None, None, None)]  # Ground: (points, None, None, None)
                all_tree_ids = [np.zeros(len(ground_points), dtype=np.int32)]
                
                current_tree_id = 1
                
                for pos, ground_z in placement_points:
                    # Choose random tree
                    tree_points, center_x, center_y, min_z = random.choice(tree_data)
                    
                    # Get coordinates and center them
                    tree_x = np.array(tree_points.x) - center_x
                    tree_y = np.array(tree_points.y) - center_y
                    tree_z = np.array(tree_points.z) - min_z
                    
                    # Apply random rotation
                    z_rotation = random.uniform(0, 2 * np.pi)
                    x_rotation = random.uniform(-0.034, 0.034)  # ±2 degrees
                    y_rotation = random.uniform(-0.034, 0.034)
                    
                    # Rotate around Z axis (most important)
                    cos_z, sin_z = np.cos(z_rotation), np.sin(z_rotation)
                    new_x = tree_x * cos_z - tree_y * sin_z
                    new_y = tree_x * sin_z + tree_y * cos_z
                    tree_x = new_x
                    tree_y = new_y
                    
                    # Small rotations in X and Y
                    if abs(x_rotation) > 1e-6:
                        cos_x, sin_x = np.cos(x_rotation), np.sin(x_rotation)
                        new_y = tree_y * cos_x - tree_z * sin_x
                        new_z = tree_y * sin_x + tree_z * cos_x
                        tree_y = new_y
                        tree_z = new_z
                    
                    if abs(y_rotation) > 1e-6:
                        cos_y, sin_y = np.cos(y_rotation), np.sin(y_rotation)
                        new_x = tree_x * cos_y + tree_z * sin_y
                        new_z = -tree_x * sin_y + tree_z * cos_y
                        tree_x = new_x
                        tree_z = new_z
                    
                    # Translate to final position
                    tree_x += pos[0]
                    tree_y += pos[1]
                    tree_z += ground_z
                    
                    # Store the original tree points and transformed coordinates
                    all_points_data.append((tree_points, tree_x, tree_y, tree_z))
                    all_tree_ids.append(np.full(len(tree_points), current_tree_id, dtype=np.int32))
                    current_tree_id += 1
                
                # Create final augmented file by manually building arrays
                template_file = point_clouds[0]
                out = laspy.create(point_format=template_file.point_format, 
                                  file_version=template_file.header.version)
                out.header.scales = template_file.header.scales
                out.header.offsets = template_file.header.offsets
                
                # Calculate total number of points
                total_points = sum(len(item[0]) for item in all_points_data)
                
                # Initialize coordinate arrays
                out.x = np.zeros(total_points, dtype=np.float64)
                out.y = np.zeros(total_points, dtype=np.float64)
                out.z = np.zeros(total_points, dtype=np.float64)
                
                # Get all available fields from ground points
                ground_points = all_points_data[0][0]
                available_fields = []
                
                # Basic LAS fields and extra dimensions
                for field in ['intensity', 'return_number', 'number_of_returns', 'scan_direction_flag', 
                             'edge_of_flight_line', 'classification', 'scan_angle_rank', 'treeSP',
                             'user_data', 'point_source_id', 'norm_intensity']:
                    if hasattr(ground_points, field):
                        available_fields.append(field)
                
                # Initialize arrays for available fields
                field_arrays = {}
                for field in available_fields:
                    sample_data = getattr(ground_points, field)
                    field_arrays[field] = np.zeros(total_points, dtype=sample_data.dtype)
                
                # Fill arrays by copying from each point group
                current_idx = 0
                for points_group, transform_x, transform_y, transform_z in all_points_data:
                    n_points = len(points_group)
                    end_idx = current_idx + n_points
                    
                    # Copy coordinates - use transforms if available, otherwise original
                    if transform_x is not None:  # Transformed tree
                        out.x[current_idx:end_idx] = transform_x
                        out.y[current_idx:end_idx] = transform_y
                        out.z[current_idx:end_idx] = transform_z
                    else:  # Ground points
                        out.x[current_idx:end_idx] = points_group.x
                        out.y[current_idx:end_idx] = points_group.y
                        out.z[current_idx:end_idx] = points_group.z
                    
                    # Copy all other available fields
                    for field in available_fields:
                        if hasattr(points_group, field):
                            field_arrays[field][current_idx:end_idx] = getattr(points_group, field)
                    
                    current_idx = end_idx
                
                # Set all fields in output
                for field, array in field_arrays.items():
                    setattr(out, field, array)
                
                # Set tree IDs
                out.treeID = np.concatenate(all_tree_ids)
                
                # Save augmented file
                output_path = self.paths['for_instance_out'] / 'train_augmented' / f'augmented_{augmented_counter:04d}.las'
                out.write(output_path)
                augmented_counter += 1
        
        print(f"Generated {augmented_counter} augmented training clouds")

    def _process_file_to_chunks(self, file, file_counter: int, chunk_function, output_folder: Path, filename_prefix: str):
        """Process a single file into chunks and save."""
        # Clean and prepare file
        invalid_classes = [self.class_mappings['for_instance']['out_points'], 
                          self.class_mappings['for_instance']['unclassified']]
        mask = ~np.isin(file.classification, invalid_classes)
        file.points = file.points[mask]
        
        # Remap semantic labels
        remap = np.copy(file.classification)
        for_inst = self.class_mappings['for_instance']
        mixed = self.class_mappings['mixed']
        
        remap = np.where(file.classification == for_inst['low_vegetation'], mixed['terrain'], remap)
        remap = np.where(file.classification == for_inst['terrain'], mixed['terrain'], remap)
        remap = np.where(file.classification == for_inst['stem'], mixed['stem'], remap)
        remap = np.where(file.classification == for_inst['live_branches'], mixed['canopy'], remap)
        remap = np.where(file.classification == for_inst['woody_branches'], mixed['canopy'], remap)
        
        file.semantic_gt = remap
        file.instance_gt = file.treeID
        
        # Generate chunks
        chunk_masks = chunk_function(file)
        chunks = []
        
        for j, mask in enumerate(chunk_masks):
            unique_vals, inv = np.unique(file.treeID[mask], return_inverse=True)
            
            # Skip chunks that have no instances (only background/ground)
            non_zero_instances = unique_vals[unique_vals > 0]
            if len(non_zero_instances) == 0:
                continue  # Skip this chunk
            
            out = laspy.create(point_format=file.point_format, file_version=file.header.version)
            out.header.scales = file.header.scales
            out.header.offsets = file.header.offsets
            out.points = file.points[mask]
            out.instance_gt = inv
            
            filename = f'{filename_prefix}_{file_counter}_{j}.las'
            
            if output_folder:  # Direct save
                out.write(output_folder / filename)
            else:  # Return for later processing
                chunks.append((out, filename))
        
        return chunks

    def __call__(self):
        """Execute the complete processing pipeline."""
        print("Processing datasets...")
        
        # 1. Split and copy FORinstance files
        all_files = sorted([f for f in self.paths['for_instance_in'].rglob("*") 
                           if f.is_file() and f.suffix.lower() in ('.laz', '.las')], key=lambda f: f.name)
        self._copy_files_to_splits(all_files, self.paths['for_instance_out'])
        
        # 2. Copy big dataset files
        big_files = sorted([f for f in self.paths['for_instance_big_in'].rglob("*") 
                           if f.is_file() and f.suffix.lower() in ('.laz', '.las')], key=lambda f: f.name)
        for counter, file_path in enumerate(tqdm(big_files, desc="Copying big scenes")):
            stem, suffix = file_path.stem, file_path.suffix
            unique_name = f"{stem}_{counter:03d}{suffix}"
            output_path = self.paths['for_instance_big_out'] / unique_name
            self._normalize_intensity_and_save(file_path, output_path)

        # 3. Augment training data with synthetic clouds
        train_files = get_las_files(self.paths['for_instance_out'] / 'train')
        if train_files:
            train_clouds = [load_las_file(f) for f in train_files]
            self.augment_data(train_clouds)
        
        # Copy original training files to train_augmented (they already have norm_intensity)
        for file_path in tqdm(train_files, desc="Copying original training data"):
            shutil.copy2(file_path, self.paths['for_instance_out'] / 'train_augmented' / file_path.name)

        # 4. Process FORinstance splits to chunks
        for split in ['train_augmented', 'val', 'test']:
            output_split = 'train' if split == 'train_augmented' else split
            split_files = get_las_files(self.paths['for_instance_out'] / split)
            
            for file_counter, file_path in enumerate(tqdm(split_files, desc=f"Processing {split}")):
                file = load_las_file(file_path)
                self._process_file_to_chunks(file, file_counter, chunkerize_four, 
                                           self.paths['mixed'] / output_split, 'plot_FORinstance')

        # 5. Process big dataset
        all_chunks = []
        big_files = get_las_files(self.paths['for_instance_big_out'])
        
        for file_counter, file_path in enumerate(tqdm(big_files, desc="Processing big dataset")):
            file = load_las_file(file_path)
            
            # Additional filtering for big dataset
            file_chunks = self._process_file_to_chunks(file, file_counter, 
                                                     lambda f: chunkerize_clean(f, chunk_size=12.5), 
                                                     None, 'plot_FORinstance_big')
            
            # Apply big dataset specific filtering
            for chunk_file, filename in file_chunks:
                mixed = self.class_mappings['mixed']
                mask = ((chunk_file.semantic_gt == mixed['terrain']) & (chunk_file.instance_gt == 0)) | \
                       ((chunk_file.semantic_gt != mixed['terrain']) & (chunk_file.instance_gt != 0))
                if np.any(mask):
                    chunk_file.points = chunk_file.points[mask]
                    all_chunks.append((chunk_file, filename))
        
        # 6. Split and save big dataset chunks
        train_chunks, val_chunks, test_chunks = self._split_files_or_chunks(all_chunks, "big chunks")
        
        for chunks, split in [(train_chunks, 'train'), (val_chunks, 'val'), (test_chunks, 'test')]:
            for chunk_file, filename in tqdm(chunks, desc=f"Saving {split} chunks"):
                chunk_file.write(self.paths['mixed'] / split / filename)

        # 7. Print stats
        print(f"\nDataset processing completed. Output: {self.paths['mixed']}")
        for split in ['train', 'val', 'test']:
            files = list((self.paths['mixed'] / split).glob('*.las'))
            print(f"{split.capitalize()}: {len(files)} files")


if __name__ == "__main__":
    processor = DatasetProcessor()
    processor()
