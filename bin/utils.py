import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_las_files(folder):
    """Get sorted list of LAS/LAZ files in folder."""
    return sorted(
        [f for f in Path(folder).rglob("*") if f.is_file() and f.suffix.lower() in ('.laz', '.las')],
        key=lambda f: f.name
    )


def load_las_file(file_path):
    """Load a single LAS file with basic preprocessing."""
    file = laspy.read(file_path)
    
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
    
    return file


def load_point_clouds(folder):
    """Load all point clouds from folder with preprocessing. 
    Use sparingly - prefer loading individual files with load_las_file()."""
    files = get_las_files(folder)
    
    for file_path in tqdm(files, desc=f'Loading {Path(folder).name}'):
        yield load_las_file(file_path)


def chunkerize_clean(file, chunk_size):
    xy = np.stack([file.x, file.y], axis=1)
    labels = file.instance_gt
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    centers = []
    for label in unique_labels:
        centers.append(xy[labels == label].mean(axis=0))

    if not centers:
        return [np.ones_like(file.instance_gt, dtype=bool)]
    
    if len(centers) == 1:
        return [(xy[:, 0] >= centers[0][0] - (chunk_size / 2)) & (xy[:, 0] <= centers[0][0] + (chunk_size / 2)) & \
            (xy[:, 1] >= centers[0][1] - (chunk_size / 2)) & (xy[:, 1] <= centers[0][1] + (chunk_size / 2))]

    centers = np.array(centers)
    while True:
        distances = np.abs(centers[None, :] - centers[:, None])
        distances = np.min(distances, axis=-1)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        idxs = np.argwhere(mask)
        vals = distances[mask]
        flat_idx = vals.argmin()
        val = vals[flat_idx]
        if val > (chunk_size / 2) * 0.6:
            break

        i, j = idxs[flat_idx]
        row_proximity = distances[i, :].sum()
        col_proximity = distances[:, j].sum()
        centers = np.delete(centers, i if row_proximity < col_proximity else j, axis=0)

    masks = []
    for center in centers:
        masks.append((xy[:, 0] >= center[0] - (chunk_size / 2)) & (xy[:, 0] <= center[0] + (chunk_size / 2)) & \
            (xy[:, 1] >= center[1] - (chunk_size / 2)) & (xy[:, 1] <= center[1] + (chunk_size / 2)))
        
    return masks


def chunkerize_four(file):
    xy = np.stack([file.x, file.y], axis=1)
    center = xy.mean(axis=0)

    return [
        (xy[:, 0] > center[0]) & (xy[:, 1] > center[1]),
        (xy[:, 0] < center[0]) & (xy[:, 1] > center[1]),
        (xy[:, 0] < center[0]) & (xy[:, 1] < center[1]),
        (xy[:, 0] > center[0]) & (xy[:, 1] < center[1])
    ]
