import numpy as np
import torch
import laspy

from copy import deepcopy
from laspy import LasData
from pathlib import Path
from tqdm import tqdm
from torch.cuda import amp
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor

from panoramix3D.models import Panoramix3D
from panoramix3D.config import ModelConfig


class CloudProcessor:
    def __init__(self):
        self._input_folder = Path('/workspace/input')
        self._output_folder = Path('/workspace/output')

        self._chunk_size = 30
        self._min_chunk_size = 15
        self._min_points_per_pc = 2000
        self._voxel_size = 0.3

        self._input_folder.mkdir(parents=True, exist_ok=True)
        self._output_folder.mkdir(parents=True, exist_ok=True)

        self._config = ModelConfig.from_yaml('/workspace/Panoramix3D/config/model/inference.yaml')

    def _chunkerize(self, file: LasData, ext: str):
        points = file.points
        coords = np.vstack((file.x, file.y)).transpose()

        idx  = np.floor_divide(coords, self._chunk_size).astype(int)
        idx -= idx.min(axis=0)

        idx = np.ravel_multi_index(idx.T, idx.max(axis=0) + 1)
        chunks = []
        for i, unique_idx in enumerate(tqdm(np.unique(idx), desc="Chunkerizing point cloud")):
            chunk_points = points[idx == unique_idx]
            if len(chunk_points) < self._min_points_per_pc:
                continue
            
            min_x, max_x = chunk_points.x.min(), chunk_points.x.max()
            min_y, max_y = chunk_points.y.min(), chunk_points.y.max()
            if (max_x - min_x) < self._min_chunk_size or (max_y - min_y) < self._min_chunk_size:
                continue

            chunk_file = laspy.LasData(deepcopy(file.header))
            chunk_file.points = chunk_points
            chunks.append(chunk_file)
        return chunks

    def __call__(self):
        extensions = ('.laz', '.las')
        point_clouds = sorted(
                    [f for f in self._input_folder.rglob("*") if f.is_file() and f.suffix.lower() in extensions],
                    key=lambda f: f.name
                )

        model = Panoramix3D.from_config(self._config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        for point_cloud in point_clouds:
            chunks = self._chunkerize(laspy.read(point_cloud), point_cloud.suffix.lower())

            instance_offset = 1
            header = deepcopy(chunks[0].header)
            header.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.int32))
            header.add_extra_dim(laspy.ExtraBytesParams(name="species_pred", type=np.int32))
            header.add_extra_dim(laspy.ExtraBytesParams(name="instance_pred", type=np.int32))
            with laspy.open(self._output_folder / f'{point_cloud.stem}_segmented.las', mode='w', header=header) as file:
                for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {point_cloud.name}")):
                    voxels = np.vstack((chunk.x, chunk.y, chunk.z)).transpose()
                    min_coords = voxels.min(axis=0)
                    voxels -= min_coords

                    intensity = np.array(chunk.intensity)[:, None]
                    min_intensity = np.min(intensity)
                    max_intensity = np.max(intensity)
                    i_norm = (intensity - min_intensity) / (max_intensity - min_intensity)

                    voxels, indices, inverse_map = sparse_quantize(voxels, self._voxel_size, return_index=True, return_inverse=True)
                    i_norm = i_norm[indices]

                    min_x, max_x = chunk.x.min(), chunk.x.max()
                    min_y, max_y = chunk.y.min(), chunk.y.max()

                    if len(voxels) < 100:
                        continue
                    if (max_x - min_x) < self._min_chunk_size or (max_y - min_y) < self._min_chunk_size:
                        continue

                    voxels = torch.tensor(voxels, dtype=torch.int).to(device)
                    batch_index = torch.zeros((voxels.shape[0], 1), dtype=torch.int, device=voxels.device)
                    voxels = torch.cat([batch_index, voxels], dim=1)
                    feat = torch.tensor(i_norm.astype(np.float32), dtype=torch.float).to(device)

                    inputs = SparseTensor(coords=voxels, feats=feat)

                    with amp.autocast(enabled=True):
                        result = model(inputs)
                    if result is None:
                        continue
                        
                    semantic_output_raw, specie_output_raw, _, _, _, instance_output_raw = result

                    semantic_output = torch.argmax(semantic_output_raw.F.cpu(), dim=1).numpy()
                    ng_mask = semantic_output != 0

                    specie_output = np.zeros_like(semantic_output)
                    specie_output[ng_mask] = torch.argmax(specie_output_raw.F.cpu(), dim=1).numpy() + 1

                    instance_output_full = np.zeros_like(semantic_output)

                    if instance_output_raw.F.shape[1] > 0:
                        instance_output = torch.argmax(instance_output_raw.F.cpu(), dim=1).numpy()
                        max_label = instance_output.max()
                        instance_output_full[ng_mask] = instance_output + instance_offset
                        instance_offset += max_label + 1

                    semantic_output = semantic_output[inverse_map]
                    specie_output = specie_output[inverse_map]
                    instance_output = instance_output_full[inverse_map]

                    out_file = laspy.LasData(header=chunk.header, points=chunk.points.copy())
                    out_file.add_extra_dims([laspy.ExtraBytesParams(name="semantic_pred", type=np.int32), laspy.ExtraBytesParams(name="species_pred", type=np.int32), laspy.ExtraBytesParams(name="instance_pred", type=np.int32)])
                    out_file.semantic_pred = semantic_output
                    out_file.species_pred = specie_output
                    out_file.instance_pred = instance_output

                    file.write_points(out_file.points)


if __name__ == "__main__":
    processor = CloudProcessor()
    processor()
