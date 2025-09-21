from typing import Any, List

import numpy as np
import torch

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate


def sparse_unique_id_collate(inputs: List[SparseTensor]) -> SparseTensor:
    """
    Collate a batch of SparseTensors while ensuring unique instance IDs across the batch.
    
    This function is specifically designed for instance label tensors where each tensor
    contains instance IDs that need to be made globally unique across the batch. It
    combines multiple SparseTensors by adding batch indices to coordinates and applying
    ID offsets to features to prevent instance ID collisions between different samples.

    Args:
        inputs: List of SparseTensors to collate. Each tensor should have:
            - .coords: Coordinate tensor of shape [N_points, spatial_dims]
            - .feats: Feature/ID tensor of shape [N_points, feat_dims]
            - .stride: Spatial stride (must be consistent across inputs)

    Returns:
        SparseTensor with batched coordinates and globally unique instance IDs:
            - .coords: Shape [total_points, spatial_dims + 1] with batch indices
            - .feats: Shape [total_points, feat_dims] with offset instance IDs
            - .stride: Preserved from input tensors

    Example:
        >>> # Two tensors with instance IDs [1, 2, 3] and [1, 2]
        >>> tensor1 = SparseTensor(coords=coords1, feats=torch.tensor([1, 2, 3]))
        >>> tensor2 = SparseTensor(coords=coords2, feats=torch.tensor([1, 2]))
        >>> batched = sparse_unique_id_collate([tensor1, tensor2])
        >>> print(batched.feats)  # Output: [1, 2, 3, 4, 5] (IDs made unique)
        tensor([1, 2, 3, 4, 5])

    Notes:
        - Instance ID 0 is typically reserved for background and remains unchanged
        - Positive instance IDs are offset to ensure global uniqueness
        - Input tensors are converted from numpy arrays to torch tensors if needed
        - All input tensors must have the same stride value
    """
    coords, feats = [], []
    stride = inputs[0].stride

    offset = 0
    for k, x in enumerate(inputs):
        if isinstance(x.coords, np.ndarray):
            x.coords = torch.tensor(x.coords)
        if isinstance(x.feats, np.ndarray):
            x.feats = torch.tensor(x.feats)

        assert isinstance(x.coords, torch.Tensor), type(x.coords)
        assert isinstance(x.feats, torch.Tensor), type(x.feats)
        assert x.stride == stride, (x.stride, stride)

        input_size = x.coords.shape[0]
        batch = torch.full((input_size, 1), k, device=x.coords.device, dtype=torch.int)
        coords.append(torch.cat((batch, x.coords), dim=1))

        N = torch.max(x.feats).item()
        x.feats[x.feats > 0] += offset
        offset += N

        feats.append(x.feats)

    coords = torch.cat(coords, dim=0)
    feats = torch.cat(feats, dim=0)
    output = SparseTensor(coords=coords, feats=feats, stride=stride)
    return output


def sparse_unique_id_collate_fn(inputs: List[Any]) -> Any:
    """
    Recursive collate function for batching complex data structures with SparseTensors.
    
    This function handles batching of nested dictionaries containing mixed data types,
    with special treatment for instance label SparseTensors that require unique ID
    preservation. It recursively processes nested structures and applies appropriate
    collation strategies based on data type.

    Args:
        inputs: List of data samples to collate. Can be:
            - List of dictionaries with mixed value types
            - List of other data types (returned as-is)
            
        Each dictionary can contain:
            - Nested dictionaries (recursively processed)
            - NumPy arrays (converted to stacked tensors)
            - PyTorch tensors (stacked along batch dimension)
            - SparseTensors (collated with appropriate strategy)
            - Other types (collected into lists)

    Returns:
        Collated data structure matching input format:
            - Dictionary inputs return dictionaries with collated values
            - Non-dictionary inputs are returned unchanged
            - SparseTensors named 'instance_labels' use unique ID collation
            - Other SparseTensors use standard sparse collation
            - Arrays/tensors are stacked along the batch dimension

    Example:
        >>> batch = [
        ...     {'coords': sparse_tensor1, 'instance_labels': instance_tensor1},
        ...     {'coords': sparse_tensor2, 'instance_labels': instance_tensor2}
        ... ]
        >>> collated = sparse_unique_id_collate_fn(batch)
        >>> # instance_labels will have unique IDs, coords will be standard collated
        >>> print(type(collated['coords']))  # SparseTensor
        >>> print(type(collated['instance_labels']))  # SparseTensor with unique IDs

    Notes:
        - Only 'instance_labels' SparseTensors receive special unique ID treatment
        - Nested dictionary structures are preserved and recursively processed
        - NumPy arrays are automatically converted to PyTorch tensors
        - Non-tensor data types are collected into lists without modification
        - The function preserves the original data structure hierarchy
    """
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
            if isinstance(inputs[0][name], dict):
                output[name] = sparse_unique_id_collate_fn([input[name] for input in inputs])
            elif isinstance(inputs[0][name], np.ndarray):
                output[name] = torch.stack(
                    [torch.tensor(input[name]) for input in inputs], dim=0
                )
            elif isinstance(inputs[0][name], torch.Tensor):
                output[name] = torch.stack([input[name] for input in inputs], dim=0)
            elif isinstance(inputs[0][name], SparseTensor):
                if name == 'instance_labels':
                    output[name] = sparse_unique_id_collate([input[name] for input in inputs])
                else:
                    output[name] = sparse_collate([input[name] for input in inputs])
            else:
                output[name] = [input[name] for input in inputs]
        return output
    else:
        return inputs