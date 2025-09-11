# AI Agent Instructions for EHydro_TreeUnet

This document provides essential guidance for AI agents working with the EHydro_TreeUnet codebase.

## Project Architecture

EHydro_TreeUnet is a deep learning project focused on processing point cloud data with tree-structured neural networks. Key components:

### Core Models (`models/`)
- `TreeProjector`: Main model combining ResNet backbone with specialized heads
  - Uses `SparseResNet` encoder and `VoxelDecoder` 
  - Multiple task heads: semantic, centroid, offset, instance
  - Works with sparse tensors for efficient point cloud processing
- `TreeUNet`: U-Net style architecture for semantic segmentation
  - Encoder-decoder structure with skip connections
  - Uses sparse convolutions for point cloud processing

### Training Components (`trainers/`)
- `TreeProjectorTrainer`: Orchestrates training workflow
  - Handles data loading, loss computation, and checkpointing
  - Manages multiple loss components (semantic, centroid, offset, instance)
  - Supports training resumption via checkpoints

### Dataset Handling (`datasets/`)
- `MixedDataset`: Main dataset class with train/val splits
- Supports data augmentation (rotation, tilt, scale)
- Uses voxelization with configurable voxel size

## Development Workflow

### Environment Setup
```env
TREE_PROJECTOR_DIR=<path>  # Set in .env file
```

### Training Workflow
1. Configure training parameters in `train.py` or notebook:
   - Dataset paths, voxel size, feature keys
   - Training hyperparameters (epochs, batch size, loss coefficients)
   - Model architecture (ResNet blocks, latent dimensions)
2. Training outputs stored in:
   - `stats/stats.pkl` and `stats/losses.pkl` for metrics
   - `<TREE_PROJECTOR_DIR>/weights/<version_name>/` for model weights

### Model Configuration
- ResNet blocks define feature extraction scales
- Multiple task heads with configurable parameters:
  - `instance_density`: Controls instance detection density
  - `score_thres`: Threshold for score predictions
  - `centroid_thres`: Threshold for centroid detection
  - `descriptor_dim`: Dimension of instance descriptors

## Key Files
- `train.py`: Main training script
- `test_train.ipynb`: Interactive training notebook
- Key configuration in environment variables or `.env` file

## Best Practices
1. Always configure TREE_PROJECTOR_DIR environment variable
2. Use provided data augmentation parameters for robust training
3. Monitor multiple loss components during training
4. Use checkpointing for long training runs

## Common Patterns
- Sparse tensor operations for efficient point cloud processing
- Multi-task learning with weighted loss components
- Hierarchical feature extraction with skip connections
