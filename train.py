from pathlib import Path
from EHydro_TreeUnet.trainers import TreeProjectorTrainer


DATASET_FOLDER = Path.home() / 'datasets/MixedDataset'
OUTPUT_DIR = Path.home() / 'tree_projector_output'
VERSION_NAME = 'tree_projector_instance_VS-0.2_DA-48_E-3_v2'
START_ON_EPOCH = 0
FEAT_KEYS = ['intensity']
CHANNELS = [8, 16, 32, 64, 128]
LATENT_DIM = 256
INSTANCE_DENSITY = 0.01
CENTROID_THRES = 0.1
DESCRIPTOR_DIM = 32
TRAIN_PCT = 0.9
VOXEL_SIZE = 0.2
DATA_AUGMENTATION_COEF = 48.0
EPOCHS = 3
SEMANTIC_LOSS_COEF = 1.0
CENTROID_LOSS_COEF = 1.0
INSTANCE_LOSS_COEF = 1.0
BATCH_SIZE = 1

def main():
    tester = TreeProjectorTrainer(
        dataset_folder=DATASET_FOLDER,
        voxel_size=VOXEL_SIZE,
        train_pct=TRAIN_PCT,
        data_augmentation_coef=DATA_AUGMENTATION_COEF,
        epochs=EPOCHS,
        feat_keys=FEAT_KEYS,
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        instance_density=INSTANCE_DENSITY,
        centroid_thres=CENTROID_THRES,
        descriptor_dim=DESCRIPTOR_DIM,
        batch_size=BATCH_SIZE,
        training=True,
        semantic_loss_coef=SEMANTIC_LOSS_COEF,
        centroid_loss_coef=CENTROID_LOSS_COEF,
        instance_loss_coef=INSTANCE_LOSS_COEF,
        output_dir=OUTPUT_DIR,
        version_name=VERSION_NAME,
        start_on_epoch=START_ON_EPOCH
    )

    tester.train()


if __name__ == '__main__':
    main()