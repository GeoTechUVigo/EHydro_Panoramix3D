from pathlib import Path
from EHydro_TreeUnet.trainers import TreeProjectorTrainer


DATASET_FOLDER = Path.home() / 'datasets/MixedDataset'
FEAT_KEYS = ['intensity']
CHANNELS = [16, 32, 64, 128]
LATENT_DIM = 256
MAX_INSTANCES = 64
TRAIN_PCT = 0.8
VOXEL_SIZE = 0.2
DATA_AUGMENTATION_COEF = 10.0
SEMANTIC_LOSS_COEF = 1.0
INSTANCE_LOSS_COEF = 1.0
BATCH_SIZE = 1

def main():
    tester = TreeProjectorTrainer(
        dataset_folder=DATASET_FOLDER,
        voxel_size=VOXEL_SIZE,
        train_pct=TRAIN_PCT,
        data_augmentation_coef=DATA_AUGMENTATION_COEF,
        feat_keys=FEAT_KEYS,
        max_instances=MAX_INSTANCES,
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        batch_size=BATCH_SIZE,
        training=True,
        semantic_loss_coef=SEMANTIC_LOSS_COEF,
        instance_loss_coef=INSTANCE_LOSS_COEF
    )

    tester.train()


if __name__ == '__main__':
    main()