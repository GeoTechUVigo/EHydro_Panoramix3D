import os

from pathlib import Path
from dotenv import load_dotenv

from EHydro_TreeUnet.trainers import TreeProjectorTrainer


load_dotenv(Path(__file__).parent / '.env')

TREE_PROJECTOR_DIR = Path(os.environ.get('TREE_PROJECTOR_DIR', Path.home() / 'tree_projector'))
DATASET_FOLDER = 'MixedDataset'
VERSION_NAME = 'tree_projector_instance_VS-0.2_DA-48_E-3_v4'

VOXEL_SIZE = 0.2
FEAT_KEYS = ['intensity']
CENTROID_SIGMA = 1.5
TRAIN_PCT = 0.9
DATA_AUGMENTATION_COEF = 48
YAW_RANGE = (0.0, 360.0)
TILT_RANGE = (-5.0, 5.0)
SCALE_RANGE = (0.9, 1.2)

EPOCHS = 3
START_ON_EPOCH = 0
BATCH_SIZE = 1
SEMANTIC_LOSS_COEF = 1.0
CENTROID_LOSS_COEF = 1.0
INSTANCE_LOSS_COEF = 1.0

RESNET_BLOCKS = [
    (3, 16, 3, 1),
    (3, 32, 3, 2),
    (3, 64, 3, 2),
    (3, 128, 3, 2),
    (1, 128, (1, 1, 3), (1, 1, 2)),
]
LATENT_DIM = 512
INSTANCE_DENSITY = 0.01
CENTROID_THRES = 0.1
DESCRIPTOR_DIM = 128


def main():
    tester = TreeProjectorTrainer(
        tree_projector_dir=TREE_PROJECTOR_DIR,
        dataset_folder=DATASET_FOLDER,
        version_name=VERSION_NAME,

        voxel_size=VOXEL_SIZE,
        feat_keys=FEAT_KEYS,
        centroid_sigma=CENTROID_SIGMA,
        train_pct=TRAIN_PCT,
        data_augmentation_coef=DATA_AUGMENTATION_COEF,
        yaw_range=YAW_RANGE,
        tilt_range=TILT_RANGE,
        scale_range=SCALE_RANGE,

        training=True,
        epochs=EPOCHS,
        start_on_epoch=START_ON_EPOCH,
        batch_size=BATCH_SIZE,
        semantic_loss_coef=SEMANTIC_LOSS_COEF,
        centroid_loss_coef=CENTROID_LOSS_COEF,
        instance_loss_coef=INSTANCE_LOSS_COEF,

        resnet_blocks=RESNET_BLOCKS,
        latent_dim=LATENT_DIM,
        instance_density=INSTANCE_DENSITY,
        centroid_thres=CENTROID_THRES,
        descriptor_dim=DESCRIPTOR_DIM
    )

    tester.train()


if __name__ == '__main__':
    main()