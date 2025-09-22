import argparse
from panoramix3D.trainers import Panoramix3DTrainer
from panoramix3D.config import AppConfig


def main(config_path: str):
    config = AppConfig.from_yaml(config_path)
    trainer = Panoramix3DTrainer(config)
    trainer.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Panoramix3D model')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file')
    
    args = parser.parse_args()
    main(args.config)
