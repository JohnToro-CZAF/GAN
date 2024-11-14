# main.py

import argparse
import json
import torch
import os

from GAN.models import make_model
from GAN.data import make_loader
from GAN.augmentations import make_augmentation
from GAN.evaluations import make_evaluation
from GAN.trainer import Trainer

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Make augmentation
    transform = make_augmentation(config)
    # Make data loader
    dataloader = make_loader(config, transform)
    # Make models
    models = make_model(config)
    # Optionally, make evaluation metric
    if 'evaluation' in config and config['evaluation']['metric']:
        evaluation_metric = make_evaluation(config)
    else:
        evaluation_metric = None
    # Initialize trainer
    trainer = Trainer(models, dataloader, config, device, evaluation_metric)
    # Start training
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Artistic Style Transfer with GANs')
    parser.add_argument('--config', type=str, required=True, help='Path to the config .json file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)