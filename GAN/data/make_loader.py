# GAN/data/make_loader.py

from torch.utils.data import DataLoader
from .datasets import GANDataset

def make_loader(config, transform):
    dataset_type = config['data'].get('type', 'unpaired').lower()
    if dataset_type == 'unpaired':
        dataset = GANDataset(
            root_dir_A=config['data']['domain_A_path'],
            root_dir_B=config['data']['domain_B_path'],
            transform=transform
        )
    elif dataset_type == 'paired':
        dataset = PairedGANDataset(
            root_dir=config['data']['paired_data_path'],
            transform=transform
        )
    else:
        raise ValueError(f"Dataset type {dataset_type} not recognized.")

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    return loader
