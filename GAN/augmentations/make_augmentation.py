# GAN/augmentations/make_augmentation.py

from .augmentations import get_augmentation_by_name

def make_augmentation(config):
    aug_name = config['augmentation']['name']
    img_size = config['data']['img_size']
    return get_augmentation_by_name(aug_name, img_size)
