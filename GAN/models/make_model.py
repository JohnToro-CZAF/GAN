# GAN/models/make_model.py

from .cyclegan import CycleGANGenerator, CycleGANDiscriminator
from .pix2pix import UNetGenerator, PatchGANDiscriminator
from .stargan import StarGANGenerator, StarGANDiscriminator
from .munit import MUNITGenerator, MUNITDiscriminator
from .stylegan import StyleGANGenerator, StyleGANDiscriminator

def make_model(config):
    model_name = config['model']['name'].lower()
    if model_name == 'cyclegan':
        G_AB = CycleGANGenerator(**config['model']['generator_params'])
        G_BA = CycleGANGenerator(**config['model']['generator_params'])
        D_A = CycleGANDiscriminator(**config['model']['discriminator_params'])
        D_B = CycleGANDiscriminator(**config['model']['discriminator_params'])
    elif model_name == 'pix2pix':
        G_AB = UNetGenerator(**config['model']['generator_params'])
        D_B = PatchGANDiscriminator(**config['model']['discriminator_params'])
        G_BA = None  # Pix2Pix is one-directional
        D_A = None
    elif model_name == 'stargan':
        G = StarGANGenerator(**config['model']['generator_params'])
        D = StarGANDiscriminator(**config['model']['discriminator_params'])
        G_AB = G_BA = G
        D_A = D_B = D
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return {'G_AB': G_AB, 'G_BA': G_BA, 'D_A': D_A, 'D_B': D_B}
