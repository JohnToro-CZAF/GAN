# GAN/models/cyclegan.py

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """Generator network for CycleGAN."""
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        model = [
            nn.ReflectionPad2d(7),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual Blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output Layer
        model += [
            nn.ReflectionPad2d(7),
            nn.Conv2d(curr_dim, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """Discriminator network for CycleGAN."""
    def __init__(self, in_channels=3):
        super(CycleGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)  # Output single channel prediction map
        )

    def forward(self, x):
        return self.model(x)
