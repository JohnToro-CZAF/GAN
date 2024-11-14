# GAN/models/pix2pix.py

import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """Generator for Pix2Pix using U-Net architecture."""
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        self.down1 = self._block(in_channels, features, name="down1", normalize=False)
        self.down2 = self._block(features, features * 2, name="down2")
        self.down3 = self._block(features * 2, features * 4, name="down3")
        self.down4 = self._block(features * 4, features * 8, name="down4")
        self.down5 = self._block(features * 8, features * 8, name="down5")
        self.down6 = self._block(features * 8, features * 8, name="down6")
        self.down7 = self._block(features * 8, features * 8, name="down7")
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1)
        )
        self.up1 = self._up_block(features * 8, features * 8, name="up1", dropout=0.5)
        self.up2 = self._up_block(features * 16, features * 8, name="up2", dropout=0.5)
        self.up3 = self._up_block(features * 16, features * 8, name="up3", dropout=0.5)
        self.up4 = self._up_block(features * 16, features * 8, name="up4")
        self.up5 = self._up_block(features * 16, features * 4, name="up5")
        self.up6 = self._up_block(features * 8, features * 2, name="up6")
        self.up7 = self._up_block(features * 4, features, name="up7")
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, name, normalize=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _up_block(self, in_channels, out_channels, name, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)
        # Upsampling
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final(torch.cat([up7, d1], 1))


class PatchGANDiscriminator(nn.Module):
    """Discriminator for Pix2Pix using PatchGAN."""
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(in_channels * 2, 64, normalize=False),
            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def _block(self, in_channels, out_channels, stride=2, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        # Concatenate image and condition image by channels to produce input
        input = torch.cat([x, y], dim=1)
        return self.model(input)
