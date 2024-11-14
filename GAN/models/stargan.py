# GAN/models/stargan.py

import torch
import torch.nn as nn

class StarGANGenerator(nn.Module):
    """Generator for StarGAN."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(StarGANGenerator, self).__init__()

        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        ]

        # Down-sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2

        # Bottleneck layers.
        for _ in range(repeat_num):
            layers += [ResidualBlock(curr_dim)]

        # Up-sampling layers.
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        layers += [
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class StarGANDiscriminator(nn.Module):
    """Discriminator for StarGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(StarGANDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        ]

        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01)
            ]
            curr_dim *= 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h).view(h.size(0), -1)
        return out_src, out_cls
