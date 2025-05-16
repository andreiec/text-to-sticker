import torch
import torch.nn as nn

from torch.nn import functional as F
from .decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            VAE_ResidualBlock(32, 32),
            VAE_ResidualBlock(32, 32),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(32, 64),
            VAE_ResidualBlock(64, 64),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(64, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            VAE_AttentionBlock(128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            nn.Conv2d(128, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar
