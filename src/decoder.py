import torch
import torch.nn as nn

from torch.nn import functional as F
from .attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residual = x
        n, c, h, w = x.shape

        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return x + residual


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 128, kernel_size=3, padding=1),

            VAE_ResidualBlock(128, 128),

            VAE_AttentionBlock(128),

            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),

            VAE_ResidualBlock(128, 64),
            VAE_ResidualBlock(64, 64),
            VAE_ResidualBlock(64, 64),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            VAE_ResidualBlock(64, 32),
            VAE_ResidualBlock(32, 32),
            VAE_ResidualBlock(32, 32),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),

            nn.GroupNorm(32, 32),

            nn.SiLU(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        for module in self:
            x = module(x)
        
        return x
