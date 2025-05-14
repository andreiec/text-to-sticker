import torch
import torch.nn as nn

from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear_1 = nn.Linear(d_embed, d_embed * 4)
        self.linear_2 = nn.Linear(d_embed * 4, 1280)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for module in self:
            if isinstance(module, UNET_AttentionBlock):
                x = module(x, context)
            elif isinstance(module, UNET_ResidualBlock):
                x = module(x, time)
            else:
                x = module(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time=1280):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_1 = nn.Linear(time, out_channels)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x, time):
        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        time = F.silu(time)
        time = self.linear_1(time)

        x = x + time.unsqueeze(-1).unsqueeze(-1)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads, d_embed, d_context=768):
        super().__init__()
        channels = n_heads * d_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)        

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, context):
        residual_long = x

        x = self.groupnorm(x)
        x = self.conv_1(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)

        residual_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residual_short

        residual_short = x

        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residual_short

        residual_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x = x + residual_short

        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return self.conv_2(x) + residual_long


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 640)),
            SwitchSequential(UNET_ResidualBlock(640, 640)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(640, 640),
            UNET_AttentionBlock(8, 80),
            UNET_ResidualBlock(640, 640),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(1280, 640)),
            SwitchSequential(UNET_ResidualBlock(1280, 640)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])
    
    def forward(self, x, context, time):
        skip_connections = []

        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)

        return x


class UNET_Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(1000, 320)
        self.time_mlp = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_Output(320, 4)
    
    def forward(self, latent, context, time):
        time = self.time_embed(time)
        time = self.time_mlp(time)
        x = self.unet(latent, context, time)
        x = self.final(x)

        return x