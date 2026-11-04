"""
LightM-UNet: Mamba Assists in Lightweight UNet for Segmentation
Adapted for real-time flood area segmentation on edge devices.

Original Repository: https://github.com/MrBlankness/LightM-UNet
Commit Hash: 38148ef8ac408014be0d668d9a98e2e97a68ed99
Reference: Liao et al., "LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation", 2024.
arXiv: https://arxiv.org/abs/2403.05246

Key Architectural Components:
- Depthwise Separable Convolutions (DWConv)
- Residual Mamba Blocks (ResMambaBlock) with bi-directional sequence scanning
- Residual Up-sampling Blocks (ResUpBlock)
- Linear computational complexity O(N)
- Native pure PyTorch Mamba fallback included for portability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_block import Mamba


def get_dwconv_layer(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
    )


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class ResMambaBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 8, kernel_size: int = 3):
        super().__init__()
        groups = min(num_groups, in_channels)
        while in_channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = MambaLayer(input_dim=in_channels, output_dim=in_channels)
        self.conv2 = MambaLayer(input_dim=in_channels, output_dim=in_channels)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x + identity
        return x


class ResUpBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 8, kernel_size: int = 3):
        super().__init__()
        groups = min(num_groups, in_channels)
        while in_channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = get_dwconv_layer(in_channels, in_channels, kernel_size=kernel_size)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x


class LightMUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        init_filters: int = 16,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        input_size: int = 128
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_filters = init_filters
        self.input_size = input_size

        self.convInit = get_dwconv_layer(in_channels, init_filters)
        self.down_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        filters = init_filters
        for i, n_blocks in enumerate(blocks_down):
            layer = nn.Sequential(*[ResMambaBlock(filters) for _ in range(n_blocks)])
            self.down_layers.append(layer)
            if i < len(blocks_down) - 1:
                downsample = nn.Sequential(
                    get_dwconv_layer(filters, filters * 2, stride=2),
                    nn.GroupNorm(min(8, filters * 2), filters * 2),
                    nn.ReLU(inplace=True)
                )
                self.downsamples.append(downsample)
                filters *= 2

        self.up_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, n_blocks in enumerate(reversed(blocks_up)):
            out_f = filters // 2
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                get_dwconv_layer(filters, out_f)
            )
            self.upsamples.append(upsample)
            layer = nn.Sequential(*[ResUpBlock(out_f) for _ in range(n_blocks)])
            self.up_layers.append(layer)
            filters = out_f

        self.final_conv = nn.Conv2d(filters, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.convInit(x)
        skips = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        skips = list(reversed(skips[:-1]))
        for i, (upsample, layer) in enumerate(zip(self.upsamples, self.up_layers)):
            x = upsample(x)
            if x.shape[2:] != skips[i].shape[2:]:
                x = F.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=True)
            x = x + skips[i]
            x = layer(x)

        out = self.final_conv(x)
        return out


def build_model(num_classes=1, input_size=128, pretrained=False, **kwargs):
    return LightMUNet(
        in_channels=3,
        num_classes=num_classes,
        init_filters=16,
        blocks_down=(1, 2, 2, 2),
        blocks_up=(1, 1, 1),
        input_size=input_size
    )
