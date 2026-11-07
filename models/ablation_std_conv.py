"""
Ablation Study 2: BGFMamba with Standard 3x3 Convolutions (StdConv).
Tests the hypothesis: OmniASDConv (dynamic 1x7, 7x1 strip + 3x3 dilated) is superior
to standard isotropic 3x3 convolutions for capturing elongated, irregular flood flows.
In this variant, all OmniASDConv modules are replaced by standard 3x3 Conv-BN-SiLU layers.
"""

import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bgfmamba import (
    PurePyTorchMamba,
    Mamba2D,
    _scaled_bottleneck,
    DifferentiableLaplacianGuidance,
    GroupedMambaFusion,
    MultiDilatedSB_PixelShuffle,
    _make_light_fuse
)


class StdConvBlock(nn.Module):
    """Standard 3x3 Conv-BN-SiLU replacement for OmniASDConv."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class StdASDBR_Block(nn.Module):
    """Bottleneck residual block using standard 3x3 Conv."""
    def __init__(self, in_c: int, out_c: int, bottleneck: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(inplace=True),
        )
        self.conv = StdConvBlock(bottleneck, bottleneck)
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.reduce(x)
        x = self.conv(x)
        x = self.expand(x)
        return self.act(x + res)


class BGFMamba_StdConv(nn.Module):
    """BGFMamba using standard 3x3 convolutions instead of OmniASDConv."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 20, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(StdConvBlock(in_channels, c), StdConvBlock(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = StdASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = StdASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = StdASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = StdASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

        self.mamba_fusion = GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = MultiDilatedSB_PixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = MultiDilatedSB_PixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = MultiDilatedSB_PixelShuffle(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        x0 = self.inc(x)
        x0_guided = self.lap_guidance(x0)

        x1 = self.stage1(self.pool(x0_guided))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3)
        d3 = self.up3(self.dec_conv3(d4), f2)

        if self.training:
            aux_out = self.aux_head(d3)
            aux_out = F.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            aux_out = None

        d2 = self.up2(self.dec_conv2(d3), x1)

        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))

        if self.training and aux_out is not None:
            return out, aux_out
        return out


def build_model(num_classes: int = 1, input_size: int = 128, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return BGFMamba_StdConv(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
