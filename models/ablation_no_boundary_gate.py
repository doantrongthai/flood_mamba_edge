"""
Ablation Study 6: BGFMamba without Spatial Boundary Gate (No-SB Gate).
Tests the hypothesis: The Multi-Dilated Spatial Boundary Gate is responsible for suppressing
irrelevant background noise and sharpening the water-land boundary on skip connections.
In this variant, skip connections concatenate directly with upsampled features without the boundary attention gate.
"""

import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bgfmamba import (
    _scaled_bottleneck,
    OmniASDConv,
    OmniASDBR_Block,
    DifferentiableLaplacianGuidance,
    GroupedMambaFusion,
    _make_light_fuse
)


class PixelShuffleDirectSkip(nn.Module):
    """Decoder block using PixelShuffle but WITHOUT the spatial boundary gate."""
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x_up, skip], dim=1))


class BGFMamba_NoBoundaryGate(nn.Module):
    """BGFMamba without Spatial Boundary Gate on skip connections."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 20, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = OmniASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

        self.mamba_fusion = GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = PixelShuffleDirectSkip(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = PixelShuffleDirectSkip(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = PixelShuffleDirectSkip(in_c=c, skip_c=c * 2, out_c=c)

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
    return BGFMamba_NoBoundaryGate(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
