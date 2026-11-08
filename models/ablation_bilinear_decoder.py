"""
Ablation Study 5: BGFMamba with Bilinear Decoder Upsampling.
Tests the hypothesis: PixelShuffle sub-pixel spatial reconstruction prevents the boundary blurring
and checkerboard smoothing inherent to standard bilinear interpolation.
In this variant, all PixelShuffle upsampling blocks are replaced with Bilinear Upsampling + Conv1x1.
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


class MultiDilatedSB_Bilinear(nn.Module):
    """Decoder block using Bilinear upsampling instead of PixelShuffle."""
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c
        k_c = 8
        self.d1 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=1, bias=False)
        self.d2 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(k_c)
        self.gate = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(k_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.d1(skip) + self.d2(skip)
        g = self.gate(self.bn(feat))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class BGFMamba_BilinearDecoder(nn.Module):
    """BGFMamba with Bilinear upsampling decoder instead of PixelShuffle."""
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
        self.up4 = MultiDilatedSB_Bilinear(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = MultiDilatedSB_Bilinear(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = MultiDilatedSB_Bilinear(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c, kernel_size=1)
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

        d1 = F.interpolate(self.up1_conv(d2), size=x0_guided.shape[2:], mode='bilinear', align_corners=False)
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))

        if self.training and aux_out is not None:
            return out, aux_out
        return out


def build_model(num_classes: int = 1, input_size: int = 128, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return BGFMamba_BilinearDecoder(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
