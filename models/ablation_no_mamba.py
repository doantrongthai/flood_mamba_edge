"""
Ablation Study 3: BGFMamba without Mamba (Pure CNN Bottleneck Fusion).
Tests the hypothesis: Mamba State Space Model provides essential global context modeling
and long-range dependency capture that pure convolutions cannot achieve.
In this variant, GroupedMambaFusion is replaced by a standard Convolutional Bottleneck Fusion block.
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
    MultiDilatedSB_PixelShuffle,
    _make_light_fuse
)


class ConvBottleneckFusion(nn.Module):
    """Pure CNN replacement for GroupedMambaFusion using 3x3 Dilated Convolutions."""
    def __init__(self, in_channels_list: List[int], embed_dim: int = 40):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            for c in in_channels_list
        ])
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        orig_shapes = [f.shape[2:] for f in features]
        target_shape = features[0].shape[2:]

        aligned = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode='bilinear', align_corners=False)
            aligned.append(p)

        fused = sum(aligned)
        conv_feat = self.conv_fuse(fused) + fused

        outs = []
        for i, target_sz in enumerate(orig_shapes):
            out_f = self.out_projs[i](conv_feat)
            if out_f.shape[2:] != target_sz:
                out_f = F.interpolate(out_f, size=target_sz, mode='bilinear', align_corners=False)
            outs.append(out_f + features[i])
        return outs


class BGFMamba_NoMamba(nn.Module):
    """Pure CNN version of BGFMamba (GroupedMambaFusion replaced by Conv fusion)."""
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

        self.mamba_fusion = ConvBottleneckFusion([c * 4, c * 8, c * 8], embed_dim=40)

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
    return BGFMamba_NoMamba(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
