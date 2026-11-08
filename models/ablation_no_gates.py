import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bgfmamba import (
    _scaled_bottleneck,
    GroupedMambaFusion,
    _make_light_fuse
)


class OmniASDConv_NoGate(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3):
        super().__init__()
        self.pw_in = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        self.dw_h = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, k // 2), groups=out_c, bias=False)
        self.dw_v = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(k // 2, 0), groups=out_c, bias=False)
        self.dw_d = nn.Conv2d(out_c, out_c, kernel_size=3, padding=2, dilation=2, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.pw_out = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        h = self.dw_h(x)
        v = self.dw_v(x)
        d = self.dw_d(x)
        fuse = (h + v + d) / 3.0
        return x + self.pw_out(self.act(self.bn(fuse)))


class OmniASDBR_Block_NoGate(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: Optional[int] = None, k: int = 3):
        super().__init__()
        mid = bottleneck if bottleneck is not None else _scaled_bottleneck(out_c)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        self.conv1 = OmniASDConv_NoGate(mid, mid, k=k)
        self.conv2 = OmniASDConv_NoGate(mid, mid, k=k)
        self.expand = nn.Sequential(
            nn.Conv2d(mid, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        y = self.reduce(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.expand(y)
        return self.act(res + y)


class PixelShuffleDirectSkip(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up_conv = nn.Conv2d(in_c, in_c * 4, kernel_size=1)
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(self.up_conv(x))
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x_up, skip], dim=1))


class BGFMamba_NoGates(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 20, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv_NoGate(in_channels, c), OmniASDConv_NoGate(c, c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = OmniASDBR_Block_NoGate(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block_NoGate(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block_NoGate(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block_NoGate(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

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
        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3)
        d3 = self.up3(self.dec_conv3(d4), f2)

        aux_out = F.interpolate(self.aux_head(d3), size=x.shape[2:], mode="bilinear", align_corners=False) if self.training else None

        d2 = self.up2(self.dec_conv2(d3), x1)
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0], dim=1))

        if self.training and aux_out is not None:
            return out, aux_out
        return out


def build_model(num_classes: int = 1, input_size: int = 128, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return BGFMamba_NoGates(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
