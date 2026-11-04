import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.mamba_block import Mamba


class StripConvBlock(nn.Module):
    """Directional asymmetric strip convolution for stream/flow patterns."""
    def __init__(self, channels: int, k: int = 7):
        super().__init__()
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, k), padding=(0, k // 2), groups=channels, bias=False)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=(k, 1), padding=(k // 2, 0), groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv_h(x) + self.conv_v(x)
        out = self.act(self.bn(self.pointwise(out)))
        return out + res


class FlowMambaBlock(nn.Module):
    """Directional strip-guided Mamba block."""
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.strip = StripConvBlock(dim, k=7)
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, expand=2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x_strip = self.strip(x)
        B, C, H, W = x_strip.shape
        x_hwc = x_strip.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_hwc)
        x_flat = x_norm.reshape(B, H * W, C)
        mamba_out = self.mamba(x_flat)
        out = self.proj(mamba_out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class ConvStage(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FlowMamba(nn.Module):
    """
    FlowMamba: Flow-Aware Directional State Space Network for Flood Segmentation.
    Extracts directional stream features with strip convolutions and aggregates
    long-range topological continuity using Mamba SSM.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3, base_c: int = 32, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )  # (B, c, H/2, W/2)

        self.enc1 = ConvStage(c, c, stride=1)      # /2
        self.enc2 = ConvStage(c, c * 2, stride=2)  # /4
        self.enc3 = ConvStage(c * 2, c * 4, stride=2)  # /8

        self.flow_bottleneck = FlowMambaBlock(dim=c * 4)

        # Decoder with strip skip-connections
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c * 4, c * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
        )
        self.dec2 = ConvStage(c * 4, c * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.dec1 = ConvStage(c * 2, c)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)       # /2
        e1 = self.enc1(x0)      # /2
        e2 = self.enc2(e1)      # /4
        e3 = self.enc3(e2)      # /8

        b = self.flow_bottleneck(e3)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final_up(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> FlowMamba:
    return FlowMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)
