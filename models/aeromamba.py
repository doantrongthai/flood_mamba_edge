import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class AeroMamba(nn.Module):
    """
    AeroMamba: Ultra-Lightweight Asymmetric State Space Network for UAV Flood Rescue.
    Designed for extreme edge latency (<8ms) on low-power onboard drone computers.
    Uses rapid stem downsampling, single-stage Mamba SSM bottleneck, and PixelShuffle decoder.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3, base_c: int = 16, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        # Rapid stem: /4 resolution reduction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True)
        )  # /4

        # Encoder stage /8
        self.enc = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True)
        )  # /8

        # Mamba Bottleneck
        self.norm = nn.LayerNorm(c * 4)
        self.mamba = Mamba(d_model=c * 4, d_state=16, expand=2)
        self.proj = nn.Linear(c * 4, c * 4)

        # Asymmetric PixelShuffle Decoder (No heavy convs)
        # /8 -> /4
        self.up1_conv = nn.Conv2d(c * 4, (c * 2) * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True)
        )

        # /4 -> /1
        self.up2_conv = nn.Conv2d(c * 2, c * 16, kernel_size=1)
        self.up2_ps = nn.PixelShuffle(upscale_factor=4)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)   # /4, c*2
        e2 = self.enc(e1)   # /8, c*4

        B, C, H, W = e2.shape
        e2_hwc = e2.permute(0, 2, 3, 1).contiguous()
        norm_e2 = self.norm(e2_hwc).reshape(B, H * W, C)
        mamba_out = self.mamba(norm_e2)
        mamba_out = self.proj(mamba_out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        b = e2 + mamba_out

        # PixelShuffle Upsampling
        d1 = self.up1_ps(self.up1_conv(b))  # /4, c*2
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))  # /4, c*2

        d0 = self.up2_ps(self.up2_conv(d1))  # /1, c
        out = self.head(d0)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> AeroMamba:
    return AeroMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)
