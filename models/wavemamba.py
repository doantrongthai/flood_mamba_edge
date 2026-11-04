import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class HaarDWT2D(nn.Module):
    """Parameter-free Haar 2D Discrete Wavelet Transform."""
    def forward(self, x: torch.Tensor):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, torch.cat([lh, hl, hh], dim=1)


class WaveMambaBlock(nn.Module):
    """
    Wavelet-Integrated State Space Block.
    Processes low-frequency macro water extent with Mamba SSM
    and high-frequency surface wave/boundary details with depthwise conv.
    """
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.norm = nn.LayerNorm(dim)
        self.mamba_ll = Mamba(d_model=dim, d_state=d_state, expand=2)
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, high = self.dwt(x)  # H/2, W/2
        B, C, H, W = ll.shape
        ll_hwc = ll.permute(0, 2, 3, 1).contiguous()
        ll_norm = self.norm(ll_hwc).reshape(B, H * W, C)
        mamba_ll = self.mamba_ll(ll_norm).reshape(B, H, W, C).permute(0, 3, 1, 2)

        feat_high = self.high_conv(high)
        fused = self.fuse_proj(torch.cat([mamba_ll, feat_high], dim=1))
        out = self.upsample(fused)
        return x + out


class WaveMamba(nn.Module):
    """
    WaveMamba: Wavelet-Decomposed Spatial State Space Network for Flood Segmentation.
    Decomposes flood imagery into smooth water expanses (low-frequency) and sharp
    boundary features (high-frequency) for robust flood mapping.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3, base_c: int = 32, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )  # /2

        self.enc1 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True)
        )  # /4

        self.enc2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True)
        )  # /8

        self.wave_block = WaveMambaBlock(dim=c * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c * 4, c * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)   # /2
        e1 = self.enc1(e0)  # /4
        e2 = self.enc2(e1)  # /8

        w = self.wave_block(e2)

        d2 = self.up2(w)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        out = self.final_head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> WaveMamba:
    return WaveMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)
