import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class RippleMambaBlock(nn.Module):
    """
    RippleMambaBlock: Dual-Scale Concentric Scanning.
    Simultaneously tracks local ripples (dense scan) and broad radial inundation (strided scan).
    """
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.norm_dense = nn.LayerNorm(dim)
        self.mamba_dense = Mamba(d_model=dim, d_state=d_state, expand=2)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.norm_ripple = nn.LayerNorm(dim)
        self.mamba_ripple = Mamba(d_model=dim, d_state=d_state, expand=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Dense local scan
        x_hwc = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm_dense(x_hwc).reshape(B, H * W, C)
        out_dense = self.mamba_dense(x_norm).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Concentric coarse ripple scan
        x_coarse = self.pool(x)  # H/2, W/2
        H_c, W_c = x_coarse.shape[2:]
        x_coarse_hwc = x_coarse.permute(0, 2, 3, 1).contiguous()
        x_coarse_norm = self.norm_ripple(x_coarse_hwc).reshape(B, H_c * W_c, C)
        out_coarse = self.mamba_ripple(x_coarse_norm).reshape(B, H_c, W_c, C).permute(0, 3, 1, 2)
        out_coarse_up = F.interpolate(out_coarse, size=(H, W), mode='bilinear', align_corners=False)

        fused = self.fusion(torch.cat([out_dense, out_coarse_up], dim=1))
        return x + fused


class RippleMamba(nn.Module):
    """
    RippleMamba: Concentric Multi-Scale Scanning Network for Flood Propagation.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3, base_c: int = 32, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )  # /2

        self.enc1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
        )  # /4
        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
        )  # /8

        self.ripple_bottleneck = RippleMambaBlock(dim=c * 4)

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

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        r = self.ripple_bottleneck(e3)

        d2 = self.up2(r)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> RippleMamba:
    return RippleMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)

