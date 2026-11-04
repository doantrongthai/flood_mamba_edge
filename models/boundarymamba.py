import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class BoundaryEnhancementModule(nn.Module):
    """Internal boundary enhancement module focusing on flood-land transitions."""
    def __init__(self, channels: int):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_map = self.edge_conv(x)
        refined = self.refine(torch.cat([x, edge_map], dim=1))
        return refined


class BoundaryMamba(nn.Module):
    """
    BoundaryMamba: Boundary-Enhanced State Space Network for Flood Area Segmentation.
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

        # Mamba Bottleneck
        self.norm = nn.LayerNorm(c * 4)
        self.mamba = Mamba(d_model=c * 4, d_state=16, expand=2)
        self.proj = nn.Linear(c * 4, c * 4)

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

        self.boundary_module = BoundaryEnhancementModule(channels=c)
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

        B, C, H, W = e3.shape
        e3_hwc = e3.permute(0, 2, 3, 1).contiguous()
        norm_e3 = self.norm(e3_hwc).reshape(B, H * W, C)
        mamba_out = self.mamba(norm_e3)
        mamba_out = self.proj(mamba_out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        b = e3 + mamba_out

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d1_enhanced = self.boundary_module(d1)
        out = self.head(d1_enhanced)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> BoundaryMamba:
    return BoundaryMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)
