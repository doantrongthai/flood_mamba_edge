import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class MultiDilatedBlock(nn.Module):
    """Multi-scale dilated depthwise feature extractor for diverse water expanses."""
    def __init__(self, channels: int):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False)
        self.d2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.d4 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.d1(x)
        out2 = self.d2(x)
        out4 = self.d4(x)
        return self.fuse(torch.cat([out1, out2, out4], dim=1))


class DeltaMamba(nn.Module):
    """
    DeltaMamba: Multi-Receptive Field Dilated State Space Network for Flood Segmentation.
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

        self.multi_dilated = MultiDilatedBlock(channels=c * 4)
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

        e3_dilated = self.multi_dilated(e3)
        B, C, H, W = e3_dilated.shape
        hwc = e3_dilated.permute(0, 2, 3, 1).contiguous()
        norm_feat = self.norm(hwc).reshape(B, H * W, C)
        mamba_out = self.mamba(norm_feat)
        mamba_out = self.proj(mamba_out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        b = e3 + mamba_out

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> DeltaMamba:
    return DeltaMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)

