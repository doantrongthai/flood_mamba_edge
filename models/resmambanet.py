import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class InvertedMambaBottleneck(nn.Module):
    """MobileNet-style Inverted Bottleneck integrated with Mamba SSM."""
    def __init__(self, in_c: int, out_c: int, expand_ratio: int = 2):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.norm = nn.LayerNorm(mid_c)
        self.mamba = Mamba(d_model=mid_c, d_state=16, expand=1)

        self.proj_conv = nn.Sequential(
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        feat = self.dw_conv(self.expand_conv(x))
        B, C, H, W = feat.shape
        hwc = feat.permute(0, 2, 3, 1).contiguous()
        norm_feat = self.norm(hwc).reshape(B, H * W, C)
        mamba_out = self.mamba(norm_feat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = self.proj_conv(mamba_out)
        return out + res


class ResMambaNet(nn.Module):
    """
    ResMambaNet: Inverted Bottleneck Residual Mamba Network for Efficient Flood Area Segmentation.
    """
    def __init__(self, num_classes: int = 1, in_channels: int = 3, base_c: int = 32, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )  # /2

        self.stage1 = InvertedMambaBottleneck(c, c, expand_ratio=2)  # /2
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False)  # /4

        self.stage2 = InvertedMambaBottleneck(c * 2, c * 2, expand_ratio=2)  # /4
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False)  # /8

        self.stage3 = InvertedMambaBottleneck(c * 4, c * 4, expand_ratio=2)  # /8

        # Decoder
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
        x0 = self.stem(x)       # /2, c
        e1 = self.stage1(x0)     # /2, c
        x1 = self.down1(e1)      # /4, 2c
        e2 = self.stage2(x1)     # /4, 2c
        x2 = self.down2(e2)      # /8, 4c
        e3 = self.stage3(x2)     # /8, 4c

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> ResMambaNet:
    return ResMambaNet(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)

