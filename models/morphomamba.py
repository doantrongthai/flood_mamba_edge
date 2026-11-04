import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class MorphologicalGate(nn.Module):
    """
    Differentiable Morphological Gradient Gate.
    Approximates morphological dilation - erosion to emphasize flood boundaries.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilation = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        erosion = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        morph_grad = dilation - erosion
        gate = self.proj(morph_grad)
        return x * (1.0 + gate)


class MambaBottleneck(nn.Module):
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, expand=2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_hwc = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_hwc).reshape(B, H * W, C)
        out = self.mamba(x_norm)
        out = self.proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class MorphoMamba(nn.Module):
    """
    MorphoMamba: Morphology-Gated State Space U-Net for Flood Segmentation.
    Uses continuous mathematical morphology to inject flood boundary priors into
    skip-connections, guiding Mamba global context aggregation.
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
        )  # /2
        self.morph_gate1 = MorphologicalGate(c)

        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
        )  # /4
        self.morph_gate2 = MorphologicalGate(c * 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
        )  # /8

        self.bottleneck = MambaBottleneck(dim=c * 4)

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
        e1_g = self.morph_gate1(e1)

        e2 = self.enc2(e1)
        e2_g = self.morph_gate2(e2)

        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2_g], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1_g], dim=1))

        out = self.head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> MorphoMamba:
    return MorphoMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)
