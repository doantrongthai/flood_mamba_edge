import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mamba_block import Mamba


class TopographicGradientModule(nn.Module):
    """Computes differentiable 2D spatial gradients via Sobel operator."""
    def __init__(self, channels: int):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.repeat(channels, 1, 1, 1).contiguous().clone()
        sobel_y = sobel_y.repeat(channels, 1, 1, 1).contiguous().clone()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.channels = channels
        self.conv_mod = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        mod_gate = self.conv_mod(grad_mag)
        return x * (1.0 + mod_gate)


class TopoMamba(nn.Module):
    """
    TopoMamba: Topographic Gradient-Modulated State Space Network for Flood Segmentation.
    Infuses spatial surface gradients into SSM sequences to enhance water boundary detection.
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

        self.topo_mod = TopographicGradientModule(channels=c * 4)
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

        e3_topo = self.topo_mod(e3)
        B, C, H, W = e3_topo.shape
        e3_hwc = e3_topo.permute(0, 2, 3, 1).contiguous()
        norm_e3 = self.norm(e3_hwc).reshape(B, H * W, C)
        mamba_out = self.mamba(norm_e3)
        mamba_out = self.proj(mamba_out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        b = e3 + mamba_out

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        return out


def build_model(num_classes: int = 1, input_size: int = 128, in_channels: int = 3, **kwargs) -> TopoMamba:
    return TopoMamba(num_classes=num_classes, in_channels=in_channels, input_size=input_size, **kwargs)

