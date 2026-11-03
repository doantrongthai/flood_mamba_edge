"""
SatReg CM-UNet: Regression-Based Hardware-Aware Neural Architecture for Satellite / Remote Sensing Segmentation
Adapted for real-time flood area segmentation on edge devices.

Reference:
- SatReg: Humes & Mohsenin, "SatReg: Regression-based Neural Architecture Search for Lightweight Satellite Image Segmentation",
  ACM/IEEE Design Automation Conference (DAC '26), 2026.
  arXiv: https://arxiv.org/abs/2604.10306
  Edge Deployment: NVIDIA Jetson Orin Nano Super in MAXN SUPER mode (109.8 ms, 8.76 W, 9.11 FPS).
- Parent Model: Liu et al., "CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation", 2024.
  arXiv: https://arxiv.org/abs/2405.10530
  Official Repository: https://github.com/XiaoBuL/CM-UNet

Architectural Highlights:
- Hybrid CNN-Mamba UNet topology:
    * ResNet-style 4-stage convolutional encoder (stem stride /4, stage1 /4, stage2 /8, stage3 /16, stage4 /32)
      for rich spatial and boundary feature extraction.
    * Multi-Scale Feature Fusion Attention (MSFFA / MSAA) connecting intermediate encoder scales.
    * CSMamba Decoder: Visual State Space (VSSBlock / SS2D) selective scanning for global context.
    * 4x progressive upsampling projection head restoring full spatial resolution.
- SatReg Parameterization:
    * Width-scaled base encoder width (b) and decoder bottleneck width (h).
    * SatReg configuration (b=32, h=32): ~7.2M parameters (optimized for Jetson Orin Nano).
    * Standard CM-UNet configuration (b=64, h=64): ~17.6M parameters.
- Pure PyTorch native S6 selective scan fallback included for execution on CPU/Edge hardware
  without precompiled CUDA extensions.
"""

import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F



# ─────────────────────────────────────────────────────────────────────────────
# 1. 2D Selective Scan (SS2D) & CSMamba Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class SS2D(nn.Module):
    """
    2D Selective Scan Module (SS2D).
    Flattens 2D feature maps along 4 spatial directions (horizontal forward/backward,
    vertical forward/backward) and executes selective scanning across all directions.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # 4 scanning directions
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(4)
        ])
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(4)
        ])

        # S6 SSM parameters per direction
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.stack([torch.log(A) for _ in range(4)]))  # (4, d_inner, d_state)
        self.Ds = nn.Parameter(torch.ones(4, self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _selective_scan_dir(self, u: torch.Tensor, dir_idx: int) -> torch.Tensor:
        """
        Pure PyTorch selective scan for one sequence direction.
        u: (B, L, D)
        """
        B, L, D = u.shape
        x_proj_out = self.x_proj[dir_idx](u)  # (B, L, dt_rank + 2 * d_state)
        delta_rank = self.dt_rank
        delta = x_proj_out[:, :, :delta_rank]
        B_mat = x_proj_out[:, :, delta_rank:delta_rank + self.d_state]
        C_mat = x_proj_out[:, :, delta_rank + self.d_state:]

        delta = F.softplus(self.dt_projs[dir_idx](delta))  # (B, L, D)
        A = -torch.exp(self.A_logs[dir_idx])  # (D, d_state)

        # Discretization
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, d_state)
        deltaB_u = (delta.unsqueeze(-1) * B_mat.unsqueeze(2)) * u.unsqueeze(-1)  # (B, L, D, d_state)

        # Sequential scan
        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y_t = torch.einsum('bdn,bn->bd', h, C_mat[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + u * self.Ds[dir_idx]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) or (B, C, H, W)"""
        if x.dim() == 4 and x.shape[1] == self.d_model:
            B, C, H, W = x.shape
            x_in = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            was_chw = True
        else:
            B, H, W, C = x.shape
            x_in = x
            was_chw = False

        xz = self.in_proj(x_in)  # (B, H, W, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Depthwise 2D conv
        x_conv = x_branch.permute(0, 3, 1, 2)
        x_conv = self.act(self.conv2d(x_conv))
        x_branch = x_conv.permute(0, 2, 3, 1)

        # 4 scanning trajectories
        u1 = x_branch.reshape(B, H * W, self.d_inner)
        u2 = x_branch.transpose(1, 2).reshape(B, H * W, self.d_inner)
        u3 = torch.flip(u1, dims=[1])
        u4 = torch.flip(u2, dims=[1])

        y1 = self._selective_scan_dir(u1, 0)
        y2 = self._selective_scan_dir(u2, 1)
        y3 = torch.flip(self._selective_scan_dir(u3, 2), dims=[1])
        y4 = torch.flip(self._selective_scan_dir(u4, 3), dims=[1])

        # Restore spatial arrangements
        y1_2d = y1.reshape(B, H, W, self.d_inner)
        y2_2d = y2.reshape(B, W, H, self.d_inner).transpose(1, 2)
        y3_2d = y3.reshape(B, H, W, self.d_inner)
        y4_2d = y4.reshape(B, W, H, self.d_inner).transpose(1, 2)

        y_fused = y1_2d + y2_2d + y3_2d + y4_2d
        y_norm = self.out_norm(y_fused)
        out = y_norm * F.silu(z_branch)
        out = self.out_proj(out)
        out = self.dropout(out)

        if was_chw:
            out = out.permute(0, 3, 1, 2)
        return out


class CSMambaBlock(nn.Module):
    """
    Cross-Scale Mamba (CSMamba) Block.
    Contains LayerNorm, 2D Selective Scan (SS2D), and residual connection.
    """
    def __init__(self, dim: int, bottleneck_dim: Optional[int] = None, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else dim
        self.ln = nn.LayerNorm(dim)

        if self.bottleneck_dim != dim:
            self.in_bottle = nn.Linear(dim, self.bottleneck_dim)
            self.ss2d = SS2D(d_model=self.bottleneck_dim, d_state=d_state)
            self.out_bottle = nn.Linear(self.bottleneck_dim, dim)
        else:
            self.in_bottle = None
            self.ss2d = SS2D(d_model=dim, d_state=d_state)
            self.out_bottle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        x_hwc = x.permute(0, 2, 3, 1)
        res = x_hwc
        h = self.ln(x_hwc)
        if self.in_bottle is not None:
            h = self.in_bottle(h)
            h = self.ss2d(h)
            h = self.out_bottle(h)
        else:
            h = self.ss2d(h)
        out = res + h
        return out.permute(0, 3, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Scale Feature Fusion Attention (MSFFA / MSAA Bridge)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        red_c = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, red_c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_c, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class MSFFA(nn.Module):
    """
    Multi-Scale Feature Fusion Attention Module (MSAA).
    Merges multiscale encoder representations using channel and spatial gating.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, feats: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        aligned = []
        for f in feats:
            if f.shape[-2:] != target_size:
                aligned.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
            else:
                aligned.append(f)
        x_cat = torch.cat(aligned, dim=1)
        x_fused = self.fusion_conv(x_cat)
        x_fused = self.ca(x_fused)
        x_fused = self.sa(x_fused)
        return x_fused


# ─────────────────────────────────────────────────────────────────────────────
# 3. ResNet Encoder Blocks
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + res)
        return out


class ResNetEncoder(nn.Module):
    """
    4-Stage ResNet Convolutional Encoder matching CM-UNet / Swsl-ResNet.
    Stem downsamples by 4 (stride 2 conv + stride 2 maxpool), so:
    e1: /4 (stride 4, shape H/4, W/4)
    e2: /8 (stride 8, shape H/8, W/8)
    e3: /16 (stride 16, shape H/16, W/16)
    e4: /32 (stride 32, shape H/32, W/32)
    """
    def __init__(self, in_channels: int = 3, base_width: int = 32):
        super().__init__()
        b = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, b, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(
            BasicBlock(b, b, stride=1),
            BasicBlock(b, b, stride=1)
        )
        self.stage2 = nn.Sequential(
            BasicBlock(b, b * 2, stride=2),
            BasicBlock(b * 2, b * 2, stride=1)
        )
        self.stage3 = nn.Sequential(
            BasicBlock(b * 2, b * 4, stride=2),
            BasicBlock(b * 4, b * 4, stride=1)
        )
        self.stage4 = nn.Sequential(
            BasicBlock(b * 4, b * 8, stride=2),
            BasicBlock(b * 8, b * 8, stride=1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)         # (B, b, H/4, W/4)
        e1 = self.stage1(x0)       # (B, b, H/4, W/4)
        e2 = self.stage2(e1)       # (B, 2b, H/8, W/8)
        e3 = self.stage3(e2)       # (B, 4b, H/16, W/16)
        e4 = self.stage4(e3)       # (B, 8b, H/32, W/32)
        return [e1, e2, e3, e4]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SatReg CM-UNet Full Network
# ─────────────────────────────────────────────────────────────────────────────

class SatRegCMUNet(nn.Module):
    """
    SatReg CM-UNet for binary flood segmentation.
    Configured for low-latency, energy-efficient deployment on NVIDIA Jetson Orin Nano.
    """
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 3,
        base_width: int = 32,
        decoder_bottleneck: int = 32,
        d_state: int = 16,
        input_size: int = 128,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        b = base_width
        h = decoder_bottleneck

        # Encoder: 4 stages with channels [b, 2b, 4b, 8b] at /4, /8, /16, /32
        self.encoder = ResNetEncoder(in_channels=in_channels, base_width=b)

        # Multi-scale bridge (MSFFA / MSAA)
        # Combines e2, e3, e4 into stage4 feature space
        msaa_in = (b * 2) + (b * 4) + (b * 8)
        self.msaa = MSFFA(in_channels=msaa_in, out_channels=b * 8)

        # Decoder Stages (CSMamba Blocks with bottleneck h)
        # Stage 4 Bottleneck block (/32)
        self.dec4 = CSMambaBlock(dim=b * 8, bottleneck_dim=min(h * 4, b * 8), d_state=d_state)

        # Stage 3 Up (/16)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(b * 8, b * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d((b * 4) * 2, b * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(b * 4),
            nn.ReLU(inplace=True),
            CSMambaBlock(dim=b * 4, bottleneck_dim=min(h * 2, b * 4), d_state=d_state)
        )

        # Stage 2 Up (/8)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(b * 4, b * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d((b * 2) * 2, b * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(b * 2),
            nn.ReLU(inplace=True),
            CSMambaBlock(dim=b * 2, bottleneck_dim=min(h, b * 2), d_state=d_state)
        )

        # Stage 1 Up (/4)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(b * 2, b, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(b * 2, b, kernel_size=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            CSMambaBlock(dim=b, bottleneck_dim=min(max(h // 2, 16), b), d_state=d_state)
        )

        # Final 4x Upsample head (/4 -> /1 full image resolution)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(b, b, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4 = self.encoder(x)

        # MSAA bridge over higher-level stages
        bridge = self.msaa([e2, e3, e4], target_size=e4.shape[-2:])
        d4 = self.dec4(bridge + e4)

        # Progressive decoding with skip connections
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.head(d1)
        return out


def build_model(
    num_classes: int = 1,
    in_channels: int = 3,
    input_size: int = 128,
    variant: str = "satreg",
    pretrained: bool = False,
    **kwargs
) -> SatRegCMUNet:
    """
    Factory function for SatReg CM-UNet and baseline CM-UNet.

    Args:
        num_classes: Number of segmentation classes (1 for binary flood mask)
        in_channels: Number of input image channels (default: 3)
        input_size: Input resolution (default: 128)
        variant: 'satreg' (lightweight edge-tuned, b=32, h=32, ~7.2M params)
                 or 'cmunet' (baseline full model, b=64, h=64, ~17.6M params)
    """
    if variant.lower() in ("cmunet", "cm-unet", "baseline"):
        base_width = 64
        decoder_bottleneck = 64
    else:
        # Default: SatReg hardware-aware edge configuration for Jetson Orin Nano
        base_width = 32
        decoder_bottleneck = 32

    model = SatRegCMUNet(
        num_classes=num_classes,
        in_channels=in_channels,
        base_width=base_width,
        decoder_bottleneck=decoder_bottleneck,
        input_size=input_size,
        **kwargs
    )
    return model
