import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_block import Mamba
from models.lcmamnet import TokenProjector, BiMambaPreNorm, _scaled_bottleneck
from models.lcmam_variants import ASDConv, ASDBR_Block, SGR, DecoupledMambaFusion, LitePixelShuffleUpBlock


# ─────────────────────────────────────────────────────────────────────────────
# Component 1: Spatial Boundary-Gated PixelShuffle Decoder Block (SBG-PixelShuffle)
# Replaces 1D channel pooling with full 2D spatial coordinate boundary attention
# ─────────────────────────────────────────────────────────────────────────────

class SpatialBoundaryPixelShuffle(nn.Module):
    """
    Spatial Boundary-Gated PixelShuffle Block.
    Unlike channel-averaged pooling, maintains 2D spatial coordinates (H x W)
    to sharply localize fine flood-land boundary transitions.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        # Full 2D Spatial Boundary Gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip_gated = skip * (1.0 + self.spatial_gate(skip))
        return self.fuse(torch.cat([x_up, skip_gated], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Component 2: Scale-Aware Multi-Scale Mamba Fusion
# Injects learnable resolution scale embeddings into tokens before 1D SSM scanning
# ─────────────────────────────────────────────────────────────────────────────

class ScaleAwareMambaFusion(nn.Module):
    """
    Scale-Aware Multi-Scale Mamba Fusion.
    Assigns learnable scale embeddings to Stage 2, 3, 4 tokens, allowing
    Mamba SSM state transitions to distinguish pyramid resolution boundaries.
    """
    def __init__(self, in_dims: List[int], embed_dim: int = 64, use_sgr: bool = True, expand: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_stages = len(in_dims)
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in in_dims
        ])
        # Scale embeddings: one vector per pyramid stage
        self.scale_embeddings = nn.Parameter(torch.randn(self.num_stages, embed_dim) * 0.02)
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=expand)
        self.refiners = nn.ModuleList([SGR(embed_dim) for _ in in_dims])

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = xs[0].shape[0]
        tokens = []
        shapes = []
        for i, x in enumerate(xs):
            feat = self.projs_in[i](x)
            # Add scale embedding across spatial dimension
            scale_bias = self.scale_embeddings[i].view(1, self.embed_dim, 1, 1)
            feat = feat + scale_bias
            h, w = feat.shape[-2:]
            shapes.append((h, w))
            tokens.append(feat.flatten(2).transpose(1, 2))

        seq = self.mixer(torch.cat(tokens, dim=1))
        outputs = []
        cursor = 0
        for i, (h, w) in enumerate(shapes):
            length = h * w
            feat = seq[:, cursor : cursor + length, :]
            cursor += length
            feat = feat.transpose(1, 2).reshape(batch, self.embed_dim, h, w)
            feat = self.refiners[i](feat)
            out = self.projs_out[i](feat)
            outputs.append(out + xs[i])
        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Component 3: Dynamic Omni-Directional Strip Convolution (Omni-ASDConv)
# Parallel Horizontal (1x7), Vertical (7x1), and Dilated Square (3x3, d=2) with Softmax Gating
# ─────────────────────────────────────────────────────────────────────────────

class OmniASDConv(nn.Module):
    """
    Dynamic Omni-Directional Strip Convolution.
    Captures orthogonal flows (1x7, 7x1) and diagonal/curved flow features (3x3 dilated)
    with adaptive channel attention gating.
    """
    def __init__(self, in_c: int, out_c: int, k: int = 7):
        super().__init__()
        self.pw_in = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        self.dw_h = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, k // 2), groups=out_c, bias=False)
        self.dw_v = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(k // 2, 0), groups=out_c, bias=False)
        self.dw_d = nn.Conv2d(out_c, out_c, kernel_size=3, padding=2, dilation=2, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        # Softmax branch selector
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.pw_out = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        res = x
        fh = self.dw_h(x)
        fv = self.dw_v(x)
        fd = self.dw_d(x)

        weights = self.atten(x)  # (B, 3, 1, 1)
        fused = (
            fh * weights[:, 0:1, :, :] +
            fv * weights[:, 1:2, :, :] +
            fd * weights[:, 2:3, :, :]
        )
        strip = self.act(self.bn(fused))
        out = self.pw_out(strip)
        return out + res


class OmniASDBR_Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(inplace=True),
        )
        self.conv = OmniASDConv(bottleneck, bottleneck, k=7)
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.reduce(x)
        x = self.conv(x)
        x = self.expand(x)
        return self.act(x + res)


# ─────────────────────────────────────────────────────────────────────────────
# Component 4: Inverted Residual ASD Bottleneck Block (InvRes-ASD)
# ─────────────────────────────────────────────────────────────────────────────

class InvResASDBlock(nn.Module):
    """
    MobileNetV2-style Inverted Bottleneck with ASD-Conv.
    Expands channel dimensions before strip depthwise filtering for richer representation.
    """
    def __init__(self, in_c: int, out_c: int, expand_ratio: int = 2):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.conv = ASDConv(mid_c, mid_c, k=7)
        self.project = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.expand(x)
        x = self.conv(x)
        x = self.project(x)
        return self.act(x + res)


# ─────────────────────────────────────────────────────────────────────────────
# Component 5: Differentiable Laplacian Boundary Guidance Module
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableLaplacianGuidance(nn.Module):
    """
    Parameter-free 2D Laplacian high-pass filter.
    Extracts physical water-soil boundary contours directly from Stem features.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Laplacian second-order kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0],
                                   [1.0, -4.0, 1.0],
                                   [0.0, 1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('lap_kernel', lap_kernel)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, max(8, in_channels // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, in_channels // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, in_channels // 4), in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Apply depthwise Laplacian
        x_reshaped = x.view(B * C, 1, H, W)
        lap = F.conv2d(x_reshaped, self.lap_kernel, padding=1).view(B, C, H, W)
        guidance = self.proj(torch.abs(lap))
        return x * (1.0 + guidance)


# ─────────────────────────────────────────────────────────────────────────────
# Component 6: Multi-Dilated Spatial-Gated Refinement (MultiDil-SGR)
# ─────────────────────────────────────────────────────────────────────────────

class MultiDilatedSGR(nn.Module):
    """
    Multi-Scale Dilated Refinement with Dilations (1, 2, 4).
    Simultaneously handles fine flood fringes and large inland lakes.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.d1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim, bias=False)
        self.d2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False)
        self.d4 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.fuse_dil = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.d1(x)
        f2 = self.d2(x)
        f4 = self.d4(x)
        f_cat = self.fuse_dil(torch.cat([f1, f2, f4], dim=1))
        f = self.act(self.bn(f_cat))
        g = self.gate(f)
        out = self.proj(f * g)
        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# Component 7: Dual-Path Local-Global Fusion Bottleneck
# Parallel 2D Local Depthwise Conv + 1D Global BiMamba
# ─────────────────────────────────────────────────────────────────────────────

class DualPathMambaFusion(nn.Module):
    """
    Dual-Path Hybrid Fusion: Parallel 2D Local Depthwise Conv + 1D Global BiMamba.
    Blends local texture detail with global context via adaptive learned gating.
    """
    def __init__(self, in_dims: List[int], embed_dim: int = 64, expand: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in in_dims
        ])
        # Local 2D Path
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True)
            ) for _ in in_dims
        ])
        # Global 1D Mamba Path
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=expand)
        self.refiners = nn.ModuleList([SGR(embed_dim) for _ in in_dims])
        # Gating parameter
        self.gate_alpha = nn.Parameter(torch.full((len(in_dims), embed_dim, 1, 1), 0.5))

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = xs[0].shape[0]
        tokens = []
        shapes = []
        local_feats = []

        for i, x in enumerate(xs):
            feat = self.projs_in[i](x)
            local_feat = self.local_convs[i](feat)
            local_feats.append(local_feat)
            h, w = feat.shape[-2:]
            shapes.append((h, w))
            tokens.append(feat.flatten(2).transpose(1, 2))

        seq = self.mixer(torch.cat(tokens, dim=1))
        outputs = []
        cursor = 0
        for i, (h, w) in enumerate(shapes):
            length = h * w
            mamba_feat = seq[:, cursor : cursor + length, :]
            cursor += length
            mamba_feat = mamba_feat.transpose(1, 2).reshape(batch, self.embed_dim, h, w)
            mamba_feat = self.refiners[i](mamba_feat)

            # Adaptive Fusion of Local and Global
            alpha = torch.sigmoid(self.gate_alpha[i])
            hybrid = alpha * mamba_feat + (1.0 - alpha) * local_feats[i]

            out = self.projs_out[i](hybrid)
            outputs.append(out + xs[i])
        return outputs
