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


# ─────────────────────────────────────────────────────────────────────────────
# Component 8: Inverted Residual Omni-ASD Block (InvRes-Omni)
# Fusing 2x Channel Expansion from Model 4 with Omni-Directional Strip Conv
# ─────────────────────────────────────────────────────────────────────────────

class InvResOmniBlock(nn.Module):
    """
    MobileNetV2-style Inverted Residual with Omni-ASD Conv.
    Expands channel dimensions 2x before omni-directional depthwise filtering
    for maximum representation of fine UAV flood details (debris, narrow streets).
    """
    def __init__(self, in_c: int, out_c: int, expand_ratio: int = 2):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.conv = OmniASDConv(mid_c, mid_c, k=7)
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
# Component 9: Direct Physical Laplacian-Coupled Boundary PixelShuffle (LapSBG)
# Injects stem physical Laplacian contour prior directly into decoder spatial gate
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianSpatialBoundaryPixelShuffle(nn.Module):
    """
    Direct Physical Laplacian-Coupled Boundary-Gated PixelShuffle Block.
    Injects high-frequency stem Laplacian boundary prior directly into 2D spatial gate.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(skip_c + 1, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
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

    def forward(self, x: torch.Tensor, skip: torch.Tensor, lap_prior: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        if lap_prior.shape[2:] != skip.shape[2:]:
            lap_prior = F.interpolate(lap_prior, size=skip.shape[2:], mode="bilinear", align_corners=False)
        gate_in = torch.cat([skip, lap_prior], dim=1)
        skip_gated = skip * (1.0 + self.spatial_gate(gate_in))
        return self.fuse(torch.cat([x_up, skip_gated], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Component 10: Reparameterizable Omni-ASD Convolution (RepOmniConv)
# Multi-branch strip conv during training -> Fused single 7x7 depthwise on Jetson
# ─────────────────────────────────────────────────────────────────────────────

class RepOmniConv(nn.Module):
    """
    Reparameterizable Omni-ASD Convolution.
    Training mode: parallel 1x7, 7x1, 3x3 depthwise convs.
    Inference mode: mathematically collapsed into a single 7x7 depthwise conv for zero Jetson latency overhead.
    """
    def __init__(self, in_c: int, out_c: int, k: int = 7):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.pw_in = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        self.is_deployed = False

        # Multi-branch training mode
        self.dw_h = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, k // 2), groups=out_c, bias=False)
        self.bn_h = nn.BatchNorm2d(out_c)
        self.dw_v = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(k // 2, 0), groups=out_c, bias=False)
        self.bn_v = nn.BatchNorm2d(out_c)
        self.dw_s = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False)
        self.bn_s = nn.BatchNorm2d(out_c)

        self.act = nn.SiLU(inplace=True)
        self.pw_out = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.dw_fused = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        res = x
        if self.is_deployed and self.dw_fused is not None:
            fused = self.dw_fused(x)
        else:
            f_h = self.bn_h(self.dw_h(x))
            f_v = self.bn_v(self.dw_v(x))
            f_s = self.bn_s(self.dw_s(x))
            fused = f_h + f_v + f_s

        out = self.act(fused)
        out = self.pw_out(out)
        return out + res

    def switch_to_deploy(self):
        if self.is_deployed:
            return
        k = self.k
        w_fused = torch.zeros(self.out_c, 1, k, k, device=self.dw_h.weight.device, dtype=self.dw_h.weight.dtype)
        b_fused = torch.zeros(self.out_c, device=self.dw_h.weight.device, dtype=self.dw_h.weight.dtype)

        # Fuse horizontal
        std_h = torch.sqrt(self.bn_h.running_var + self.bn_h.eps)
        w_h = self.dw_h.weight * (self.bn_h.weight / std_h).view(-1, 1, 1, 1)
        b_h = self.bn_h.bias - self.bn_h.running_mean * self.bn_h.weight / std_h
        pad_top = (k - 1) // 2
        w_fused[:, :, pad_top:pad_top+1, :] += w_h
        b_fused += b_h

        # Fuse vertical
        std_v = torch.sqrt(self.bn_v.running_var + self.bn_v.eps)
        w_v = self.dw_v.weight * (self.bn_v.weight / std_v).view(-1, 1, 1, 1)
        b_v = self.bn_v.bias - self.bn_v.running_mean * self.bn_v.weight / std_v
        pad_left = (k - 1) // 2
        w_fused[:, :, :, pad_left:pad_left+1] += w_v
        b_fused += b_v

        # Fuse square 3x3
        std_s = torch.sqrt(self.bn_s.running_var + self.bn_s.eps)
        w_s = self.dw_s.weight * (self.bn_s.weight / std_s).view(-1, 1, 1, 1)
        b_s = self.bn_s.bias - self.bn_s.running_mean * self.bn_s.weight / std_s
        pad_s = (k - 3) // 2
        w_fused[:, :, pad_s:pad_s+3, pad_s:pad_s+3] += w_s
        b_fused += b_s

        self.dw_fused = nn.Conv2d(self.out_c, self.out_c, kernel_size=k, padding=k//2, groups=self.out_c, bias=True)
        self.dw_fused.weight.data.copy_(w_fused)
        self.dw_fused.bias.data.copy_(b_fused)

        del self.dw_h, self.bn_h, self.dw_v, self.bn_v, self.dw_s, self.bn_s
        self.is_deployed = True


class RepOmniASDBR_Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(inplace=True),
        )
        self.conv = RepOmniConv(bottleneck, bottleneck, k=7)
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
# Component 11: Coordinate-Aware Spatial Gated Refinement (CoordSGR)
# Injects normalized (x, y) spatial coordinates for topographic waterflow awareness
# ─────────────────────────────────────────────────────────────────────────────

class CoordSGR(nn.Module):
    """
    Coordinate-Aware Spatial Gated Refinement Module.
    Injects normalized 2D spatial coordinate grids into the spatial gating mechanism
    to model gravity-driven topographic water distribution.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 2, max(8, dim // 4), 1, bias=False),
            nn.BatchNorm2d(max(8, dim // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, dim // 4), 1, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        feat = self.conv(x)
        g = self.gate(torch.cat([feat, coords], dim=1))
        out = self.proj(feat * g)
        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# Component 12: 4-Way Bidirectional Cross-Scan Mamba Fusion (BiCrossScanMamba)
# Horizontal forward/backward + Vertical forward/backward 2D omni-scanning
# ─────────────────────────────────────────────────────────────────────────────

class BiCrossScanMambaFusion(nn.Module):
    """
    4-Way Bidirectional Cross-Scan Multi-Scale Mamba Fusion.
    Captures isotropic flood propagation via 4 scan trajectories:
    1. Horizontal Forward  (left to right)
    2. Horizontal Backward (right to left)
    3. Vertical Forward    (top to bottom)
    4. Vertical Backward   (bottom to top)
    """
    def __init__(self, in_dims: List[int], embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in in_dims
        ])
        self.scale_embeddings = nn.Parameter(torch.randn(len(in_dims), embed_dim) * 0.02)
        self.h_mamba = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=1)
        self.v_mamba = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=1)
        self.refiners = nn.ModuleList([SGR(embed_dim) for _ in in_dims])
        self.fuse_hv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = xs[0].shape[0]
        h_tokens, v_tokens, shapes = [], [], []

        for i, x in enumerate(xs):
            feat = self.projs_in[i](x) + self.scale_embeddings[i].view(1, self.embed_dim, 1, 1)
            h, w = feat.shape[-2:]
            shapes.append((h, w))
            h_tokens.append(feat.flatten(2).transpose(1, 2))
            v_tokens.append(feat.permute(0, 1, 3, 2).flatten(2).transpose(1, 2))

        h_seq = self.h_mamba(torch.cat(h_tokens, dim=1))
        v_seq = self.v_mamba(torch.cat(v_tokens, dim=1))

        outputs = []
        cursor = 0
        for i, (h, w) in enumerate(shapes):
            length = h * w
            hf = h_seq[:, cursor : cursor + length, :].transpose(1, 2).reshape(batch, self.embed_dim, h, w)
            vf = v_seq[:, cursor : cursor + length, :].transpose(1, 2).reshape(batch, self.embed_dim, w, h).permute(0, 1, 3, 2)
            cursor += length

            fused = self.fuse_hv(torch.cat([hf, vf], dim=1))
            fused = self.refiners[i](fused)
            out = self.projs_out[i](fused)
            outputs.append(out + xs[i])

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Component 13: Haar Discrete Wavelet Transform Residual Skip (HaarWaveletSkip)
# Parameter-free physical high-frequency edge isolation
# ─────────────────────────────────────────────────────────────────────────────

class HaarWaveletSkip(nn.Module):
    """
    2D Haar Wavelet Decomposition Skip.
    Isolates orthogonal high-frequency edge bands (LH, HL, HH) without learnable parameters.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c * 3, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        high_pass = torch.cat([lh, hl, hh], dim=1)
        return self.proj(high_pass)


# ─────────────────────────────────────────────────────────────────────────────
# Component 14: Asymmetric Strip Pyramid Pooling Module (AsymStripPPM)
# ─────────────────────────────────────────────────────────────────────────────

class AsymStripPPM(nn.Module):
    """
    Asymmetric Strip Pyramid Pooling Module.
    Combines global pooling with horizontal (1xW) and vertical (Hx1) strip pooling.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Sequential(nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c), nn.SiLU(inplace=True))

        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Sequential(nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c), nn.SiLU(inplace=True))

        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_v = nn.Sequential(nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c), nn.SiLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_c + mid_c * 3, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        g = F.interpolate(self.conv_global(self.pool_global(x)), size=(H, W), mode='bilinear', align_corners=False)
        h = F.interpolate(self.conv_h(self.pool_h(x)), size=(H, W), mode='bilinear', align_corners=False)
        v = F.interpolate(self.conv_v(self.pool_v(x)), size=(H, W), mode='bilinear', align_corners=False)
        return self.conv_out(torch.cat([x, g, h, v], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Component 15: Dynamic Feature Alignment Decoder (DynamicAlignPixelShuffle)
# ─────────────────────────────────────────────────────────────────────────────

class DynamicAlignPixelShuffle(nn.Module):
    """
    Dynamic Feature Alignment PixelShuffle Decoder Block.
    Learns spatial deformation offsets to align skip features with upsampled semantic features.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(mid_c + skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), skip_c, kernel_size=1)
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(skip_c, 1, kernel_size=1),
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
        offset = self.offset_conv(torch.cat([x_up, skip], dim=1))
        aligned_skip = skip + torch.tanh(offset) * skip
        skip_gated = aligned_skip * (1.0 + self.spatial_gate(aligned_skip))
        return self.fuse(torch.cat([x_up, skip_gated], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Component 16: Content-Aware Boundary Modulated Attention (CABMA_PixelShuffle)
# ─────────────────────────────────────────────────────────────────────────────

class CABMA_PixelShuffle(nn.Module):
    """
    Content-Aware Boundary Modulated Attention PixelShuffle.
    Suppresses water reflection glares by checking local spatial variance.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.box_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.mod_gate = nn.Sequential(
            nn.Conv2d(skip_c + 1, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
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
        skip_mean = skip.mean(dim=1, keepdim=True)
        local_mean = self.box_pool(skip_mean)
        local_sq = self.box_pool(skip_mean ** 2)
        local_var = torch.clamp(local_sq - local_mean ** 2, min=1e-6)
        mod_in = torch.cat([skip, local_var], dim=1)
        skip_gated = skip * (1.0 + self.mod_gate(mod_in))
        return self.fuse(torch.cat([x_up, skip_gated], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Component 17: Multi-Dilated Strip Pyramid Block (DilPyramidBlock)
# ─────────────────────────────────────────────────────────────────────────────

class DilPyramidBlock(nn.Module):
    """
    Multi-Dilated Strip Pyramid Block.
    Parallel strip convolutions with dilations (d=1, 2, 3) to adaptively capture
    narrow street runoff as well as wide riverbanks.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c = out_c // 2
        self.pw_in = nn.Conv2d(in_c, mid_c, 1, bias=False) if in_c != mid_c else nn.Identity()
        self.strip_d1 = nn.Conv2d(mid_c, mid_c, (1, 7), padding=(0, 3), dilation=1, groups=mid_c, bias=False)
        self.strip_d2 = nn.Conv2d(mid_c, mid_c, (7, 1), padding=(3, 0), dilation=1, groups=mid_c, bias=False)
        self.strip_d3 = nn.Conv2d(mid_c, mid_c, 3, padding=2, dilation=2, groups=mid_c, bias=False)
        self.bn = nn.BatchNorm2d(mid_c)
        self.act = nn.SiLU(inplace=True)
        self.pw_out = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        feat = self.pw_in(x)
        fused = self.strip_d1(feat) + self.strip_d2(feat) + self.strip_d3(feat)
        out = self.pw_out(self.act(self.bn(fused)))
        return self.act(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# Component 18: Differentiable Morphological Boundary Refiner (MorphoBoundaryRefiner)
# ─────────────────────────────────────────────────────────────────────────────

class MorphoBoundaryRefiner(nn.Module):
    """
    Differentiable Soft Morphological Boundary Refiner.
    Computes a transition zone boundary map: B = Dilation(F) - Erosion(F).
    """
    def __init__(self, in_c: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_c + 1, max(8, in_c // 4), 1, bias=False),
            nn.BatchNorm2d(max(8, in_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, in_c // 4), in_c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1, keepdim=True)
        dil = F.max_pool2d(x_mean, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-x_mean, kernel_size=3, stride=1, padding=1)
        transition = dil - ero
        gate = self.gate(torch.cat([x, transition], dim=1))
        return x * (1.0 + gate)


# ─────────────────────────────────────────────────────────────────────────────
# Component 19: Edge-Enhanced Selective Feature Fusion (SelectiveFeatureFusion)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveFeatureFusion(nn.Module):
    """
    Selective Feature Fusion module for Skip Connections.
    Dynamically weights skip feature channels vs upsampled channels via channel cross-attention.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        total_c = mid_c + skip_c
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_c, max(8, total_c // 4), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, total_c // 4), total_c, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(total_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        cat_feat = torch.cat([x_up, skip], dim=1)
        attn = self.ca(cat_feat)
        return self.conv(cat_feat * attn)


# ─────────────────────────────────────────────────────────────────────────────
# Component 20: ConvNeXt-Style Inverted Omni-ASD Block (OmniNeXtBlock)
# ─────────────────────────────────────────────────────────────────────────────

class OmniNeXtBlock(nn.Module):
    """
    ConvNeXt-style Inverted Omni-ASD Block.
    Large 7x7 depthwise + LayerNorm + Inverted 2x Expansion + Omni-ASD strip depthwise.
    """
    def __init__(self, in_c: int, out_c: int, expand: int = 2):
        super().__init__()
        mid_c = in_c * expand
        self.dw = nn.Conv2d(in_c, in_c, 7, padding=3, groups=in_c, bias=False)
        self.norm = nn.GroupNorm(1, in_c)
        self.pw_expand = nn.Conv2d(in_c, mid_c, 1)
        self.act = nn.GELU()
        self.omni = OmniASDConv(mid_c, mid_c, k=7)
        self.pw_project = nn.Conv2d(mid_c, out_c, 1)
        self.shortcut = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.omni(x)
        x = self.pw_project(x)
        return x + res
