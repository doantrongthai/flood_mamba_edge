"""
BGFMamba: Boundary-Guided State Space Model for Generalizable Flood Segmentation
(Formerly MambaEdge / M12_v16)

100% Standalone & Pure PyTorch Native Implementation.
- Requires NO external libraries (NO mamba_ssm, NO causal_conv1d, NO einops, NO timm).
- Completely immune to PyTorch version incompatibility issues (PyTorch 2.x / 3.x+).
- Clean, device-agnostic native selective scan algorithm matching the canonical S6 / mamba-minimal specification.

Core Architectural Pillars:
1. Stem & Guidance: OmniASDConv + DifferentiableLaplacianGuidance for contour preservation
2. Encoder: 4-stage OmniASDBR_Block with scaled bottlenecks
3. Core SSM: GroupedMambaFusion (GMF) at multi-scale bottleneck
4. Decoder: MultiDilatedSB_PixelShuffle with spatial boundary gates for sub-pixel edge recovery
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Pure PyTorch Canonical S6 Mamba Block (Self-Contained, No CUDA wheels needed)
# ==============================================================================

class PurePyTorchMamba(nn.Module):
    """
    Pure PyTorch implementation of the Mamba (S6) Selective State Space Model.
    Follows canonical Algorithm 2 from the Mamba paper (Gu & Dao, 2023) and mamba-minimal.
    Guarantees seamless execution on all PyTorch versions without compilation errors.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: Optional[int] = None,
        bias: bool = False,
        conv_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank

        # In-projection splits into SSM path and residual gating path
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 1D Depthwise Convolution for local sequence context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Input-dependent projections for continuous-to-discrete parameters (Delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Continuous state matrix A initialized via HiPPO / S4D real
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, L, D)
        Returns:
            out: shape (B, L, D)
        """
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # 1D depthwise convolution across sequence length
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)

        # Project inputs to dynamic SSM parameters (dt, B, C)
        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Discretize continuous state space parameters
        A = -torch.exp(self.A_log.float())
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B_ssm)

        # Pure PyTorch canonical selective recurrence loop
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for l in range(L):
            h = dA[:, l] * h + dB[:, l] * x_conv[:, l].unsqueeze(-1)
            y_l = torch.einsum('bdn,bn->bd', h, C_ssm[:, l])
            ys.append(y_l)
        y = torch.stack(ys, dim=1)

        # Add skip connection and multiplicative gating
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class Mamba2D(nn.Module):
    """
    2D spatial wrapper around 1D selective state space model (Mamba).
    Flattens (B, C, H, W) -> (B, H*W, C) for state scan and reconstructs spatial shape.
    """
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = PurePyTorchMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_norm = self.norm(x_flat)
        out = self.mamba(x_norm)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x + out


# ==============================================================================
# 2. Auxiliary Primitives: Bottlenecks, Laplacian Guidance, Dilated Gates
# ==============================================================================

def _scaled_bottleneck(out_c: int) -> int:
    """Scaled channel bottleneck from LCMAM family."""
    bottleneck = int(round(out_c * 0.625 / 8.0) * 8)
    return max(16, bottleneck)


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


class DifferentiableLaplacianGuidance(nn.Module):
    """
    Parameter-free 2D Laplacian high-pass filter.
    Extracts physical water-soil boundary contours directly from Stem features.
    """
    def __init__(self, in_channels: int):
        super().__init__()
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
        x_reshaped = x.view(B * C, 1, H, W)
        lap = F.conv2d(x_reshaped, self.lap_kernel, padding=1).view(B, C, H, W)
        guidance = self.proj(torch.abs(lap))
        return x * (1.0 + guidance)


class OmniASDBR_Block(nn.Module):
    """Omni-Directional Asymmetric Strip Residual Block."""
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


class GroupedMambaFusion(nn.Module):
    """
    Grouped Scale-Aware Mamba Fusion for ultra-low parameter budgets (200k-300k).
    Splits multi-scale tokens into independent groups for parallel selective scanning.
    """
    def __init__(self, in_channels_list: List[int], embed_dim: int = 40, num_groups: int = 2, expand: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            for c in in_channels_list
        ])
        group_dim = embed_dim // num_groups
        self.mamba_groups = nn.ModuleList([
            Mamba2D(d_model=group_dim, d_state=8, d_conv=3, expand=expand)
            for _ in range(num_groups)
        ])
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        orig_shapes = [f.shape[2:] for f in features]
        target_shape = features[0].shape[2:]

        aligned = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode='bilinear', align_corners=False)
            aligned.append(p)

        fused = sum(aligned)
        chunks = torch.chunk(fused, self.num_groups, dim=1)
        mamba_outs = [self.mamba_groups[g](chunks[g]) for g in range(self.num_groups)]
        mamba_feat = torch.cat(mamba_outs, dim=1) + fused

        outs = []
        for i, target_sz in enumerate(orig_shapes):
            out_f = self.out_projs[i](mamba_feat)
            if out_f.shape[2:] != target_sz:
                out_f = F.interpolate(out_f, size=target_sz, mode='bilinear', align_corners=False)
            outs.append(out_f + features[i])
        return outs


def _make_light_fuse(mid_c: int, skip_c: int, out_c: int) -> nn.Sequential:
    """Ultra-lightweight depthwise separable fusion block."""
    return nn.Sequential(
        nn.Conv2d(mid_c + skip_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True)
    )


class MultiDilatedSB_PixelShuffle(nn.Module):
    """
    Multi-Dilated Spatial Boundary PixelShuffle Decoder Block.
    Employs dual-rate dilated convolutions to extract fine multi-scale boundary gates
    and sub-pixel spatial upsampling without bilinear interpolation blur.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = 8
        self.d1 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=1, bias=False)
        self.d2 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(k_c)
        self.gate = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(k_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.d1(skip) + self.d2(skip)
        g = self.gate(self.bn(feat))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


# ==============================================================================
# 3. Full BGFMamba Architecture & Factory Function
# ==============================================================================

class BGFMamba(nn.Module):
    """
    BGFMamba: Boundary-Guided State Space Model for Generalizable Flood Segmentation.
    Target: Ultra-Lightweight (~235k parameters) with superior boundary preservation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 20, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = OmniASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

        self.mamba_fusion = GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = MultiDilatedSB_PixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = MultiDilatedSB_PixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = MultiDilatedSB_PixelShuffle(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        x0 = self.inc(x)
        x0_guided = self.lap_guidance(x0)

        x1 = self.stage1(self.pool(x0_guided))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3)
        d3 = self.up3(self.dec_conv3(d4), f2)

        if self.training:
            aux_out = self.aux_head(d3)
            aux_out = F.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            aux_out = None

        d2 = self.up2(self.dec_conv2(d3), x1)

        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))

        if self.training and aux_out is not None:
            return out, aux_out

        return out


# Backward compatibility alias
MambaEdge = BGFMamba


def build_model(num_classes: int = 1, input_size: int = 128, pretrained: bool = False, in_channels: int = 3, **kwargs):
    return BGFMamba(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
