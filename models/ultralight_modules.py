import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_block import Mamba


class Mamba2D(nn.Module):
    """
    2D spatial wrapper around 1D selective state space model (Mamba).
    Flattens (B, C, H, W) -> (B, H*W, C) for state scan and reconstructs spatial shape.
    """
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_norm = self.norm(x_flat)
        out = self.mamba(x_norm)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x + out


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


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ultra-Lightweight Mamba SSM Fusion Variants
# ─────────────────────────────────────────────────────────────────────────────

class GroupedMambaFusion(nn.Module):
    """
    Grouped Scale-Aware Mamba Fusion for ultra-low parameter budgets (200k-300k).
    Splits multi-scale tokens into 2 independent groups for parallel selective scanning,
    reducing SSM parameter footprint by 50% while accelerating memory throughput.
    """
    def __init__(self, in_channels_list, embed_dim=40, num_groups=2, expand=1):
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

    def forward(self, features):
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


class FactorizedMambaFusion(nn.Module):
    """
    Factorized Low-Rank Mamba Fusion.
    Projects spatial-channel tokens through a bottleneck rank before 2D Mamba scanning,
    maintaining global receptive field at a fraction of the parameter cost.
    """
    def __init__(self, in_channels_list, embed_dim=48, rank=24, expand=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            for c in in_channels_list
        ])
        self.down_proj = nn.Conv2d(embed_dim, rank, kernel_size=1, bias=False)
        self.mamba = Mamba2D(d_model=rank, d_state=8, d_conv=3, expand=expand)
        self.up_proj = nn.Sequential(
            nn.Conv2d(rank, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in in_channels_list
        ])

    def forward(self, features):
        orig_shapes = [f.shape[2:] for f in features]
        target_shape = features[0].shape[2:]

        aligned = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode='bilinear', align_corners=False)
            aligned.append(p)

        fused = sum(aligned)
        bottleneck = self.down_proj(fused)
        mamba_feat = self.up_proj(self.mamba(bottleneck)) + fused

        outs = []
        for i, target_sz in enumerate(orig_shapes):
            out_f = self.out_projs[i](mamba_feat)
            if out_f.shape[2:] != target_sz:
                out_f = F.interpolate(out_f, size=target_sz, mode='bilinear', align_corners=False)
            outs.append(out_f + features[i])
        return outs


class SharedRecurrentMambaFusion(nn.Module):
    """
    Shared-State Recurrent Mamba Fusion (SSR-Mamba).
    Shares a single recurrent 2D Mamba block across all pyramid scales,
    eliminating redundant parameter duplication.
    """
    def __init__(self, in_channels_list, embed_dim=40, expand=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            for c in in_channels_list
        ])
        self.shared_mamba = Mamba2D(d_model=embed_dim, d_state=8, d_conv=3, expand=expand)
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in in_channels_list
        ])

    def forward(self, features):
        orig_shapes = [f.shape[2:] for f in features]
        outs = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            p_mamba = self.shared_mamba(p)
            out = self.out_projs[i](p_mamba)
            outs.append(out + f)
        return outs


class DecoupledDirectionalMambaFusion(nn.Module):
    """
    Decoupled Orthogonal Directional Mamba Fusion (ODS-Mamba).
    Scans feature tokens along horizontal and vertical trajectories separately.
    """
    def __init__(self, in_channels_list, embed_dim=40, expand=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            for c in in_channels_list
        ])
        half_dim = embed_dim // 2
        self.mamba_h = Mamba2D(d_model=half_dim, d_state=8, d_conv=3, expand=expand)
        self.mamba_w = Mamba2D(d_model=half_dim, d_state=8, d_conv=3, expand=expand)
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in in_channels_list
        ])

    def forward(self, features):
        orig_shapes = [f.shape[2:] for f in features]
        target_shape = features[0].shape[2:]
        aligned = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode='bilinear', align_corners=False)
            aligned.append(p)
        fused = sum(aligned)
        c1, c2 = torch.chunk(fused, 2, dim=1)
        h_out = self.mamba_h(c1)
        w_out = self.mamba_w(c2.transpose(2, 3)).transpose(2, 3)
        mamba_feat = torch.cat([h_out, w_out], dim=1) + fused

        outs = []
        for i, target_sz in enumerate(orig_shapes):
            out_f = self.out_projs[i](mamba_feat)
            if out_f.shape[2:] != target_sz:
                out_f = F.interpolate(out_f, size=target_sz, mode='bilinear', align_corners=False)
            outs.append(out_f + features[i])
        return outs


class SparseTokenMambaFusion(nn.Module):
    """
    Dynamic Sparse-Token Mamba Fusion.
    Computes spatial boundary saliency map to identify top informative tokens.
    """
    def __init__(self, in_channels_list, embed_dim=40, expand=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim)
            )
            for c in in_channels_list
        ])
        self.saliency_head = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.mamba = Mamba2D(d_model=embed_dim, d_state=8, d_conv=3, expand=expand)
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True)
            )
            for c in in_channels_list
        ])

    def forward(self, features):
        orig_shapes = [f.shape[2:] for f in features]
        target_shape = features[0].shape[2:]
        aligned = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode='bilinear', align_corners=False)
            aligned.append(p)
        fused = sum(aligned)
        saliency = self.saliency_head(fused)
        mamba_feat = self.mamba(fused * saliency) + fused

        outs = []
        for i, target_sz in enumerate(orig_shapes):
            out_f = self.out_projs[i](mamba_feat)
            if out_f.shape[2:] != target_sz:
                out_f = F.interpolate(out_f, size=target_sz, mode='bilinear', align_corners=False)
            outs.append(out_f + features[i])
        return outs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Novel CABMA Variants (Model 11 Family)
# ─────────────────────────────────────────────────────────────────────────────

class FCABMA_PixelShuffle(nn.Module):
    """Frequency-Gated CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.box = nn.AvgPool2d(3, stride=1, padding=1)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        low_f = self.box(skip)
        high_f = torch.abs(skip - low_f)
        g = self.freq_gate(high_f)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class ASCABMA_PixelShuffle(nn.Module):
    """Asymmetric Strip CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.strip_h = nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.strip_w = nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        strip_feat = self.strip_h(skip) + self.strip_w(skip)
        g = self.gate_conv(strip_feat)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class CoordCABMA_PixelShuffle(nn.Module):
    """Coordinate Attention CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = max(8, skip_c // 4)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_shared = nn.Sequential(
            nn.Conv2d(skip_c, k_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(k_c),
            nn.SiLU(inplace=True)
        )
        self.conv_h = nn.Conv2d(k_c, 1, kernel_size=1)
        self.conv_w = nn.Conv2d(k_c, 1, kernel_size=1)
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        B, C, H, W = skip.shape
        x_h = self.pool_h(skip)
        x_w = self.pool_w(skip).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv_shared(y)
        y_h, y_w = torch.split(y, [H, W], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.transpose(2, 3)))
        g = a_h * a_w
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class MorphoCABMA_PixelShuffle(nn.Module):
    """Morphological Difference CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.morpho_gate = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dil = F.max_pool2d(skip, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-skip, kernel_size=3, stride=1, padding=1)
        g = self.morpho_gate(dil - ero)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class SobelLaplacianCABMA_PixelShuffle(nn.Module):
    """Sobel-Laplacian Dual-Prior CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('lap', lap)

        self.dual_gate = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip_gray = skip.mean(dim=1, keepdim=True)
        gx = F.conv2d(skip_gray, self.sobel_x, padding=1)
        gy = F.conv2d(skip_gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        lap_mag = torch.abs(F.conv2d(skip_gray, self.lap, padding=1))
        dual_prior = torch.cat([grad_mag, lap_mag], dim=1)
        g = self.dual_gate(dual_prior)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class ContrastNormPixelShuffle(nn.Module):
    """Contrast-Normalized Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.contrast_gate = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        mean = F.avg_pool2d(skip, kernel_size=5, stride=1, padding=2)
        sq_mean = F.avg_pool2d(skip ** 2, kernel_size=5, stride=1, padding=2)
        variance = torch.clamp(sq_mean - mean ** 2, min=1e-6)
        rms = torch.sqrt(variance)
        g = self.contrast_gate(rms)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class WaveletCABMA_PixelShuffle(nn.Module):
    """Haar Wavelet Subband Boundary CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(skip_c * 3, max(8, skip_c // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x00 = skip[:, :, 0::2, 0::2]
        x01 = skip[:, :, 0::2, 1::2]
        x10 = skip[:, :, 1::2, 0::2]
        x11 = skip[:, :, 1::2, 1::2]
        lh = x00 - x01 + x10 - x11
        hl = x00 + x01 - x10 - x11
        hh = x00 - x01 - x10 + x11
        hf = torch.cat([lh, hl, hh], dim=1)
        hf_up = F.interpolate(hf, size=skip.shape[2:], mode='bilinear', align_corners=False)
        g = self.hf_conv(hf_up)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class CurvatureCABMA_PixelShuffle(nn.Module):
    """Curvature-Guided CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.curv_head = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        pad = F.pad(skip, (1, 1, 1, 1), mode='replicate')
        curv_x = pad[:, :, 1:-1, 2:] + pad[:, :, 1:-1, :-2] - 2.0 * skip
        curv_y = pad[:, :, 2:, 1:-1] + pad[:, :, :-2, 1:-1] - 2.0 * skip
        curv_mag = torch.sqrt(curv_x ** 2 + curv_y ** 2 + 1e-6)
        g = self.curv_head(curv_mag)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class CrossScaleCABMA_PixelShuffle(nn.Module):
    """Cross-Scale Boundary Injection CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c, stem_c=20):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.stem_proj = nn.Conv2d(stem_c, 8, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_c + 8, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip, stem_guide=None):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if stem_guide is not None:
            guide_down = F.adaptive_avg_pool2d(stem_guide, skip.shape[2:])
            guide_feat = self.stem_proj(guide_down)
            gate_in = torch.cat([skip, guide_feat], dim=1)
        else:
            dummy_guide = torch.zeros(skip.size(0), 8, skip.size(2), skip.size(3), device=skip.device, dtype=skip.dtype)
            gate_in = torch.cat([skip, dummy_guide], dim=1)
        g = self.gate(gate_in)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class BoundarySqueezeCABMA_PixelShuffle(nn.Module):
    """Boundary Squeeze-and-Amplification CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.squeeze = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        edge_weight = self.squeeze(skip)
        amp = torch.exp(edge_weight) - 1.0
        return self.fuse(torch.cat([x_up, skip * (1.0 + amp)], dim=1))


class MultiDilatedCABMA_PixelShuffle(nn.Module):
    """Multi-Dilated Strip CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.d1 = nn.Conv2d(skip_c, skip_c, kernel_size=(1, 5), padding=(0, 2), groups=skip_c, bias=False)
        self.d2 = nn.Conv2d(skip_c, skip_c, kernel_size=(1, 5), padding=(0, 4), dilation=(1, 2), groups=skip_c, bias=False)
        self.bn_dil = nn.BatchNorm2d(skip_c)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dil_feat = self.d1(skip) + self.d2(skip)
        g = self.gate(self.bn_dil(dil_feat))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class SelfCalibratedCABMA_PixelShuffle(nn.Module):
    """Self-Calibrating Spatial CABMA PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = max(8, skip_c // 4)
        self.conv_calib = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(skip_c, k_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(k_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        g_low = self.conv_calib(skip)
        g = F.interpolate(g_low, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Novel Spatial Boundary Variants (Model 12 Family)
# ─────────────────────────────────────────────────────────────────────────────

class DGSB_PixelShuffle(nn.Module):
    """Dual-Gated Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(skip_c, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), skip_c, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        sg = self.spatial_gate(skip)
        cg = self.channel_gate(skip)
        gated_skip = skip * sg * cg
        return self.fuse(torch.cat([x_up, gated_skip], dim=1))


class SobelBoundaryPixelShuffle(nn.Module):
    """Sobel-Guided Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.gate_proj = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip_gray = skip.mean(dim=1, keepdim=True)
        gx = F.conv2d(skip_gray, self.sobel_x, padding=1)
        gy = F.conv2d(skip_gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        g = self.gate_proj(grad_mag)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class CoordSB_PixelShuffle(nn.Module):
    """Coordinate-Aware Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = max(8, skip_c // 4)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_shared = nn.Sequential(
            nn.Conv2d(skip_c, k_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(k_c),
            nn.SiLU(inplace=True)
        )
        self.conv_h = nn.Conv2d(k_c, 1, kernel_size=1)
        self.conv_w = nn.Conv2d(k_c, 1, kernel_size=1)
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        B, C, H, W = skip.shape
        x_h = self.pool_h(skip)
        x_w = self.pool_w(skip).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv_shared(y)
        y_h, y_w = torch.split(y, [H, W], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.transpose(2, 3)))
        g = a_h * a_w
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class MorphoSB_PixelShuffle(nn.Module):
    """Morphological Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dil = F.max_pool2d(skip, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-skip, kernel_size=3, stride=1, padding=1)
        g = self.gate(dil - ero)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class WaveletSB_PixelShuffle(nn.Module):
    """Wavelet High-Frequency Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_c * 3, max(8, skip_c // 4), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x00 = skip[:, :, 0::2, 0::2]
        x01 = skip[:, :, 0::2, 1::2]
        x10 = skip[:, :, 1::2, 0::2]
        x11 = skip[:, :, 1::2, 1::2]
        lh = x00 - x01 + x10 - x11
        hl = x00 + x01 - x10 - x11
        hh = x00 - x01 - x10 + x11
        hf = torch.cat([lh, hl, hh], dim=1)
        hf_up = F.interpolate(hf, size=skip.shape[2:], mode='bilinear', align_corners=False)
        g = self.gate(hf_up)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class CurvatureSB_PixelShuffle(nn.Module):
    """Curvature Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.curv_gate = nn.Sequential(
            nn.Conv2d(skip_c, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        pad = F.pad(skip, (1, 1, 1, 1), mode='replicate')
        curv_x = pad[:, :, 1:-1, 2:] + pad[:, :, 1:-1, :-2] - 2.0 * skip
        curv_y = pad[:, :, 2:, 1:-1] + pad[:, :, :-2, 1:-1] - 2.0 * skip
        curv = torch.sqrt(curv_x ** 2 + curv_y ** 2 + 1e-6)
        g = self.curv_gate(curv)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class RepOmniSB_PixelShuffle(nn.Module):
    """RepStructural Omni-Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = 8
        self.conv_k3 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=1, bias=False)
        self.conv_1x5 = nn.Conv2d(skip_c, k_c, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.conv_5x1 = nn.Conv2d(skip_c, k_c, kernel_size=(5, 1), padding=(2, 0), bias=False)
        self.bn_gate = nn.BatchNorm2d(k_c)
        self.act_gate = nn.SiLU(inplace=True)
        self.gate_out = nn.Sequential(
            nn.Conv2d(k_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        g_feat = self.conv_k3(skip) + self.conv_1x5(skip) + self.conv_5x1(skip)
        g = self.gate_out(self.act_gate(self.bn_gate(g_feat)))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class MultiDilatedSB_PixelShuffle(nn.Module):
    """Multi-Dilated Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
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

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.d1(skip) + self.d2(skip)
        g = self.gate(self.bn(feat))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class SelfCalibratedSB_PixelShuffle(nn.Module):
    """Self-Calibrated Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = max(8, skip_c // 4)
        self.calib_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(skip_c, k_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(k_c, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        g_low = self.calib_conv(skip)
        g = F.interpolate(g_low, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class CrossScaleSB_PixelShuffle(nn.Module):
    """Cross-Scale Boundary Injection Spatial Boundary PixelShuffle."""
    def __init__(self, in_c, skip_c, out_c, stem_c=20):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.stem_proj = nn.Conv2d(stem_c, 8, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_c + 8, max(8, skip_c // 4), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, skip_c // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 4), 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fuse = _make_light_fuse(mid_c, skip_c, out_c)

    def forward(self, x, skip, stem_guide=None):
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if stem_guide is not None:
            guide_down = F.adaptive_avg_pool2d(stem_guide, skip.shape[2:])
            guide_feat = self.stem_proj(guide_down)
            gate_in = torch.cat([skip, guide_feat], dim=1)
        else:
            dummy_guide = torch.zeros(skip.size(0), 8, skip.size(2), skip.size(3), device=skip.device, dtype=skip.dtype)
            gate_in = torch.cat([skip, dummy_guide], dim=1)
        g = self.gate(gate_in)
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ultra-Lightweight Encoder Blocks
# ─────────────────────────────────────────────────────────────────────────────

class MicroOmniBlock(nn.Module):
    """
    Micro Omni-ASD Block (parameter budget 200k-300k).
    Utilizes depthwise separable strip convolutions with expansion ratio 1.5.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = int(in_c * 1.5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.dw_h = nn.Conv2d(mid_c, mid_c, kernel_size=(1, 5), padding=(0, 2), groups=mid_c, bias=False)
        self.dw_w = nn.Conv2d(mid_c, mid_c, kernel_size=(5, 1), padding=(2, 0), groups=mid_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_c)
        self.act_dw = nn.SiLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.final_act = nn.SiLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv1(x)
        x = self.act_dw(self.bn_dw(self.dw_h(x) + self.dw_w(x)))
        x = self.conv2(x)
        return self.final_act(x + res)


class RepMicroStripBlock(nn.Module):
    """
    Structural Reparameterization Micro-Strip Block.
    Multi-branch 3x3, 1x5, 5x1 strip convs during training.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU(inplace=True)
        )
        self.dw_k3 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.dw_k5h = nn.Conv2d(in_c, in_c, kernel_size=(1, 5), padding=(0, 2), groups=in_c, bias=False)
        self.dw_k5w = nn.Conv2d(in_c, in_c, kernel_size=(5, 1), padding=(2, 0), groups=in_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        h = self.conv1(x)
        dw = self.dw_k3(h) + self.dw_k5h(h) + self.dw_k5w(h)
        h = self.conv2(self.act(self.bn_dw(dw)))
        return self.act(h + res)


class InvResMicroBlock(nn.Module):
    """
    Inverted Residual Micro-Strip Block.
    Expands channel width by 1.5x before applying orthogonal strip depthwise convs.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        mid_c = int(in_c * 1.5)
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.dw_h = nn.Conv2d(mid_c, mid_c, kernel_size=(1, 5), padding=(0, 2), groups=mid_c, bias=False)
        self.dw_w = nn.Conv2d(mid_c, mid_c, kernel_size=(5, 1), padding=(2, 0), groups=mid_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_c)
        self.act_dw = nn.SiLU(inplace=True)
        self.project = nn.Sequential(
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.expand(x)
        x = self.act_dw(self.bn_dw(self.dw_h(x) + self.dw_w(x)))
        x = self.project(x)
        return self.act(x + res)
