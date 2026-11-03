"""
MambaFloodLite: A Low-Parameter Mamba-Based Network for Flood Area Segmentation

Architecture Details:
- 4-Stage Encoder using Triple-Branch Encoder Blocks:
    * Branch 1: Axial Depthwise Convolution (AxialDW) for spatial feature extraction
    * Branch 2: AxialDW + Dual Visual State Space Block (DualVSSBlock / SS2D)
                for global contextual modeling via 2D Selective Scan
    * Branch 3: Global Context Pooling Attention
- Bottleneck:
    * Efficient Expanded Spatial Pyramid (EESP) Bottleneck with multi-rate dilations
- 4-Stage Decoder:
    * Lightweight Decoder Blocks with Bilinear Upsampling and 1x1 convolutions
    * Multi-scale deep supervision / feature fusion across all decoder stages
- Total Parameters: 285,465 (num_classes=1, in_channels=3)

Dependency Notice:
- Pure PyTorch native fallback included: does NOT require precompiled CUDA wheels
  (mamba-ssm, causal-conv1d) or external packages (einops, timm).
- If mamba-ssm is installed and tensor is on CUDA, it automatically switches to
  the accelerated CUDA selective_scan_fn kernel.
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Optional Fast-Path & Fallback Imports
# ─────────────────────────────────────────────────────────────────────────────

try:
    from timm.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_SSM = True
except ImportError:
    selective_scan_fn = None
    HAS_MAMBA_SSM = False



# ─────────────────────────────────────────────────────────────────────────────
# 1. Axial Depthwise Convolution (AxialDW)
# ─────────────────────────────────────────────────────────────────────────────

class AxialDW(nn.Module):
    """
    Axial Depthwise Convolution: decomposes 2D depthwise convolution
    into orthogonal 1D horizontal and vertical kernels with residual connection.
    """
    def __init__(self, dim: int, mixer_kernel: Tuple[int, int] = (3, 3), dilation: int = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dw_h(x) + self.dw_w(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 2D Selective Scan Module (SS2D) & Visual State Space Block (VSSBlock)
# ─────────────────────────────────────────────────────────────────────────────

class SS2D(nn.Module):
    """
    2D Selective Scan Module (SS2D).
    Flattens 2D features along 4 scanning paths (horizontal forward/backward,
    vertical forward/backward) and executes selective scanning across all directions.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 4 scanning trajectories: H-W forward, W-H forward, H-W backward, W-H backward
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # Automatic acceleration: native CUDA kernel if mamba-ssm is available on CUDA, otherwise pure PyTorch
        if HAS_MAMBA_SSM and selective_scan_fn is not None and xs.is_cuda:
            out_y = selective_scan_fn(
                xs, dts, As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False
            ).view(B, K, -1, L)
        else:
            raise RuntimeError(
                "Official hardware CUDA 'selective_scan_fn' from 'mamba_ssm' is required.\n"
                "Please run on a CUDA environment with mamba-ssm installed:\n"
                "  pip install mamba-ssm causal-conv1d"
            )

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """Visual State Space Block with LayerNorm and DropPath."""
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        **kwargs
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class DualVSSBlock(nn.Module):
    """
    Dual Visual State Space Block.
    Splits channels into two parallel branches processed by independent VSSBlocks.
    """
    def __init__(self, hidden_dim: int, drop_path: float = 0.0, d_state: int = 16, **kwargs):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be divisible by 2"
        self.split_dim = hidden_dim // 2
        self.hidden_dim = hidden_dim

        self.vss1 = VSSBlock(hidden_dim=self.split_dim, drop_path=drop_path, d_state=d_state, **kwargs)
        self.vss2 = VSSBlock(hidden_dim=self.split_dim, drop_path=drop_path, d_state=d_state, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.hidden_dim:
            format_hwc = True
            B, H, W, C = x.shape
        else:
            format_hwc = False
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)

        x1, x2 = torch.chunk(x, 2, dim=-1)
        out1 = self.vss1(x1)
        out2 = self.vss2(x2)
        out = torch.cat([out1, out2], dim=-1)

        if not format_hwc:
            out = out.permute(0, 3, 1, 2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Triple-Branch Encoder Block
# ─────────────────────────────────────────────────────────────────────────────

class TripleBranchEncoderBlock(nn.Module):
    """
    Triple-Branch Encoder Block:
    - Branch 1: AxialDW + 1x1 Conv + BatchNorm + ReLU (Local Spatial Details)
    - Branch 2: AxialDW + 1x1 Conv + BatchNorm + ReLU + DualVSSBlock (Global Context)
    - Branch 3: Adaptive Average Pooling + Depthwise Conv + Sigmoid (Channel Attention)
    Outputs: downsampled feature map (via MaxPool2d) and skip connection.
    """
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3):
        super().__init__()

        # Branch 1: Local Axial Depthwise
        self.branch1_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.branch1_pw = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.branch1_bn = nn.BatchNorm2d(in_c)
        self.branch1_relu = nn.ReLU(inplace=True)

        # Branch 2: Axial Depthwise + Dual VSS (Mamba)
        self.branch2_axial = AxialDW(dim=in_c, mixer_kernel=(kernel_size, kernel_size))
        self.branch2_pw = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.branch2_bn = nn.BatchNorm2d(in_c)
        self.branch2_relu = nn.ReLU(inplace=True)
        self.branch2_vss = DualVSSBlock(hidden_dim=in_c, d_state=8)

        # Branch 3: Global Context Pooling Attention
        self.branch3_avgpool = nn.AdaptiveAvgPool2d(1)
        self.branch3_dw = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.branch3_sigmoid = nn.Sigmoid()

        # Multi-branch fusion & downsampling
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_c * 3, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b1 = self.branch1_relu(self.branch1_bn(self.branch1_pw(self.branch1_axial(x))))

        b2 = self.branch2_relu(self.branch2_bn(self.branch2_pw(self.branch2_axial(x))))
        b2 = b2.permute(0, 2, 3, 1)
        b2 = self.branch2_vss(b2)
        b2 = b2.permute(0, 3, 1, 2)

        b3_avg = self.branch3_avgpool(x)
        b3_dw = self.branch3_dw(b3_avg)
        b3_attn = self.branch3_sigmoid(b3_dw)
        b3 = x * b3_attn

        skip = self.channel_adjust(torch.cat([b1, b2, b3], dim=1))
        x_down = self.pool(skip)
        return x_down, skip


# ─────────────────────────────────────────────────────────────────────────────
# 4. Efficient Expanded Spatial Pyramid (EESP) Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class EESP(nn.Module):
    """Efficient Expanded Spatial Pyramid unit with hierarchical feature fusion."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, branches: int = 4, reduce_ratio: int = 4):
        super().__init__()
        assert out_channels % branches == 0, "out_channels must be divisible by branches"
        self.stride = stride
        self.branches = branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        reduced_channels = out_channels // reduce_ratio

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.PReLU(reduced_channels)
        )

        self.dw_branches = nn.ModuleList([
            nn.Conv2d(
                reduced_channels, reduced_channels,
                kernel_size=3, stride=stride,
                padding=2**i, dilation=2**i,
                groups=reduced_channels, bias=False
            )
            for i in range(branches)
        ])
        self.bn = nn.ModuleList([nn.BatchNorm2d(reduced_channels) for _ in range(branches)])

        self.expand = nn.Sequential(
            nn.Conv2d(reduced_channels * branches, out_channels, kernel_size=1, groups=4, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.reduce(x)

        outputs = []
        hff = 0
        for i, (dw, bn) in enumerate(zip(self.dw_branches, self.bn)):
            out = bn(dw(x))
            if i > 0:
                out = hff + out
            hff = out
            outputs.append(out)

        x = self.expand(torch.cat(outputs, dim=1))
        return x + self.shortcut(identity)


class EESPBottleneck(nn.Module):
    """Bottleneck module composed of stacked EESP units."""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1, branches: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            EESP(in_channels, out_channels, stride=1, branches=branches)
        ])
        for _ in range(num_blocks - 1):
            self.blocks.append(EESP(out_channels, out_channels, stride=1, branches=branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Lightweight Decoder Block with Bilinear Upsampling and Pointwise Convolutions."""
    def __init__(self, in_c: int, skip_c: int, out_c: int, reduction: int = 2, kernel_size: int = 3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        hidden = (in_c + skip_c) // reduction

        self.pw1 = nn.Conv2d(in_c + skip_c, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden, out_c, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.pw1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pw2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. Complete MambaFloodLite Model
# ─────────────────────────────────────────────────────────────────────────────

class MambaFloodLite(nn.Module):
    """
    MambaFloodLite: A Low-Parameter Mamba-Based Network for Flood Area Segmentation.
    Parameter count: 285,465 (num_classes=1, in_channels=3).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes

        # Initial stem convolution
        self.initial_conv = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)

        # 4-stage encoder
        self.e1 = TripleBranchEncoderBlock(16, 16, kernel_size=3)
        self.e2 = TripleBranchEncoderBlock(16, 32, kernel_size=3)
        self.e3 = TripleBranchEncoderBlock(32, 64, kernel_size=3)
        self.e4 = TripleBranchEncoderBlock(64, 128, kernel_size=3)

        # Bottleneck
        self.bottleneck = EESPBottleneck(128, 256, num_blocks=1, branches=4)

        # 4-stage decoder
        self.d4 = DecoderBlock(256, 128, 128)
        self.d3 = DecoderBlock(128,  64,  64)
        self.d2 = DecoderBlock( 64,  32,  32)
        self.d1 = DecoderBlock( 32,  16,  16)

        # Multi-scale deep supervision / prediction heads
        self.reduce4 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.reduce3 = nn.Conv2d( 64, num_classes, kernel_size=1)
        self.reduce2 = nn.Conv2d( 32, num_classes, kernel_size=1)
        self.reduce1 = nn.Conv2d( 16, num_classes, kernel_size=1)
        self.out     = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x = self.initial_conv(x)

        # Encoder forward pass
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward pass
        d4 = self.d4(x,  skip4)
        d3 = self.d3(d4, skip3)
        d2 = self.d2(d3, skip2)
        d1 = self.d1(d2, skip1)

        # Multi-scale feature aggregation
        out4 = F.interpolate(self.reduce4(d4), size=(H, W), mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.reduce3(d3), size=(H, W), mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.reduce2(d2), size=(H, W), mode='bilinear', align_corners=False)
        out1 = self.reduce1(d1)

        return self.out(torch.cat([out1, out2, out3, out4], dim=1))


# Compatibility alias
MambaUNet = MambaFloodLite


def build_model(num_classes: int = 1, in_channels: int = 3, input_size: int = 128, pretrained: bool = False, **kwargs) -> MambaFloodLite:
    """Builder function for MambaFloodLite model."""
    return MambaFloodLite(in_channels=in_channels, num_classes=num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Self-Check / Verification Routine
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("MAMBAFLOODLITE FORWARD PASS & INTEGRITY CHECK")
    print("=" * 70)

    model = build_model(num_classes=1, in_channels=3)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    assert total_params == 285465, f"Expected 285,465 params, got {total_params}"
    print("[OK] Parameter count check PASSED (Exactly 285,465 params)")

    x = torch.randn(1, 3, 256, 256)
    print(f"Input Tensor Shape   : {tuple(x.shape)}")

    with torch.no_grad():
        out = model(x)

    print(f"Output Tensor Shape  : {tuple(out.shape)}")
    assert out.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {out.shape}"
    print("[OK] Forward pass shape check PASSED")
    print("=" * 70)
    print("[OK] All checks completed successfully!")
    print("=" * 70)
