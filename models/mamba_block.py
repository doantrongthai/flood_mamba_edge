"""
Mamba State Space Model (SSM) Block with Pure PyTorch Native Fallback
Supports seamless execution across CPU, CUDA, and edge devices without requiring
compiled C++/CUDA wheels (mamba-ssm or causal-conv1d).

If mamba_ssm is installed, it automatically uses the hardware-accelerated CUDA kernel.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_MAMBA_SSM = True
except ImportError:
    OfficialMamba = None
    HAS_MAMBA_SSM = False


class PurePyTorchMamba(nn.Module):
    """
    Pure PyTorch implementation of the Mamba (S6) block.
    Guarantees compatibility across CPU and GPU on all platforms.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank is None else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, L, D)
        returns: shape (B, L, D)
        """
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Depthwise 1D conv
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # SSM projections
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B_ssm)

        # Scan across sequence
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for l in range(L):
            h = dA[:, l] * h + dB[:, l] * x_conv[:, l].unsqueeze(-1)
            y_l = torch.einsum('bdn,bn->bd', h, C_ssm[:, l])
            ys.append(y_l)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


def Mamba(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs) -> nn.Module:
    """
    Factory function returning Official Mamba if available and on CUDA,
    otherwise returning PurePyTorchMamba for native portability.
    """
    if HAS_MAMBA_SSM and torch.cuda.is_available():
        try:
            return OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
        except Exception:
            pass
    return PurePyTorchMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
