import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class Mamba2D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: int = 1):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required")
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_norm = self.norm(x_flat)
        y_fwd = self.mamba_fwd(x_norm)
        y_bwd = self.mamba_bwd(torch.flip(x_norm, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        out = (y_fwd + y_bwd) * 0.5
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x + out


def _scaled_bottleneck(out_c: int) -> int:
    bottleneck = int(round(out_c * 0.625 / 8.0) * 8)
    return max(16, bottleneck)


class OmniASDConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3):
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
        w = self.atten(x)
        h = self.dw_h(x)
        v = self.dw_v(x)
        d = self.dw_d(x)
        fuse = w[:, 0:1] * h + w[:, 1:2] * v + w[:, 2:3] * d
        return x + self.pw_out(self.act(self.bn(fuse)))


class OmniASDBR_Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: Optional[int] = None, k: int = 3):
        super().__init__()
        mid = bottleneck if bottleneck is not None else _scaled_bottleneck(out_c)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        self.conv1 = OmniASDConv(mid, mid, k=k)
        self.conv2 = OmniASDConv(mid, mid, k=k)
        self.expand = nn.Sequential(
            nn.Conv2d(mid, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        y = self.reduce(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.expand(y)
        return self.act(res + y)


def _make_light_fuse(in_c: int, skip_c: Optional[int] = None, out_c: Optional[int] = None) -> nn.Sequential:
    if skip_c is None and out_c is None:
        dim = in_c
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )
    return nn.Sequential(
        nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU(inplace=True)
    )


class GroupedMambaFusion(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim: int = 40, num_groups: int = 2):
        super().__init__()
        self.num_groups = num_groups
        self.embed_dim = embed_dim
        group_dim = embed_dim // num_groups
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.SiLU(inplace=True)
            ) for c in in_channels
        ])
        self.mamba_groups = nn.ModuleList([
            Mamba2D(d_model=group_dim, d_state=8, d_conv=3, expand=1)
            for _ in range(num_groups)
        ])
        self.post_fuse = _make_light_fuse(embed_dim)
        self.outs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c)
            ) for c in in_channels
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        target_size = feats[-1].shape[2:]
        proj_feats = []
        for i, (f, p) in enumerate(zip(feats, self.projs)):
            pf = p(f)
            if pf.shape[2:] != target_size:
                pf = F.interpolate(pf, size=target_size, mode="bilinear", align_corners=False)
            proj_feats.append(pf)
        fused = sum(proj_feats)
        chunks = torch.chunk(fused, self.num_groups, dim=1)
        mamba_chunks = [m(c) for m, c in zip(self.mamba_groups, chunks)]
        fused = torch.cat(mamba_chunks, dim=1)
        fused = self.post_fuse(fused)
        outputs = []
        for orig_f, out_proj in zip(feats, self.outs):
            out = out_proj(fused)
            if out.shape[2:] != orig_f.shape[2:]:
                out = F.interpolate(out, size=orig_f.shape[2:], mode="bilinear", align_corners=False)
            outputs.append(orig_f + out)
        return outputs


class MultiDilatedSB_PixelShuffle(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up_conv = nn.Conv2d(in_c, in_c * 4, kernel_size=1)
        self.ps = nn.PixelShuffle(upscale_factor=2)
        mid = max(8, skip_c // 4)
        self.d1 = nn.Conv2d(skip_c, mid, kernel_size=3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(skip_c, mid, kernel_size=3, padding=2, dilation=2, bias=False)
        self.d3 = nn.Conv2d(skip_c, mid, kernel_size=3, padding=3, dilation=3, bias=False)
        self.bn = nn.BatchNorm2d(mid * 3)
        self.gate = nn.Sequential(
            nn.Conv2d(mid * 3, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(self.up_conv(x))
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        feat = torch.cat([self.d1(skip), self.d2(skip), self.d3(skip)], dim=1)
        g = self.gate(self.bn(feat))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))


class BGFMamba(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 20, input_size: int = 128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
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
        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3)
        d3 = self.up3(self.dec_conv3(d4), f2)

        aux_out = F.interpolate(self.aux_head(d3), size=x.shape[2:], mode="bilinear", align_corners=False) if self.training else None

        d2 = self.up2(self.dec_conv2(d3), x1)
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0], dim=1))

        if self.training and aux_out is not None:
            return out, aux_out
        return out


MambaEdge = BGFMamba


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return BGFMamba(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)


__all__ = [
    'BGFMamba',
    'MambaEdge',
    'build_model',
    'Mamba2D',
    'GroupedMambaFusion',
    'OmniASDConv',
    'OmniASDBR_Block',
    'MultiDilatedSB_PixelShuffle',
    '_scaled_bottleneck',
    '_make_light_fuse',
]
