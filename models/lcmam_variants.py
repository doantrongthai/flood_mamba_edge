import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from models.mamba_block import Mamba
from models.lcmamnet import ConvBlock, SBA, SBAUpBlock, CDConv, CDConvBlock, CDBR_Block, TokenProjector, BiMambaPreNorm, LCR, _scaled_bottleneck


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEW COMPONENT: Asymmetric Strip-Depthwise Convolution (ASD-Conv)
# ─────────────────────────────────────────────────────────────────────────────

class ASDConv(nn.Module):
    """
    Asymmetric Strip-Depthwise Convolution (ASD-Conv).
    Replaces 4-branch CDConv with parallel orthogonal 1D strip kernels (1x7 and 7x1),
    capturing directional flood flow with 40% fewer parameters.
    """
    def __init__(self, in_c: int, out_c: int, k: int = 7):
        super().__init__()
        self.pw_in = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        self.dw_h = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, k // 2), groups=out_c, bias=False)
        self.dw_v = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(k // 2, 0), groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.pw_out = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        res = x
        strip = self.act(self.bn(self.dw_h(x) + self.dw_v(x)))
        out = self.pw_out(strip)
        return out + res


class ASDBR_Block(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.SiLU(inplace=True),
        )
        self.conv = ASDConv(bottleneck, bottleneck, k=7)
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
# 2. NEW COMPONENT: Spatial-Gated Dilated Refinement (SGR)
# ─────────────────────────────────────────────────────────────────────────────

class SGR(nn.Module):
    """
    Spatial-Gated Dilated Refinement (SGR).
    Replaces heavy dual 3x3 + 5x5 LCR with depthwise dilated conv and spatial gating.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.act(self.bn(self.dw(x)))
        g = self.gate(f)
        out = self.proj(f * g)
        return x + out


# ─────────────────────────────────────────────────────────────────────────────
# 3. NEW COMPONENT: Decoupled Multi-Scale Fusion (Stage 2, 3, 4 only)
# ─────────────────────────────────────────────────────────────────────────────

class DecoupledMambaFusion(nn.Module):
    """
    Decoupled Multi-Scale Fusion: Excludes stage 1 (64x64, 4096 tokens) from Mamba.
    Processes only stages 2, 3, 4 (1344 tokens), saving 75% of SSM token computation!
    """
    def __init__(self, in_dims: List[int], embed_dim: int = 64, use_sgr: bool = False, expand: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in in_dims
        ])
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=expand)
        self.refiners = nn.ModuleList([SGR(embed_dim) if use_sgr else LCR(embed_dim) for _ in in_dims])

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = xs[0].shape[0]
        tokens = []
        shapes = []
        for i, x in enumerate(xs):
            feat = self.projs_in[i](x)
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
# 4. NEW COMPONENT: Lite-Boundary PixelShuffle Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class LitePixelShuffleUpBlock(nn.Module):
    """
    Replaces SBAUpBlock with parameter-free PixelShuffle and lightweight channel gating.
    """
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        # in_c -> upscale 2x reduces channels by 4
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        # Shared boundary gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_c, max(8, skip_c // 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, skip_c // 8), skip_c, 1, bias=False),
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
        skip_g = skip * self.gate(skip)
        return self.fuse(torch.cat([x_up, skip_g], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LCMam_Step1 (1 Bottleneck Resolved: Decoupled Token Length)
# ─────────────────────────────────────────────────────────────────────────────

class LCMam_Step1(nn.Module):
    """
    LCMamNet Step 1: Decoupled Pyramid Tokenization.
    Excludes Stage 1 (4096 tokens) from Mamba sequence, passing only Stages 2, 3, 4 (1344 tokens).
    Retains CDConv, LCR, and SBAUpBlock.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128):
        super().__init__()
        self.inc = nn.Sequential(CDConvBlock(in_channels, base_c), CDConvBlock(base_c, base_c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = CDBR_Block(base_c, base_c * 2, bottleneck=_scaled_bottleneck(base_c * 2))
        self.stage2 = CDBR_Block(base_c * 2, base_c * 4, bottleneck=_scaled_bottleneck(base_c * 4))
        self.stage3 = CDBR_Block(base_c * 4, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))
        self.stage4 = CDBR_Block(base_c * 8, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))

        # Only stages 2, 3, 4 into Mamba! (1344 tokens instead of 5440)
        self.stageA = DecoupledMambaFusion([base_c * 4, base_c * 8, base_c * 8], embed_dim=72, use_sgr=False)
        self.stageB = DecoupledMambaFusion([base_c * 4, base_c * 8, base_c * 8], embed_dim=40, use_sgr=False)

        self.up4 = SBAUpBlock(base_c * 8 + base_c * 8, base_c * 4)
        self.up3 = SBAUpBlock(base_c * 4 + base_c * 4, base_c * 2)
        self.up2 = SBAUpBlock(base_c * 2 + base_c * 2, base_c)
        self.up1 = SBAUpBlock(base_c + base_c, base_c)
        self.outc = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.stage1(self.pool(x0))  # /2 (stays local, skip directly to decoder)
        x2 = self.stage2(self.pool(x1))  # /4
        x3 = self.stage3(self.pool(x2))  # /8
        x4 = self.stage4(self.pool(x3))  # /16

        a2, a3, a4 = self.stageA([x2, x3, x4])
        b2, b3, b4 = self.stageB([a2, a3, a4])

        d4 = self.up4(b4, b3)
        d3 = self.up3(d4, b2)
        d2 = self.up2(d3, x1)
        out_feat = self.up1(d2, x0)
        return self.outc(out_feat)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LCMam_Step2 (2 Bottlenecks: Decoupled Token + ASD-Conv)
# ─────────────────────────────────────────────────────────────────────────────

class LCMam_Step2(nn.Module):
    """
    LCMamNet Step 2: Decoupled Token + Asymmetric Strip-Depthwise Conv (ASD-Conv).
    Replaces heavy CDConv with 1D orthogonal strip convolutions (1x7 + 7x1).
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128):
        super().__init__()
        self.inc = nn.Sequential(ASDConv(in_channels, base_c), ASDConv(base_c, base_c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = ASDBR_Block(base_c, base_c * 2, bottleneck=_scaled_bottleneck(base_c * 2))
        self.stage2 = ASDBR_Block(base_c * 2, base_c * 4, bottleneck=_scaled_bottleneck(base_c * 4))
        self.stage3 = ASDBR_Block(base_c * 4, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))
        self.stage4 = ASDBR_Block(base_c * 8, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))

        self.stageA = DecoupledMambaFusion([base_c * 4, base_c * 8, base_c * 8], embed_dim=72, use_sgr=False)
        self.stageB = DecoupledMambaFusion([base_c * 4, base_c * 8, base_c * 8], embed_dim=40, use_sgr=False)

        self.up4 = SBAUpBlock(base_c * 8 + base_c * 8, base_c * 4)
        self.up3 = SBAUpBlock(base_c * 4 + base_c * 4, base_c * 2)
        self.up2 = SBAUpBlock(base_c * 2 + base_c * 2, base_c)
        self.up1 = SBAUpBlock(base_c + base_c, base_c)
        self.outc = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        a2, a3, a4 = self.stageA([x2, x3, x4])
        b2, b3, b4 = self.stageB([a2, a3, a4])

        d4 = self.up4(b4, b3)
        d3 = self.up3(d4, b2)
        d2 = self.up2(d3, x1)
        out_feat = self.up1(d2, x0)
        return self.outc(out_feat)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: LCMam_Step3 (3 Bottlenecks: Token + ASD-Conv + SGR Refinement)
# ─────────────────────────────────────────────────────────────────────────────

class LCMam_Step3(nn.Module):
    """
    LCMamNet Step 3: Decoupled Token + ASD-Conv + Spatial-Gated Dilated Refinement (SGR).
    Replaces LCR with SGR.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128):
        super().__init__()
        self.inc = nn.Sequential(ASDConv(in_channels, base_c), ASDConv(base_c, base_c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = ASDBR_Block(base_c, base_c * 2, bottleneck=_scaled_bottleneck(base_c * 2))
        self.stage2 = ASDBR_Block(base_c * 2, base_c * 4, bottleneck=_scaled_bottleneck(base_c * 4))
        self.stage3 = ASDBR_Block(base_c * 4, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))
        self.stage4 = ASDBR_Block(base_c * 8, base_c * 8, bottleneck=_scaled_bottleneck(base_c * 8))

        # use_sgr=True
        self.stageA = DecoupledMambaFusion([base_c * 4, base_c * 8, base_c * 8], embed_dim=72, use_sgr=True)
        self.stageB = DecoupledMambaFusion([base_c * 4, base_c * 8, base_c * 8], embed_dim=40, use_sgr=True)

        self.up4 = SBAUpBlock(base_c * 8 + base_c * 8, base_c * 4)
        self.up3 = SBAUpBlock(base_c * 4 + base_c * 4, base_c * 2)
        self.up2 = SBAUpBlock(base_c * 2 + base_c * 2, base_c)
        self.up1 = SBAUpBlock(base_c + base_c, base_c)
        self.outc = nn.Conv2d(base_c, num_classes, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        a2, a3, a4 = self.stageA([x2, x3, x4])
        b2, b3, b4 = self.stageB([a2, a3, a4])

        d4 = self.up4(b4, b3)
        d3 = self.up3(d4, b2)
        d2 = self.up2(d3, x1)
        out_feat = self.up1(d2, x0)
        return self.outc(out_feat)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: AquaMamNet (ALL 4 BOTTLENECKS RESOLVED SIMULTANEOUSLY)
# ─────────────────────────────────────────────────────────────────────────────

class AquaMamNet(nn.Module):
    """
    AquaMamNet: Adaptive Multi-Scale State Space Network for Edge Flood Segmentation.
    Simultaneously resolves all 4 LCMamNet bottlenecks:
    1. Decoupled Token Length (1344 tokens instead of 5440)
    2. Asymmetric Strip-Depthwise Convolutions (ASD-Conv)
    3. Spatial-Gated Dilated Refinement (SGR)
    4. Lite-Boundary PixelShuffle Decoder (No heavy SBAUpBlocks)
    5. Compact Mamba Expansion (expand=1)
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(ASDConv(in_channels, c), ASDConv(c, c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = ASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))  # /2, 64
        self.stage2 = ASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))  # /4, 128
        self.stage3 = ASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))  # /8, 256
        self.stage4 = ASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))  # /16, 256

        # Mamba with expand=1 and SGR (64 embed_dim)
        self.mamba_fusion = DecoupledMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, use_sgr=True, expand=1)

        # High-Speed PixelShuffle Decoder
        # /16 -> /8: in=256 -> ps=64, skip=256 -> out=128
        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = LitePixelShuffleUpBlock(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        # /8 -> /4: in=128 -> ps=32, skip=128 -> out=64
        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = LitePixelShuffleUpBlock(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        # /4 -> /2: in=64 -> ps=16, skip=64 -> out=32
        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = LitePixelShuffleUpBlock(in_c=c, skip_c=c * 2, out_c=c)

        # /2 -> /1
        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)        # /1, c
        x1 = self.stage1(self.pool(x0))  # /2, 2c
        x2 = self.stage2(self.pool(x1))  # /4, 4c
        x3 = self.stage3(self.pool(x2))  # /8, 8c
        x4 = self.stage4(self.pool(x3))  # /16, 8c

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3)  # /8, 4c
        d3 = self.up3(self.dec_conv3(d4), f2)  # /4, 2c
        d2 = self.up2(self.dec_conv2(d3), x1)  # /2, c

        d1 = self.up1_ps(self.up1_conv(d2))    # /1, c
        out = self.final_fuse(torch.cat([d1, x0], dim=1))
        return out
