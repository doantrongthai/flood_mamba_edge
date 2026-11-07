"""
Ablation Study: Activation Function ReLU vs SiLU (Swish).
Journal Hypothesis: SiLU maintains smooth first and second derivatives that improve gradient flow
through deep recurrent state models and sharp edge boundary filters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bgfmamba import (
    _scaled_bottleneck, DifferentiableLaplacianGuidance,
    GroupedMambaFusion
)

class OmniASDConv_GELU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 7):
        super().__init__()
        self.pw_in = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        self.dw_h = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, k // 2), groups=out_c, bias=False)
        self.dw_v = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(k // 2, 0), groups=out_c, bias=False)
        self.dw_d = nn.Conv2d(out_c, out_c, kernel_size=3, padding=2, dilation=2, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.atten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_c, 3, kernel_size=1), nn.Softmax(dim=1))
        self.pw_out = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=1, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        res = x
        fh, fv, fd = self.dw_h(x), self.dw_v(x), self.dw_d(x)
        w = self.atten(x)
        fused = fh * w[:, 0:1] + fv * w[:, 1:2] + fd * w[:, 2:3]
        return self.pw_out(self.act(self.bn(fused))) + res

class OmniASDBR_Block_GELU(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: int):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_c, bottleneck, 1, bias=False), nn.BatchNorm2d(bottleneck), nn.GELU())
        self.conv = OmniASDConv_GELU(bottleneck, bottleneck, k=7)
        self.expand = nn.Sequential(nn.Conv2d(bottleneck, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.expand(self.conv(self.reduce(x))) + self.shortcut(x))

class MultiDilatedSB_PixelShuffle_GELU(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        mid_c = in_c // 4
        self.ps = nn.PixelShuffle(upscale_factor=2)
        k_c = 8
        self.d1 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=1, bias=False)
        self.d2 = nn.Conv2d(skip_c, k_c, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(k_c)
        self.gate = nn.Sequential(nn.GELU(), nn.Conv2d(k_c, 1, kernel_size=1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_c + skip_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.ps(x)
        if x_up.shape[2:] != skip.shape[2:]:
            x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.d1(skip) + self.d2(skip)
        g = self.gate(self.bn(feat))
        return self.fuse(torch.cat([x_up, skip * (1.0 + g)], dim=1))

class BGFMamba_ActGELU(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_c=20, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv_GELU(in_channels, c), OmniASDConv_GELU(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)
        self.stage1 = OmniASDBR_Block_GELU(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block_GELU(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block_GELU(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block_GELU(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.mamba_fusion = GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)
        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = MultiDilatedSB_PixelShuffle_GELU(in_c=c * 4, skip_c=c * 8, out_c=c * 4)
        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = MultiDilatedSB_PixelShuffle_GELU(in_c=c * 2, skip_c=c * 4, out_c=c * 2)
        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)
        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = MultiDilatedSB_PixelShuffle_GELU(in_c=c, skip_c=c * 2, out_c=c)
        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
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
        aux_out = F.interpolate(self.aux_head(d3), size=x.shape[2:], mode='bilinear', align_corners=False) if self.training else None
        d2 = self.up2(self.dec_conv2(d3), x1)
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))
        return (out, aux_out) if self.training and aux_out is not None else out

def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return BGFMamba_ActGELU(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
