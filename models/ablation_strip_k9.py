"""
Ablation Study: Strip Convolution Kernel Size k=5 vs k=7.
Journal Hypothesis: A smaller strip kernel (k=5) has insufficient longitudinal reach to connect
discontinuous flood paths along road corridors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bgfmamba import (
    _scaled_bottleneck, GroupedMambaFusion, MultiDilatedSB_PixelShuffle
)

class OmniASDConv_K9(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        k = 9
        self.pw_in = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()
        self.dw_h = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, k // 2), groups=out_c, bias=False)
        self.dw_v = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(k // 2, 0), groups=out_c, bias=False)
        self.dw_d = nn.Conv2d(out_c, out_c, kernel_size=3, padding=2, dilation=2, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.atten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_c, 3, kernel_size=1), nn.Softmax(dim=1))
        self.pw_out = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=1, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_in(x)
        res = x
        fh, fv, fd = self.dw_h(x), self.dw_v(x), self.dw_d(x)
        w = self.atten(x)
        fused = fh * w[:, 0:1] + fv * w[:, 1:2] + fd * w[:, 2:3]
        return self.pw_out(self.act(self.bn(fused))) + res

class OmniASDBR_Block_K9(nn.Module):
    def __init__(self, in_c: int, out_c: int, bottleneck: int):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_c, bottleneck, 1, bias=False), nn.BatchNorm2d(bottleneck), nn.SiLU(inplace=True))
        self.conv = OmniASDConv_K9(bottleneck, bottleneck)
        self.expand = nn.Sequential(nn.Conv2d(bottleneck, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.expand(self.conv(self.reduce(x))) + self.shortcut(x))

class BGFMamba_StripK9(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_c=20, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv_K9(in_channels, c), OmniASDConv_K9(c, c))
        self.pool = nn.MaxPool2d(2, 2)
        self.stage1 = OmniASDBR_Block_K9(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block_K9(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block_K9(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block_K9(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))
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
        aux_out = F.interpolate(self.aux_head(d3), size=x.shape[2:], mode='bilinear', align_corners=False) if self.training else None
        d2 = self.up2(self.dec_conv2(d3), x1)
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0], dim=1))
        return (out, aux_out) if self.training and aux_out is not None else out

def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return BGFMamba_StripK9(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
