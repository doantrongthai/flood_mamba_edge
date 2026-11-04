import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aquamam_variants import (
    OmniASDConv,
    InvResOmniBlock,
    ScaleAwareMambaFusion,
    SpatialBoundaryPixelShuffle,
    DifferentiableLaplacianGuidance
)


class Model2(nn.Module):
    """
    Model 2: InvRes-Omni Super-Aqua.
    Combines the 2x inverted residual expansion of Model 4 with OmniASDConv from Model 12.
    Provides double channel capacity in depthwise strip convolutions,
    specifically boosting fine UAV flood details on AIFloodSense while maintaining high FloodKaggle mIoU.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = InvResOmniBlock(c, c * 2, expand_ratio=2)
        self.stage2 = InvResOmniBlock(c * 2, c * 4, expand_ratio=2)
        self.stage3 = InvResOmniBlock(c * 4, c * 8, expand_ratio=1)
        self.stage4 = InvResOmniBlock(c * 8, c * 8, expand_ratio=1)

        self.mamba_fusion = ScaleAwareMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, use_sgr=True, expand=1)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = SpatialBoundaryPixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = SpatialBoundaryPixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = SpatialBoundaryPixelShuffle(in_c=c, skip_c=c * 2, out_c=c)

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


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model2(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
