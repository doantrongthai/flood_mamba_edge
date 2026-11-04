import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lcmam_variants import _scaled_bottleneck
from models.aquamam_variants import (
    OmniASDConv,
    OmniASDBR_Block,
    ScaleAwareMambaFusion,
    SpatialBoundaryPixelShuffle,
    DifferentiableLaplacianGuidance
)


class Model4(nn.Module):
    """
    Model 4: Dual-Stage Deep Supervision Super-Aqua.
    Features dual auxiliary prediction heads:
    1. Aux Head 1 at stage d3 (32x32, 1/4 resolution) for bottleneck gradient flow.
    2. Aux Head 2 at stage d2 (64x64, 1/2 resolution) for high-frequency boundary gradient guidance.
    Pruned at inference for zero Jetson latency overhead.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = OmniASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

        self.mamba_fusion = ScaleAwareMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, use_sgr=True, expand=1)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = SpatialBoundaryPixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = SpatialBoundaryPixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        # Dual Aux Heads
        self.aux_head3 = nn.Conv2d(c * 2, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c, num_classes, kernel_size=1)

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

        d2 = self.up2(self.dec_conv2(d3), x1)

        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))

        if self.training:
            aux3 = self.aux_head3(d3)
            aux3 = F.interpolate(aux3, size=x.shape[2:], mode='bilinear', align_corners=False)
            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux3, aux2

        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model4(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
