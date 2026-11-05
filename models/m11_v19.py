import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lcmam_variants import _scaled_bottleneck
from models.aquamam_variants import (
    OmniASDConv,
    OmniASDBR_Block,
    CABMA_PixelShuffle,
    SpatialBoundaryPixelShuffle,
    DifferentiableLaplacianGuidance
)
from models.ultralight_modules import (
    GroupedMambaFusion,
    FactorizedMambaFusion,
    SharedRecurrentMambaFusion,
    DecoupledDirectionalMambaFusion,
    SparseTokenMambaFusion,
    FCABMA_PixelShuffle,
    ASCABMA_PixelShuffle,
    CoordCABMA_PixelShuffle,
    MorphoCABMA_PixelShuffle,
    SobelLaplacianCABMA_PixelShuffle,
    ContrastNormPixelShuffle,
    WaveletCABMA_PixelShuffle,
    CurvatureCABMA_PixelShuffle,
    CrossScaleCABMA_PixelShuffle,
    BoundarySqueezeCABMA_PixelShuffle,
    MultiDilatedCABMA_PixelShuffle,
    SelfCalibratedCABMA_PixelShuffle,
    DGSB_PixelShuffle,
    SobelBoundaryPixelShuffle,
    CoordSB_PixelShuffle,
    MorphoSB_PixelShuffle,
    WaveletSB_PixelShuffle,
    CurvatureSB_PixelShuffle,
    RepOmniSB_PixelShuffle,
    MultiDilatedSB_PixelShuffle,
    SelfCalibratedSB_PixelShuffle,
    CrossScaleSB_PixelShuffle,
    MicroOmniBlock,
    RepMicroStripBlock,
    InvResMicroBlock
)


class Model_m11_v19(nn.Module):
    """
    M11_V19: Self-Calibrating Spatial CABMA (SC-CABMA)
    Target: Ultra-Lightweight (200k-300k parameters) for high-speed edge deployment.
    Contribution: Incorporates self-calibrated spatial convolutions in decoder stages to enrich feature representations with long-range context.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=20, input_size=128, **kwargs):
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
        self.up4 = SelfCalibratedCABMA_PixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = SelfCalibratedCABMA_PixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = SelfCalibratedCABMA_PixelShuffle(in_c=c, skip_c=c * 2, out_c=c)

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
    return Model_m11_v19(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
