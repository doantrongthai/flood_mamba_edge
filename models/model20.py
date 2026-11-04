import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aquamam_variants import (
    RepOmniConv,
    ScaleAwareMambaFusion,
    SpatialBoundaryPixelShuffle,
    DifferentiableLaplacianGuidance
)


class JetsonStreamBlock(nn.Module):
    """
    TensorRT-optimized Inverted Residual Strip Convolution Block.
    Multi-branch horizontal, vertical, and square depthwise convs during training.
    Collapses into a single 7x7 depthwise convolution during edge inference via switch_to_deploy().
    """
    def __init__(self, in_c: int, out_c: int, expand_ratio: int = 2):
        super().__init__()
        mid_c = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True)
        )
        self.conv = RepOmniConv(mid_c, mid_c, k=7)
        self.project = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Identity() if in_c == out_c else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.expand(x)
        x = self.conv(x)
        x = self.project(x)
        return self.act(x + res)


class Model20(nn.Module):
    """
    Model 20: Jetson Streamlined TensorRT Champion (JetsonStream-Aqua).
    Engineered specifically for peak FPS and lowest latency on Nvidia Jetson (Orin Nano/Xavier NX):
    1. Structural Reparameterization (RepOmni) in encoder: collapses to single 7x7 depthwise convs.
    2. Tensor Core FP16/FP32 aligned channels (32, 64, 128, 256, embed_dim 64).
    3. Fused PixelShuffle upsamplers (zero-copy memory transposer).
    4. Dual-Stage Deep Supervision (stages d3 and d2) for training, pruned for inference.
    5. Call model.switch_to_deploy() before ONNX/TensorRT serialization.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(RepOmniConv(in_channels, c), RepOmniConv(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = JetsonStreamBlock(c, c * 2, expand_ratio=2)
        self.stage2 = JetsonStreamBlock(c * 2, c * 4, expand_ratio=2)
        self.stage3 = JetsonStreamBlock(c * 4, c * 8, expand_ratio=1)
        self.stage4 = JetsonStreamBlock(c * 8, c * 8, expand_ratio=1)

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

    def switch_to_deploy(self):
        """Call this before exporting to ONNX / TensorRT for Nvidia Jetson"""
        for m in self.modules():
            if isinstance(m, RepOmniConv):
                m.switch_to_deploy()

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
    return Model20(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
