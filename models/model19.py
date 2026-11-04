import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aquamam_variants import (
    OmniASDConv,
    InvResOmniBlock,
    ScaleAwareMambaFusion,
    LaplacianSpatialBoundaryPixelShuffle,
    MorphoBoundaryRefiner
)


class Model19(nn.Module):
    """
    Model 19: All-In-One Unified Champion Super-Aqua (Unified Super-Aqua).
    Integrates the peak architectural innovations:
    1. InvRes-Omni Encoder (2x channel expansion + Omni-directional strip convolutions)
    2. Direct Physical Laplacian Boundary Prior coupled into all decoder gates
    3. Scale-Aware Multi-Scale Mamba Fusion with Multi-Dilated SGR
    4. Dual-Stage Deep Supervision (aux heads at stages d3 and d2)
    5. Soft Morphological Transition Zone Boundary Refiner at final stage.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0],
                                   [1.0, -4.0, 1.0],
                                   [0.0, 1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('lap_kernel', lap_kernel)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = InvResOmniBlock(c, c * 2, expand_ratio=2)
        self.stage2 = InvResOmniBlock(c * 2, c * 4, expand_ratio=2)
        self.stage3 = InvResOmniBlock(c * 4, c * 8, expand_ratio=1)
        self.stage4 = InvResOmniBlock(c * 8, c * 8, expand_ratio=1)

        self.mamba_fusion = ScaleAwareMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, use_sgr=True, expand=1)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = LaplacianSpatialBoundaryPixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = LaplacianSpatialBoundaryPixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        # Dual Aux Heads
        self.aux_head3 = nn.Conv2d(c * 2, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = LaplacianSpatialBoundaryPixelShuffle(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)

        self.morpho_refiner = MorphoBoundaryRefiner(in_c=c)

        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def extract_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        x_gray = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(x_gray, self.lap_kernel, padding=1)
        return torch.abs(lap)

    def forward(self, x: torch.Tensor):
        x0 = self.inc(x)
        lap_prior = self.extract_laplacian(x0)

        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3, lap_prior)
        d3 = self.up3(self.dec_conv3(d4), f2, lap_prior)

        d2 = self.up2(self.dec_conv2(d3), x1, lap_prior)
        d1 = self.up1_ps(self.up1_conv(d2))

        fused = self.final_fuse(torch.cat([d1, x0], dim=1))
        fused = self.morpho_refiner(fused)
        out = self.head(fused)

        if self.training:
            aux3 = self.aux_head3(d3)
            aux3 = F.interpolate(aux3, size=x.shape[2:], mode='bilinear', align_corners=False)
            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux3, aux2

        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model19(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
