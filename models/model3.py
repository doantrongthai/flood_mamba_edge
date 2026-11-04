import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lcmam_variants import _scaled_bottleneck
from models.aquamam_variants import (
    OmniASDConv,
    OmniASDBR_Block,
    ScaleAwareMambaFusion,
    LaplacianSpatialBoundaryPixelShuffle
)


class Model3(nn.Module):
    """
    Model 3: Direct Physical Laplacian-Coupled Decoder Super-Aqua.
    Extracts physical 2D Laplacian high-pass contours directly at the stem,
    and pipes the physical boundary map directly into every decoder stage's
    2D spatial coordinate gate.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        # Differentiable Laplacian kernel
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0],
                                   [1.0, -4.0, 1.0],
                                   [0.0, 1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('lap_kernel', lap_kernel)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = OmniASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = OmniASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = OmniASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = OmniASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

        self.mamba_fusion = ScaleAwareMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, use_sgr=True, expand=1)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = LaplacianSpatialBoundaryPixelShuffle(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = LaplacianSpatialBoundaryPixelShuffle(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = LaplacianSpatialBoundaryPixelShuffle(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def extract_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise 2D Laplacian on mean grayscale channel
        x_gray = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(x_gray, self.lap_kernel, padding=1)
        return torch.abs(lap)

    def forward(self, x: torch.Tensor):
        x0 = self.inc(x)
        lap_prior = self.extract_laplacian(x0)  # (B, 1, 128, 128) physical boundary prior

        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3, lap_prior)
        d3 = self.up3(self.dec_conv3(d4), f2, lap_prior)

        if self.training:
            aux_out = self.aux_head(d3)
            aux_out = F.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            aux_out = None

        d2 = self.up2(self.dec_conv2(d3), x1, lap_prior)
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0], dim=1))

        if self.training and aux_out is not None:
            return out, aux_out
        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model3(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
