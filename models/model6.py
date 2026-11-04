import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from models.mamba_block import Mamba
from models.lcmamnet import TokenProjector, BiMambaPreNorm, _scaled_bottleneck
from models.aquamam_variants import (
    OmniASDConv,
    OmniASDBR_Block,
    SpatialBoundaryPixelShuffle,
    DifferentiableLaplacianGuidance,
    CoordSGR
)


class CoordScaleAwareMambaFusion(nn.Module):
    """Scale-Aware Multi-Scale Mamba Fusion with Topographic Coordinate-Aware SGR Refiners"""
    def __init__(self, in_dims: List[int], embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in in_dims
        ])
        self.scale_embeddings = nn.Parameter(torch.randn(len(in_dims), embed_dim) * 0.02)
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=1)
        self.refiners = nn.ModuleList([CoordSGR(embed_dim) for _ in in_dims])

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        batch = xs[0].shape[0]
        tokens, shapes = [], []
        for i, x in enumerate(xs):
            feat = self.projs_in[i](x) + self.scale_embeddings[i].view(1, self.embed_dim, 1, 1)
            h, w = feat.shape[-2:]
            shapes.append((h, w))
            tokens.append(feat.flatten(2).transpose(1, 2))

        seq = self.mixer(torch.cat(tokens, dim=1))
        outputs = []
        cursor = 0
        for i, (h, w) in enumerate(shapes):
            length = h * w
            feat = seq[:, cursor : cursor + length, :].transpose(1, 2).reshape(batch, self.embed_dim, h, w)
            cursor += length
            feat = self.refiners[i](feat)
            out = self.projs_out[i](feat)
            outputs.append(out + xs[i])
        return outputs


class Model6(nn.Module):
    """
    Model 6: Topographic Coordinate-Aware SGR Super-Aqua (Coord-Aqua).
    Injects normalized 2D spatial coordinate coordinates into the spatial refinement gates,
    providing topographic elevation and flow direction priors to the Mamba bottleneck.
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

        self.mamba_fusion = CoordScaleAwareMambaFusion([c * 4, c * 8, c * 8], embed_dim=64)

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
    return Model6(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
