import torch
import torch.nn as nn
from typing import List
from models.mamba_block import Mamba
from models.lcmamnet import TokenProjector, BiMambaPreNorm
from models.lcmam_variants import ASDConv, ASDBR_Block, LitePixelShuffleUpBlock, _scaled_bottleneck
from models.aquamam_variants import MultiDilatedSGR


class MultiDilMambaFusion(nn.Module):
    def __init__(self, in_dims: List[int], embed_dim: int = 64, expand: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projs_in = nn.ModuleList([TokenProjector(c, embed_dim) for c in in_dims])
        self.projs_out = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in in_dims
        ])
        self.mixer = BiMambaPreNorm(embed_dim, gamma_init=0.1, expand=expand)
        self.refiners = nn.ModuleList([MultiDilatedSGR(embed_dim) for _ in in_dims])

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


class Model6(nn.Module):
    """
    Model 6: Multi-Dilated AquaMamNet (MultiDil-Aqua).
    Upgrades SGR with multi-dilation branches (1, 2, 4) for variable flood expanses.
    """
    def __init__(self, in_channels=3, num_classes=1, base_c=32, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(ASDConv(in_channels, c), ASDConv(c, c))
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = ASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))
        self.stage2 = ASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))
        self.stage3 = ASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))
        self.stage4 = ASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))

        self.mamba_fusion = MultiDilMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, expand=1)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = LitePixelShuffleUpBlock(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = LitePixelShuffleUpBlock(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = LitePixelShuffleUpBlock(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)
        self.final_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.stage1(self.pool(x0))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))

        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])

        d4 = self.up4(self.dec_conv4(f4), f3)
        d3 = self.up3(self.dec_conv3(d4), f2)
        d2 = self.up2(self.dec_conv2(d3), x1)

        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0], dim=1))
        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model6(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
