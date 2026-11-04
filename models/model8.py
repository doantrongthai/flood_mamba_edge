import torch
import torch.nn as nn
from models.lcmam_variants import ASDConv, ASDBR_Block, DecoupledMambaFusion, LitePixelShuffleUpBlock, _scaled_bottleneck
from models.boundarymamba import BoundaryEnhancementModule


class Model8(nn.Module):
    """
    Model 8: Boundary Enhancement AquaMamNet (BEM-Aqua).
    Integrates BoundaryEnhancementModule at the final stage to sharply separate flood transitions.
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

        self.mamba_fusion = DecoupledMambaFusion([c * 4, c * 8, c * 8], embed_dim=64, use_sgr=True, expand=1)

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = LitePixelShuffleUpBlock(in_c=c * 4, skip_c=c * 8, out_c=c * 4)

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = LitePixelShuffleUpBlock(in_c=c * 2, skip_c=c * 4, out_c=c * 2)

        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = LitePixelShuffleUpBlock(in_c=c, skip_c=c * 2, out_c=c)

        self.up1_conv = nn.Conv2d(c, c * 4, kernel_size=1)
        self.up1_ps = nn.PixelShuffle(upscale_factor=2)

        # Boundary Enhancement Module
        self.bem = BoundaryEnhancementModule(channels=c * 2)
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
        cat_feat = self.bem(torch.cat([d1, x0], dim=1))
        out = self.final_fuse(cat_feat)
        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model8(in_channels=in_channels, num_classes=num_classes, base_c=32, input_size=input_size)
