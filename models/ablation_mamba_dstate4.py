"""
Ablation Study: State Space Dimension d_state=4 vs d_state=8.
Journal Hypothesis: Restricting state memory to d_state=4 curtails long-range recurrence capacity
over wide geographic extents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from models.bgfmamba import (
    _scaled_bottleneck, OmniASDConv, OmniASDBR_Block,
    DifferentiableLaplacianGuidance, MultiDilatedSB_PixelShuffle,
    Mamba2D
)

class GroupedMambaFusion_DState4(nn.Module):
    def __init__(self, in_channels_list: List[int], embed_dim: int = 40, num_groups: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, embed_dim, kernel_size=1, bias=False), nn.BatchNorm2d(embed_dim))
            for c in in_channels_list
        ])
        group_dim = embed_dim // num_groups
        self.mamba_groups = nn.ModuleList([
            Mamba2D(d_model=group_dim, d_state=4, d_conv=3, expand=1)
            for _ in range(num_groups)
        ])
        self.out_projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, c, kernel_size=1, bias=False), nn.BatchNorm2d(c), nn.SiLU(inplace=True))
            for c in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        orig_shapes = [f.shape[2:] for f in features]
        target_shape = features[0].shape[2:]
        aligned = []
        for i, f in enumerate(features):
            p = self.projs[i](f)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode='bilinear', align_corners=False)
            aligned.append(p)
        fused = sum(aligned)
        chunks = torch.chunk(fused, self.num_groups, dim=1)
        mamba_outs = [self.mamba_groups[g](chunks[g]) for g in range(self.num_groups)]
        mamba_feat = torch.cat(mamba_outs, dim=1) + fused
        outs = []
        for i, target_sz in enumerate(orig_shapes):
            out_f = self.out_projs[i](mamba_feat)
            if out_f.shape[2:] != target_sz:
                out_f = F.interpolate(out_f, size=target_sz, mode='bilinear', align_corners=False)
            outs.append(out_f + features[i])
        return outs

class BGFMamba_DState4(nn.Module):
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
        self.mamba_fusion = GroupedMambaFusion_DState4([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)
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
        x0_guided = self.lap_guidance(x0)
        x1 = self.stage1(self.pool(x0_guided))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        x4 = self.stage4(self.pool(x3))
        f2, f3, f4 = self.mamba_fusion([x2, x3, x4])
        d4 = self.up4(self.dec_conv4(f4), f3)
        d3 = self.up3(self.dec_conv3(d4), f2)
        aux_out = F.interpolate(self.aux_head(d3), size=x.shape[2:], mode='bilinear', align_corners=False) if self.training else None
        d2 = self.up2(self.dec_conv2(d3), x1)
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))
        return (out, aux_out) if self.training and aux_out is not None else out

def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return BGFMamba_DState4(in_channels=in_channels, num_classes=num_classes, base_c=20, input_size=input_size)
