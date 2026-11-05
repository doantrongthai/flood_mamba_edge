import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dual_stream_modules import (
    HaarDWT,
    RepRIRBlock,
    CoordRepRIRBlock,
    MicroStripRepRIRBlock,
    StripPPMMamba,
    CrossScan4WayMamba,
    RecurrentMamba,
    LowRankMamba,
    SparseRouterMamba,
    EdgeStream,
    MorphoEdgeStream,
    SobelLapEdgeStream,
    CurvatureEdgeStream,
    MultiDilatedEdgeStream,
    ContrastNormEdgeStream,
    BodyDecoder,
    CrossStreamFusion,
    SelectiveCrossStreamFusion,
    SelfCalibratedCrossStreamFusion
)


class Model_dual_v14(nn.Module):
    """
    Dual_v14: Sparse-Token Router Mamba Dual-Stream (Sparse-DDSM)
    Paper 1 Family: Decoupled Dual-Stream Wavelet Mamba (DDS-Mamba)
    Target: Ultra-Lightweight (200k-300k parameters) for high-speed edge deployment.
    Contribution: Dynamically identifies informative flood frontier tokens and filters background redundancy prior to Mamba selective state space scanning.
    """
    def __init__(self, in_channels=3, num_classes=1, c1=40, c2=72, c3=112, c4=144, edge_c=28, input_size=128, **kwargs):
        super().__init__()
        self.dwt = HaarDWT()

        # Body Stream (Semantic / Context)
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        self.s1 = RepRIRBlock(c1, c1, stride=1)
        self.s2 = RepRIRBlock(c1, c2, stride=2)
        self.s3 = RepRIRBlock(c2, c3, stride=2)
        self.s4 = RepRIRBlock(c3, c4, stride=2)

        # Bottleneck
        self.bottleneck = SparseRouterMamba(c4, embed_dim=48)

        # Body Stream Reconstructor
        self.body_decoder = BodyDecoder(c4, c3, c2, c1)

        # Edge Stream (High-Frequency Details)
        self.edge_stream = EdgeStream(9, edge_c)

        # Cross-Stream Interaction (CSFI)
        self.cross_fusion = CrossStreamFusion(c1, edge_c)

        # Final Sub-Pixel Prediction Head
        self.final_head = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        ll, high = self.dwt(x)

        # Body Stream Forward
        x0 = self.inc(ll)
        x1 = self.s1(x0)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        b4 = self.bottleneck(x4)
        body_feat = self.body_decoder(b4, x3, x2, x1)

        # Edge Stream Forward
        edge_feat, edge_logits = self.edge_stream(high)

        # Cross-Stream Fusion
        fused = self.cross_fusion(body_feat, edge_feat)
        out = self.final_head(fused)

        if self.training:
            aux_out = F.interpolate(edge_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_out

        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return Model_dual_v14(in_channels=in_channels, num_classes=num_classes, input_size=input_size)
