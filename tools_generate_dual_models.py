"""
Generator for 20 Decoupled Dual-Stream Wavelet Mamba Architectures (dual_v1.py to dual_v20.py)
Paper 1 Architecture Family: Decoupled Dual-Stream (Body & Edge Streams) with Haar Wavelet & RepStructural Convs.
Strict budget: 200,000 to 300,000 parameters.
Zero overlap with m12_v16 (No OmniASDConv, No Laplacian Stem, No OmniASDBR, No MultiDilatedSB).
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

VARIANTS = [
    {
        "id": "v1", "name": "Baseline Decoupled Dual-Stream Wavelet Mamba (DDS-Mamba)",
        "doc": "Baseline Decoupled Dual-Stream Mamba: Haar DWT stem, Rep-Structural Inverted Residual (RIR) body encoder, Strip-PPM Mamba bottleneck, dedicated Wavelet Edge Stream, and boundary-guided cross-stream interaction.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v2", "name": "Coordinate-Enhanced Decoupled Dual-Stream (Coord-DDSM)",
        "doc": "Embeds 2D normalized coordinate grid encodings into the Body Stream via Coordinate-Aware Rep-RIR blocks for topographic elevation awareness.",
        "enc_cls": "CoordRepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v3", "name": "4-Way Cross-Scan Decoupled Dual-Stream (4WCS-DDSM)",
        "doc": "Integrates 4-Way Bidirectional Cross-Scan Mamba in bottleneck (Horizontal/Vertical Forward & Backward) to capture arbitrary flood propagation paths.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "CrossScan4WayMamba(c4, embed_dim=48)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v4", "name": "Morphological Gradient Decoupled Dual-Stream (Morpho-DDSM)",
        "doc": "Incorporates Differentiable Morphological Gradient (Soft Dilation minus Soft Erosion) into the high-frequency Edge Stream to capture water boundary transitions.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "MorphoEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v5", "name": "Sobel-Laplacian Dual-Prior Boundary Stream (SL-DDSM)",
        "doc": "Combines 1st-order Sobel directional gradients and 2nd-order Laplacian differential operators directly inside the high-frequency Edge Stream.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "SobelLapEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v6", "name": "Selective Kernel Cross-Stream Interaction (SK-DDSM)",
        "doc": "Employs an adaptive dynamic Softmax channel selector between Body and Edge streams for context-dependent boundary recalibration.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "SelectiveCrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v7", "name": "Curvature Divergence Decoupled Dual-Stream (Curv-DDSM)",
        "doc": "Leverages 2D curvature tensor divergence on high-frequency wavelet subbands to delineate meandering riverbanks and irregular flood coastlines.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "CurvatureEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v8", "name": "Factorized Low-Rank Mamba Dual-Stream (LR-DDSM)",
        "doc": "Compresses Mamba SSM projection matrices via rank-constrained factorization for ultra-low computational latency on embedded edge chips.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "LowRankMamba(c4, embed_dim=48, rank=24)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 116, "c4": 148, "edge_c": 28
    },
    {
        "id": "v9", "name": "Multi-Scale Dilated Edge Stream (MD-DDSM)",
        "doc": "Applies dual parallel dilation paths (dilation 1 and 2) inside the dedicated Edge Stream to simultaneously capture micro-ripples and broad river deltas.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "MultiDilatedEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v10", "name": "Local Contrast-Normalized Dual-Stream (LCN-DDSM)",
        "doc": "Applies spatial local variance contrast normalization across high-frequency wavelet channels to cancel specular sun glare and cloud haze.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "ContrastNormEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v11", "name": "Self-Calibrated Spatial Cross-Stream Fusion (SC-DDSM)",
        "doc": "Incorporates depthwise self-calibrated spatial attention into the cross-stream fusion module to balance semantic body features and high-resolution edges.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "SelfCalibratedCrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v12", "name": "Micro-Strip Rep-RIR Dual-Stream (MStrip-DDSM)",
        "doc": "Deploys orthogonal micro-strip kernels (1x5 and 5x1) in Rep-Structural encoder blocks to enhance horizontal and vertical flood orientation sensitivity.",
        "enc_cls": "MicroStripRepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v13", "name": "Bi-Directional Recurrent Mamba Dual-Stream (Rec-DDSM)",
        "doc": "Employs a shared-state bidirectional recurrent Mamba SSM that scans both forward and flipped spatial trajectories with zero parameter expansion.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "RecurrentMamba(c4, embed_dim=48)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v14", "name": "Sparse-Token Router Mamba Dual-Stream (Sparse-DDSM)",
        "doc": "Dynamically identifies informative flood frontier tokens and filters background redundancy prior to Mamba selective state space scanning.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "SparseRouterMamba(c4, embed_dim=48)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v15", "name": "Coordinate Body + Morphological Edge Synergy (C-Morpho DDSM)",
        "doc": "Synergizes Coordinate-Aware Body encoder with Differentiable Morphological Gradient Edge Stream for topographically bounded flood delineations.",
        "enc_cls": "CoordRepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "MorphoEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v16", "name": "4-Way Mamba + Sobel-Laplacian Edge Synergy (4W-SL DDSM)",
        "doc": "Unites 4-Way Cross-Scan Mamba in bottleneck with Sobel-Laplacian dual-prior boundary extraction in the edge stream.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "CrossScan4WayMamba(c4, embed_dim=48)",
        "edge_cls": "SobelLapEdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v17", "name": "Micro-Strip Body + 4-Way Mamba Synergy (MStrip-4W DDSM)",
        "doc": "Couples Micro-Strip orthogonal depthwise encoding with 4-Way Cross-Scan Mamba for isotropic flood flow coverage.",
        "enc_cls": "MicroStripRepRIRBlock", "bottleneck_cls": "CrossScan4WayMamba(c4, embed_dim=48)",
        "edge_cls": "EdgeStream(9, edge_c)", "fuse_cls": "CrossStreamFusion(c1, edge_c)",
        "c1": 44, "c2": 76, "c3": 120, "c4": 152, "edge_c": 28
    },
    {
        "id": "v18", "name": "Morphological Edge + Selective Softmax Fusion (Morpho-SK DDSM)",
        "doc": "Combines soft morphological boundary gradients with dynamic selective softmax cross-stream interaction.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "MorphoEdgeStream(9, edge_c)", "fuse_cls": "SelectiveCrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v19", "name": "Curvature Edge + Self-Calibrated Fusion (Curv-SC DDSM)",
        "doc": "Fuses curvature divergence boundary priors with self-calibrated spatial feature modulation.",
        "enc_cls": "RepRIRBlock", "bottleneck_cls": "StripPPMMamba(c4, embed_dim=56)",
        "edge_cls": "CurvatureEdgeStream(9, edge_c)", "fuse_cls": "SelfCalibratedCrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    },
    {
        "id": "v20", "name": "Master Decoupled Dual-Stream Mamba (Master DDSM)",
        "doc": "The complete flagship synthesis for Paper 1: Coordinate Rep-RIR encoder, 4-Way Cross-Scan Mamba bottleneck, Differentiable Morphological Edge Stream, and Selective Softmax Cross-Stream Interaction under 275K params.",
        "enc_cls": "CoordRepRIRBlock", "bottleneck_cls": "CrossScan4WayMamba(c4, embed_dim=48)",
        "edge_cls": "MorphoEdgeStream(9, edge_c)", "fuse_cls": "SelectiveCrossStreamFusion(c1, edge_c)",
        "c1": 40, "c2": 72, "c3": 112, "c4": 144, "edge_c": 28
    }
]

TEMPLATE = '''import torch
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


class Model_dual_{id}(nn.Module):
    """
    Dual_{id}: {name}
    Paper 1 Family: Decoupled Dual-Stream Wavelet Mamba (DDS-Mamba)
    Target: Ultra-Lightweight (200k-300k parameters) for high-speed edge deployment.
    Contribution: {doc}
    """
    def __init__(self, in_channels=3, num_classes=1, c1={c1}, c2={c2}, c3={c3}, c4={c4}, edge_c={edge_c}, input_size=128, **kwargs):
        super().__init__()
        self.dwt = HaarDWT()

        # Body Stream (Semantic / Context)
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        self.s1 = {enc_cls}(c1, c1, stride=1)
        self.s2 = {enc_cls}(c1, c2, stride=2)
        self.s3 = {enc_cls}(c2, c3, stride=2)
        self.s4 = {enc_cls}(c3, c4, stride=2)

        # Bottleneck
        self.bottleneck = {bottleneck_cls}

        # Body Stream Reconstructor
        self.body_decoder = BodyDecoder(c4, c3, c2, c1)

        # Edge Stream (High-Frequency Details)
        self.edge_stream = {edge_cls}

        # Cross-Stream Interaction (CSFI)
        self.cross_fusion = {fuse_cls}

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
    return Model_dual_{id}(in_channels=in_channels, num_classes=num_classes, input_size=input_size)
'''

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for spec in VARIANTS:
        code = TEMPLATE.format(
            id=spec["id"],
            name=spec["name"],
            doc=spec["doc"],
            enc_cls=spec["enc_cls"],
            bottleneck_cls=spec["bottleneck_cls"],
            edge_cls=spec["edge_cls"],
            fuse_cls=spec["fuse_cls"],
            c1=spec["c1"],
            c2=spec["c2"],
            c3=spec["c3"],
            c4=spec["c4"],
            edge_c=spec["edge_c"]
        )
        filepath = os.path.join(MODELS_DIR, f"dual_{spec['id']}.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[OK] Generated: models/dual_{spec['id']}.py")

if __name__ == "__main__":
    main()
