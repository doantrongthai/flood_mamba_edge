"""
Generator script to build and verify all 40 ultra-lightweight models:
- 20 variants for Model 11: m11_v1.py to m11_v20.py
- 20 variants for Model 12: m12_v1.py to m12_v20.py

Strict budget: strictly 200,000 to 300,000 parameters.
Paper contributions: Novel edge/boundary/SSM mechanisms.
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Template for Model 11 variants
M11_SPECS = [
    {
        "id": "v1", "name": "Grouped-Mamba CABMA",
        "doc": "Splits multi-scale tokens into 2 parallel SSM scan groups, cutting SSM latency while retaining cross-scale context.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, num_groups=2)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v2", "name": "Frequency-Gated CABMA (F-CABMA)",
        "doc": "Integrates dynamic high-pass frequency gating into CABMA decoder to isolate boundary gradients.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "FCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v3", "name": "Asymmetric Strip CABMA (AS-CABMA)",
        "doc": "Replaces isotropic convs with orthogonal (1x5 and 5x1) strip convs to capture elongated flood contours.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "ASCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v4", "name": "Coordinate-Aware CABMA (Coord-CABMA)",
        "doc": "Embeds explicit 2D spatial coordinate positional encodings into CABMA to preserve spatial topography.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CoordCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v5", "name": "Morphological Difference CABMA (Morpho-CABMA)",
        "doc": "Uses differentiable morphological gradient operations (soft max minus min pooling) to guide boundary modulation.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "MorphoCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v6", "name": "Factorized Low-Rank Mamba CABMA (FLR-CABMA)",
        "doc": "Employs low-rank channel bottlenecking prior to selective state-space scanning in Mamba fusion.",
        "fusion": "FactorizedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, rank=24)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v7", "name": "Sobel-Laplacian Dual-Prior CABMA (SL-CABMA)",
        "doc": "Combines multi-directional Sobel 1st-order and Laplacian 2nd-order differential operators for edge guidance.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, num_groups=2)",
        "up_cls": "SobelLaplacianCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v8", "name": "Contrast-Normalized CABMA (CN-CABMA)",
        "doc": "Applies local contrast normalization and dynamic variance scaling inside CABMA to handle reflections and shadows.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "ContrastNormPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v9", "name": "RepStructural Micro-Strip CABMA (RMS-CABMA)",
        "doc": "Multi-branch reparameterizable strip convolutions in encoder that fold into single depthwise convs at inference.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "RepMicroStripBlock", "base_c": 20
    },
    {
        "id": "v10", "name": "Wavelet Subband Boundary CABMA (Wavelet-CABMA)",
        "doc": "Decomposes multi-scale skip connections into 2D Haar wavelet subbands to isolate high-frequency flood perimeters.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "WaveletCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v11", "name": "Dynamic Sparse-Token Mamba CABMA (Sparse-CABMA)",
        "doc": "Computes spatial boundary saliency and selects only top informative tokens for Mamba sequence modeling.",
        "fusion": "SparseTokenMambaFusion([c * 4, c * 8, c * 8], embed_dim=40)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v12", "name": "Curvature-Modulated CABMA (Curv-CABMA)",
        "doc": "Computes second-order surface normal curvature tensor to delineate subtle water ripples from submerged boundaries.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CurvatureCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v13", "name": "Cross-Scale Boundary Injection CABMA (CSBI-CABMA)",
        "doc": "Injects high-resolution stem boundary representations directly into all decoder levels via lightweight residual bypasses.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CrossScaleCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20, "has_stem_guide": True
    },
    {
        "id": "v14", "name": "Inverted Residual Omni-CABMA (InvOmni-CABMA)",
        "doc": "Replaces bottleneck blocks in encoder with expanded inverted-residual strip convs for richer non-linear capacity.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=32, num_groups=2)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "InvResMicroBlock", "base_c": 20
    },
    {
        "id": "v15", "name": "Shared-State Recurrent Mamba CABMA (SSR-CABMA)",
        "doc": "Shares the core Mamba SSM weight matrices across multiple pyramid scales in bottleneck fusion.",
        "fusion": "SharedRecurrentMambaFusion([c * 4, c * 8, c * 8], embed_dim=40)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v16", "name": "Directional Dual-Scan CABMA (DDS-CABMA)",
        "doc": "Conducts orthogonal horizontal and vertical bidirectional scans in Mamba to capture horizontal flood lines and vertical structures.",
        "fusion": "DecoupledDirectionalMambaFusion([c * 4, c * 8, c * 8], embed_dim=40)",
        "up_cls": "CABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v17", "name": "Boundary Squeeze-and-Amplification CABMA (BSA-CABMA)",
        "doc": "Implements a boundary-squeezed attention gate that suppresses interior water pixels while amplifying high-entropy transitions.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "BoundarySqueezeCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v18", "name": "Multi-Dilated Receptive CABMA (MDR-CABMA)",
        "doc": "Employs cascaded dilated strip depthwise convolutions (rates 1, 2, 4) in CABMA skip projection.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, num_groups=2)",
        "up_cls": "MultiDilatedCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v19", "name": "Self-Calibrating Spatial CABMA (SC-CABMA)",
        "doc": "Incorporates self-calibrated spatial convolutions in decoder stages to enrich feature representations with long-range context.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "SelfCalibratedCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v20", "name": "Edge-Adaptive Master CABMA (EAM-CABMA)",
        "doc": "Unified synthesis uniting Grouped Mamba Fusion, Dual Sobel-Laplacian Boundary Guidance, and RepStructural Micro-Convs.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=56, num_groups=2)",
        "up_cls": "SobelLaplacianCABMA_PixelShuffle",
        "enc_block": "RepMicroStripBlock", "base_c": 20
    }
]

# Template for Model 12 variants
M12_SPECS = [
    {
        "id": "v1", "name": "Grouped-Mamba Super-Aqua",
        "doc": "Splits multi-scale tokens into grouped parallel SSM scans with Lite Deep Supervision at stage d3.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, num_groups=2)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v2", "name": "Dual-Gated Spatial Boundary Super-Aqua (DGSB-Aqua)",
        "doc": "Introduces dual spatial-and-channel gating mechanism in the decoder to independently calibrate semantic class and spatial sharpness.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "DGSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v3", "name": "Sobel-Gradient Spatial Boundary Super-Aqua (SGSB-Aqua)",
        "doc": "Directly integrates directional Sobel filters into the spatial boundary gate of the decoder.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, num_groups=2)",
        "up_cls": "SobelBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v4", "name": "Asymmetric Strip Super-Aqua (AS-Aqua)",
        "doc": "Enhances the spatial boundary decoder with cross-shaped strip convolutions (1x5, 5x1) to align with elongated water bodies.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "ASCABMA_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v5", "name": "Factorized Low-Rank Mamba Super-Aqua (FLR-Aqua)",
        "doc": "Applies low-rank linear bottlenecking before selective state space scans in multi-scale fusion.",
        "fusion": "FactorizedMambaFusion([c * 4, c * 8, c * 8], embed_dim=48, rank=24)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v6", "name": "Coordinate Spatial Boundary Super-Aqua (CoordSB-Aqua)",
        "doc": "Embeds coordinate attention (X-axis and Y-axis 1D poolings) into the spatial boundary gate to retain exact spatial coordinates.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CoordSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v7", "name": "Contrast-Normalized Spatial Boundary Super-Aqua (CNSB-Aqua)",
        "doc": "Normalizes local feature contrast in the spatial gate to prevent illumination shifts and reflection artifacts from blurring boundaries.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "ContrastNormPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v8", "name": "RepStructural Omni Super-Aqua (RepOmni-Aqua)",
        "doc": "Uses reparameterizable multi-branch convs in the spatial boundary gate that fold into single 3x3 convs during inference.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "RepOmniSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v9", "name": "Morphological Spatial Boundary Super-Aqua (MSB-Aqua)",
        "doc": "Applies differentiable morphological boundary approximations to sharpen the spatial boundary mask before PixelShuffle upsampling.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "MorphoSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v10", "name": "Wavelet Spatial Boundary Super-Aqua (WaveletSB-Aqua)",
        "doc": "Decomposes spatial boundary features into Haar wavelet subbands to isolate high-frequency flood perimeters.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "WaveletSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v11", "name": "Curvature Spatial Boundary Super-Aqua (CurvSB-Aqua)",
        "doc": "Uses differential surface curvature tensors to preserve curved shorelines and rivers.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CurvatureSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v12", "name": "Dynamic Sparse-Token Mamba Super-Aqua (Sparse-Aqua)",
        "doc": "Dynamic boundary saliency token selection for Mamba SSM, decreasing latency while focusing sequence capacity on edge regions.",
        "fusion": "SparseTokenMambaFusion([c * 4, c * 8, c * 8], embed_dim=40)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v13", "name": "Cross-Scale Boundary Injection Super-Aqua (CSBI-Aqua)",
        "doc": "Direct high-to-low boundary feature bypass connecting stem Laplacian features to all spatial boundary decoder stages.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "CrossScaleSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20, "has_stem_guide": True
    },
    {
        "id": "v14", "name": "Shared-State Recurrent Mamba Super-Aqua (SSR-Aqua)",
        "doc": "Parameter-shared recurrent Mamba SSM scanning across all pyramid levels, minimizing parameter redundancy.",
        "fusion": "SharedRecurrentMambaFusion([c * 4, c * 8, c * 8], embed_dim=40)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v15", "name": "Decoupled Directional Mamba Super-Aqua (DDM-Aqua)",
        "doc": "Decoupled orthogonal horizontal/vertical 1D Mamba scanning in bottleneck fusion.",
        "fusion": "DecoupledDirectionalMambaFusion([c * 4, c * 8, c * 8], embed_dim=40)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v16", "name": "Multi-Dilated Spatial Boundary Super-Aqua (MDSB-Aqua)",
        "doc": "Multi-scale dilated convolutions in spatial boundary gate to span varied flood frontiers.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "MultiDilatedSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v17", "name": "Self-Calibrated Spatial Boundary Super-Aqua (SCSB-Aqua)",
        "doc": "Self-calibrating spatial feature transformation in spatial boundary decoder for expansive receptive field.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "SelfCalibratedSB_PixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20
    },
    {
        "id": "v18", "name": "Inverted Residual Omni Super-Aqua (InvOmni-Aqua)",
        "doc": "Inverted residual micro-strip blocks in encoder stages for high representational capacity with low parameter footprint.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=32, num_groups=2)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "InvResMicroBlock", "base_c": 20
    },
    {
        "id": "v19", "name": "Dual-Supervised Spatial Boundary Super-Aqua (DSSB-Aqua)",
        "doc": "Dual deep supervision heads at d3 (/4) and d2 (/2) scales for enriched gradient propagation.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "SpatialBoundaryPixelShuffle",
        "enc_block": "OmniASDBR_Block", "base_c": 20, "dual_aux": True
    },
    {
        "id": "v20", "name": "Master Edge-Optimized Super-Aqua (MEA-Aqua)",
        "doc": "The complete unified ultra-lightweight synthesis: RepMicroStrip encoder, Grouped Mamba Fusion, and DGSB decoder under 300K params.",
        "fusion": "GroupedMambaFusion([c * 4, c * 8, c * 8], embed_dim=40, num_groups=2)",
        "up_cls": "DGSB_PixelShuffle",
        "enc_block": "RepMicroStripBlock", "base_c": 20
    }
]


def generate_code(spec, family="m11"):
    vid = spec["id"]
    name = spec["name"]
    doc = spec["doc"]
    fusion = spec["fusion"]
    up_cls = spec["up_cls"]
    enc_block = spec["enc_block"]
    base_c = spec.get("base_c", 20)
    has_stem_guide = spec.get("has_stem_guide", False)
    dual_aux = spec.get("dual_aux", False)

    cls_name = f"Model_{family}_{vid}"

    # Build encoder stage instantiation
    if enc_block == "OmniASDBR_Block":
        s1 = "OmniASDBR_Block(c, c * 2, bottleneck=_scaled_bottleneck(c * 2))"
        s2 = "OmniASDBR_Block(c * 2, c * 4, bottleneck=_scaled_bottleneck(c * 4))"
        s3 = "OmniASDBR_Block(c * 4, c * 8, bottleneck=_scaled_bottleneck(c * 8))"
        s4 = "OmniASDBR_Block(c * 8, c * 8, bottleneck=_scaled_bottleneck(c * 8))"
    else:
        s1 = f"{enc_block}(c, c * 2)"
        s2 = f"{enc_block}(c * 2, c * 4)"
        s3 = f"{enc_block}(c * 4, c * 8)"
        s4 = f"{enc_block}(c * 8, c * 8)"

    # Build decoder up blocks instantiation
    if has_stem_guide:
        up4_init = f"{up_cls}(in_c=c * 4, skip_c=c * 8, out_c=c * 4, stem_c=c)"
        up3_init = f"{up_cls}(in_c=c * 2, skip_c=c * 4, out_c=c * 2, stem_c=c)"
        up2_init = f"{up_cls}(in_c=c, skip_c=c * 2, out_c=c, stem_c=c)"
        up4_call = "self.up4(self.dec_conv4(f4), f3, x0_guided)"
        up3_call = "self.up3(self.dec_conv3(d4), f2, x0_guided)"
        up2_call = "self.up2(self.dec_conv2(d3), x1, x0_guided)"
    else:
        up4_init = f"{up_cls}(in_c=c * 4, skip_c=c * 8, out_c=c * 4)"
        up3_init = f"{up_cls}(in_c=c * 2, skip_c=c * 4, out_c=c * 2)"
        up2_init = f"{up_cls}(in_c=c, skip_c=c * 2, out_c=c)"
        up4_call = "self.up4(self.dec_conv4(f4), f3)"
        up3_call = "self.up3(self.dec_conv3(d4), f2)"
        up2_call = "self.up2(self.dec_conv2(d3), x1)"

    dual_aux_init = "        self.aux_head2 = nn.Conv2d(c, num_classes, kernel_size=1)\n" if dual_aux else ""
    dual_aux_forward = """
        if self.training:
            aux_out2 = self.aux_head2(d2)
            aux_out2 = F.interpolate(aux_out2, size=x.shape[2:], mode='bilinear', align_corners=False)
""" if dual_aux else ""
    dual_aux_return = "            return out, (aux_out + aux_out2) * 0.5\n" if dual_aux else "            return out, aux_out\n"

    code = f'''import torch
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


class {cls_name}(nn.Module):
    """
    {family.upper()}_{vid.upper()}: {name}
    Target: Ultra-Lightweight (200k-300k parameters) for high-speed edge deployment.
    Contribution: {doc}
    """
    def __init__(self, in_channels=3, num_classes=1, base_c={base_c}, input_size=128, **kwargs):
        super().__init__()
        c = base_c
        self.inc = nn.Sequential(OmniASDConv(in_channels, c), OmniASDConv(c, c))
        self.lap_guidance = DifferentiableLaplacianGuidance(in_channels=c)
        self.pool = nn.MaxPool2d(2, 2)

        self.stage1 = {s1}
        self.stage2 = {s2}
        self.stage3 = {s3}
        self.stage4 = {s4}

        self.mamba_fusion = {fusion}

        self.dec_conv4 = nn.Conv2d(c * 8, c * 4, kernel_size=1)
        self.up4 = {up4_init}

        self.dec_conv3 = nn.Conv2d(c * 4, c * 2, kernel_size=1)
        self.up3 = {up3_init}

        self.aux_head = nn.Conv2d(c * 2, num_classes, kernel_size=1)
{dual_aux_init}
        self.dec_conv2 = nn.Conv2d(c * 2, c, kernel_size=1)
        self.up2 = {up2_init}

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

        d4 = {up4_call}
        d3 = {up3_call}

        if self.training:
            aux_out = self.aux_head(d3)
            aux_out = F.interpolate(aux_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        else:
            aux_out = None

        d2 = {up2_call}
{dual_aux_forward}
        d1 = self.up1_ps(self.up1_conv(d2))
        out = self.final_fuse(torch.cat([d1, x0_guided], dim=1))

        if self.training and aux_out is not None:
{dual_aux_return}
        return out


def build_model(num_classes=1, input_size=128, pretrained=False, in_channels=3, **kwargs):
    return {cls_name}(in_channels=in_channels, num_classes=num_classes, base_c={base_c}, input_size=input_size)
'''
    return code


def main():
    print("Generating 20 Model 11 variants and 20 Model 12 variants...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Generate m11_v1 to m11_v20
    for spec in M11_SPECS:
        filename = f"m11_{spec['id']}.py"
        path = os.path.join(MODELS_DIR, filename)
        code = generate_code(spec, family="m11")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  + Generated: {filename}")

    # 2. Generate m12_v1 to m12_v20
    for spec in M12_SPECS:
        filename = f"m12_{spec['id']}.py"
        path = os.path.join(MODELS_DIR, filename)
        code = generate_code(spec, family="m12")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  + Generated: {filename}")

    print("\nAll 40 model files successfully generated!")


if __name__ == "__main__":
    main()
