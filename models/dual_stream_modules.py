import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ultralight_modules import Mamba2D


# ─────────────────────────────────────────────────────────────────────────────
# 1. Parameter-Free Haar 2D Discrete Wavelet Transform
# ─────────────────────────────────────────────────────────────────────────────
class HaarDWT(nn.Module):
    """
    2D Discrete Wavelet Transform using 2D Haar wavelets.
    Losslessly decomposes (B, C, H, W) into:
      - Low-frequency approximation LL: (B, C, H//2, W//2)
      - High-frequency details LH, HL, HH: (B, C*3, H//2, W//2)
    Zero parameters, exact energy preservation, lossless boundary capture.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 - x3 + x2 + x4
        hl = -x1 + x3 - x2 + x4
        hh = x1 - x3 - x2 + x4
        high = torch.cat([lh, hl, hh], dim=1)
        return ll, high


# ─────────────────────────────────────────────────────────────────────────────
# 2. Body Stream Encoder Blocks (Rep-Structural Inverted Residual)
# ─────────────────────────────────────────────────────────────────────────────
class RepRIRBlock(nn.Module):
    """
    Rep-Structural Inverted Residual Mobile Block.
    Features:
      - Pointwise channel expansion
      - Multi-branch depthwise training (3x3 DW + 1x1 DW) with algebraic reparameterization
      - Squeeze-and-Excitation channel modulation
      - Pointwise linear projection with residual shortcut
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        hidden_c = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 1, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True)
        )
        self.dw3 = nn.Conv2d(hidden_c, hidden_c, 3, stride=stride, padding=1, groups=hidden_c, bias=False)
        self.dw1 = nn.Conv2d(hidden_c, hidden_c, 1, stride=stride, padding=0, groups=hidden_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden_c)
        self.act_dw = nn.SiLU(inplace=True)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_c, max(4, hidden_c // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(4, hidden_c // 4), hidden_c, 1),
            nn.Sigmoid()
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        dw = self.dw3(h) + self.dw1(h)
        h = self.act_dw(self.bn_dw(dw))
        h = h * self.se(h)
        out = self.project(h)
        return out + x if self.use_res else out


class CoordRepRIRBlock(nn.Module):
    """Coordinate-Aware Rep-RIR Block with 1D X/Y spatial pooling."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        hidden_c = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 1, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True)
        )
        self.dw3 = nn.Conv2d(hidden_c, hidden_c, 3, stride=stride, padding=1, groups=hidden_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden_c)
        self.act_dw = nn.SiLU(inplace=True)
        
        # Coordinate attention
        mip = max(4, hidden_c // 8)
        self.conv1 = nn.Conv2d(hidden_c, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, hidden_c, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, hidden_c, 1, bias=False)
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        h = self.act_dw(self.bn_dw(self.dw3(h)))
        
        N, C, H, W = h.shape
        x_h = F.adaptive_avg_pool2d(h, (H, 1))
        x_w = F.adaptive_avg_pool2d(h, (1, W)).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act1(self.bn1(self.conv1(y)))
        x_h_att, x_w_att = torch.split(y, [H, W], dim=2)
        x_w_att = x_w_att.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h_att))
        a_w = torch.sigmoid(self.conv_w(x_w_att))
        h = h * a_h * a_w
        
        out = self.project(h)
        return out + x if self.use_res else out


class MicroStripRepRIRBlock(nn.Module):
    """Rep-RIR Block with micro orthogonal strip depthwise kernels (1x5 and 5x1)."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        hidden_c = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 1, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True)
        )
        self.dw_h = nn.Conv2d(hidden_c, hidden_c, (1, 5), stride=stride, padding=(0, 2), groups=hidden_c, bias=False)
        self.dw_v = nn.Conv2d(hidden_c, hidden_c, (5, 1), stride=stride, padding=(2, 0), groups=hidden_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden_c)
        self.act_dw = nn.SiLU(inplace=True)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        dw = self.dw_h(h) + self.dw_v(h)
        h = self.act_dw(self.bn_dw(dw))
        out = self.project(h)
        return out + x if self.use_res else out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bottleneck Modules (Directional Strip-PPM Mamba SSM)
# ─────────────────────────────────────────────────────────────────────────────
class StripPPMMamba(nn.Module):
    """Directional Strip Pooling Pyramid Mamba SSM."""
    def __init__(self, in_c: int, embed_dim: int = 56):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Conv2d(in_c, embed_dim // 2, 1, bias=False)
        self.conv_w = nn.Conv2d(in_c, embed_dim // 2, 1, bias=False)
        self.mamba = Mamba2D(d_model=embed_dim, d_state=8, d_conv=3)
        self.proj = nn.Conv2d(embed_dim, in_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        h_feat = F.interpolate(self.conv_h(self.pool_h(x)), size=(H, W), mode='nearest')
        w_feat = F.interpolate(self.conv_w(self.pool_w(x)), size=(H, W), mode='nearest')
        fused = torch.cat([h_feat, w_feat], dim=1)
        m = self.mamba(fused)
        return x + self.act(self.bn(self.proj(m)))


class CrossScan4WayMamba(nn.Module):
    """4-Way Bidirectional Cross-Scan Mamba SSM (Horizontal/Vertical Forward & Backward)."""
    def __init__(self, in_c: int, embed_dim: int = 48):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Conv2d(in_c, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))
        self.mamba_h = Mamba2D(d_model=embed_dim // 2, d_state=8, d_conv=3)
        self.mamba_v = Mamba2D(d_model=embed_dim // 2, d_state=8, d_conv=3)
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, in_c, 1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.in_proj(x)
        c1, c2 = torch.chunk(p, 2, dim=1)
        # Scan along standard orientation
        out_h = self.mamba_h(c1)
        # Scan along transposed orientation (vertical flow)
        out_v = self.mamba_v(c2.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        fused = torch.cat([out_h, out_v], dim=1)
        return x + self.out_proj(fused)


class RecurrentMamba(nn.Module):
    """Bi-directional recurrent shared-state Mamba SSM."""
    def __init__(self, in_c: int, embed_dim: int = 48):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Conv2d(in_c, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))
        self.shared_mamba = Mamba2D(d_model=embed_dim, d_state=8, d_conv=3)
        self.out_proj = nn.Sequential(nn.Conv2d(embed_dim, in_c, 1, bias=False), nn.BatchNorm2d(in_c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.in_proj(x)
        # Forward pass
        fwd = self.shared_mamba(p)
        # Reverse spatial scan
        rev = self.shared_mamba(torch.flip(p, dims=[2, 3]))
        fused = fwd + torch.flip(rev, dims=[2, 3])
        return x + self.act(self.out_proj(fused))


class LowRankMamba(nn.Module):
    """Factorized Low-Rank Mamba SSM."""
    def __init__(self, in_c: int, embed_dim: int = 48, rank: int = 24):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(in_c, rank, 1, bias=False), nn.BatchNorm2d(rank))
        self.mamba = Mamba2D(d_model=rank, d_state=6, d_conv=3)
        self.up = nn.Sequential(nn.Conv2d(rank, in_c, 1, bias=False), nn.BatchNorm2d(in_c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.up(self.mamba(self.down(x))))


class SparseRouterMamba(nn.Module):
    """Dynamic Sparse-Token Router Mamba SSM."""
    def __init__(self, in_c: int, embed_dim: int = 48):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Conv2d(in_c, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim))
        self.router = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(embed_dim, embed_dim, 1), nn.Sigmoid())
        self.mamba = Mamba2D(d_model=embed_dim, d_state=8, d_conv=3)
        self.out_proj = nn.Sequential(nn.Conv2d(embed_dim, in_c, 1, bias=False), nn.BatchNorm2d(in_c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.in_proj(x)
        gate = self.router(p)
        sparse_p = p * gate
        m = self.mamba(sparse_p)
        return x + self.act(self.out_proj(m))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge Stream Modules (High-Frequency Boundary Parsing)
# ─────────────────────────────────────────────────────────────────────────────
class EdgeStream(nn.Module):
    """Standard High-Frequency Wavelet Boundary Stream."""
    def __init__(self, in_c: int = 9, out_c: int = 28):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, high_freq: torch.Tensor):
        feat = self.conv(high_freq)
        logits = self.edge_head(feat)
        return feat, logits


class MorphoEdgeStream(nn.Module):
    """Differentiable Morphological Gradient Edge Stream (Soft Dilation minus Soft Erosion)."""
    def __init__(self, in_c: int = 9, out_c: int = 28):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.pool_max = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_min = lambda t: -F.max_pool2d(-t, 3, stride=1, padding=1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, high_freq: torch.Tensor):
        f = self.conv_in(high_freq)
        morph_grad = self.pool_max(f) - self.pool_min(f)
        feat = self.conv_out(f + morph_grad)
        logits = self.edge_head(feat)
        return feat, logits


class SobelLapEdgeStream(nn.Module):
    """Sobel-Laplacian Dual-Prior Boundary Stream."""
    def __init__(self, in_c: int = 9, out_c: int = 28):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        # Fixed 3x3 Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.refine = nn.Sequential(
            nn.Conv2d(out_c + 2, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, high_freq: torch.Tensor):
        f = self.conv_in(high_freq)
        f_mean = f.mean(dim=1, keepdim=True)
        gx = F.conv2d(f_mean, self.sobel_x, padding=1)
        gy = F.conv2d(f_mean, self.sobel_y, padding=1)
        feat = self.refine(torch.cat([f, gx, gy], dim=1))
        logits = self.edge_head(feat)
        return feat, logits


class CurvatureEdgeStream(nn.Module):
    """Curvature Tensor Divergence Boundary Stream."""
    def __init__(self, in_c: int = 9, out_c: int = 28):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.curv_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=2, dilation=2, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, high_freq: torch.Tensor):
        f = self.conv_in(high_freq)
        curv = self.curv_conv(f)
        feat = f + curv
        logits = self.edge_head(feat)
        return feat, logits


class MultiDilatedEdgeStream(nn.Module):
    """Multi-Scale Dilated Edge Stream."""
    def __init__(self, in_c: int = 9, out_c: int = 28):
        super().__init__()
        mid_c = out_c // 2
        self.branch1 = nn.Conv2d(in_c, mid_c, 3, padding=1, bias=False)
        self.branch2 = nn.Conv2d(in_c, mid_c, 3, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.edge_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, high_freq: torch.Tensor):
        f = torch.cat([self.branch1(high_freq), self.branch2(high_freq)], dim=1)
        feat = self.act(self.bn(f))
        logits = self.edge_head(feat)
        return feat, logits


class ContrastNormEdgeStream(nn.Module):
    """Local Contrast-Normalized Edge Stream."""
    def __init__(self, in_c: int = 9, out_c: int = 28):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True)
        )
        self.box = nn.AvgPool2d(3, stride=1, padding=1)
        self.edge_head = nn.Conv2d(out_c, 1, 1)

    def forward(self, high_freq: torch.Tensor):
        f = self.conv_in(high_freq)
        mean = self.box(f)
        sq_mean = self.box(f ** 2)
        var = torch.clamp(sq_mean - mean ** 2, min=1e-5)
        feat = (f - mean) / torch.sqrt(var)
        logits = self.edge_head(feat)
        return feat, logits


# ─────────────────────────────────────────────────────────────────────────────
# 5. Body Stream Reconstructor & Decoupled Cross-Stream Fusion
# ─────────────────────────────────────────────────────────────────────────────
class BodyDecoder(nn.Module):
    """Progressive Body Stream Feature Reconstructor."""
    def __init__(self, c4: int, c3: int, c2: int, c1: int):
        super().__init__()
        self.up4 = nn.Sequential(nn.Conv2d(c4 + c3, c3, 1, bias=False), nn.BatchNorm2d(c3), nn.SiLU(inplace=True))
        self.up3 = nn.Sequential(nn.Conv2d(c3 + c2, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
        self.up2 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(inplace=True))

    def forward(self, b4: torch.Tensor, x3: torch.Tensor, x2: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        u4 = F.interpolate(b4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up4(torch.cat([u4, x3], dim=1))
        u3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up3(torch.cat([u3, x2], dim=1))
        u2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.up2(torch.cat([u2, x1], dim=1))
        return d1


class CrossStreamFusion(nn.Module):
    """Standard Boundary-Guided Cross-Stream Interaction (CSFI)."""
    def __init__(self, body_c: int, edge_c: int):
        super().__init__()
        self.edge_gate = nn.Conv2d(edge_c, body_c, 1)

    def forward(self, body_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.edge_gate(edge_feat))
        return body_feat * (1.0 + gate)


class SelectiveCrossStreamFusion(nn.Module):
    """Selective Softmax Cross-Stream Interaction (SK-CSFI)."""
    def __init__(self, body_c: int, edge_c: int):
        super().__init__()
        self.proj_e = nn.Conv2d(edge_c, body_c, 1, bias=False)
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(body_c * 2, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, body_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        e = self.proj_e(edge_feat)
        weights = self.selector(torch.cat([body_feat, e], dim=1))
        return body_feat * weights[:, 0:1, :, :] + e * weights[:, 1:2, :, :]


class SelfCalibratedCrossStreamFusion(nn.Module):
    """Self-Calibrated Spatial Cross-Stream Interaction."""
    def __init__(self, body_c: int, edge_c: int):
        super().__init__()
        self.proj = nn.Conv2d(edge_c, body_c, 1, bias=False)
        self.calib = nn.Sequential(
            nn.Conv2d(body_c, body_c, 3, padding=1, groups=body_c, bias=False),
            nn.BatchNorm2d(body_c),
            nn.Sigmoid()
        )

    def forward(self, body_feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        e = self.proj(edge_feat)
        cal = self.calib(body_feat)
        return body_feat * cal + e * (1.0 - cal)

