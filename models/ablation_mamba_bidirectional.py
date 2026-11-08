"""
Ablation Mamba Bidirectional (Aliased to proposed BGFMamba).
Maintained so that checkpoints trained under 'ablation_mamba_bidirectional' can be directly evaluated.
"""

from models.bgfmamba import BGFMamba, build_model

__all__ = ['BGFMamba', 'build_model']
