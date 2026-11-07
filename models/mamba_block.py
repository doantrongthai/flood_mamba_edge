"""
Mamba State Space Model (SSM) Block
Backed strictly by the official hardware-accelerated mamba_ssm (selective_scan_cuda).
"""

from mamba_ssm import Mamba

__all__ = ['Mamba']
