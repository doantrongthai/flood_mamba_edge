"""
Official Hardware-Accelerated Mamba State Space Model (SSM) Block
Direct interface to official Tri Dao / Albert Gu mamba_ssm implementation.
Reference: Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "Official 'mamba_ssm' package is required to run Mamba models.\n"
                "Please install official mamba-ssm and causal-conv1d in your CUDA environment:\n"
                "  pip install mamba-ssm causal-conv1d"
            )
