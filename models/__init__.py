"""
Flood Mamba Edge Models Registry
Provides unified `get_model` dynamic loader for BGFMamba and official baselines.
"""

import os
import torch
import importlib
import inspect

AUTO_PRETRAINED_MODELS = {'deeplabv3'}

# Mapping variations/hyphens to canonical model filenames
ALIAS_MAP = {
    # Proposed Architecture
    'bgfmamba': 'bgfmamba',
    'bgf-mamba': 'bgfmamba',
    'bgf_mamba': 'bgfmamba',
    'mambaedge': 'bgfmamba',
    'mamba-edge': 'bgfmamba',
    'mamba_edge': 'bgfmamba',
    'ablation_mamba_bidirectional': 'bgfmamba',
    'ablation-mamba-bidirectional': 'bgfmamba',

    # Mamba Baselines
    'mambafloodlite': 'mambafloodlite',
    'mamba_flood_lite': 'mambafloodlite',
    'mamba-flood-lite': 'mambafloodlite',
    'hybridseg': 'hybridseg',
    'hybrid_seg': 'hybridseg',
    'hybrid-seg': 'hybridseg',
    'lightm_unet': 'lightm_unet',
    'lightmunet': 'lightm_unet',
    'lightm-unet': 'lightm_unet',
    'ultralight_vmunet': 'ultralight_vmunet',
    'ultralight': 'ultralight_vmunet',
    'ultralight-vmunet': 'ultralight_vmunet',
    'ultralight_vm_unet': 'ultralight_vmunet',

    # Real-Time & Edge CNN Baselines
    'ulite': 'ulite',
    'u_lite': 'ulite',
    'u-lite': 'ulite',
    'fast_scnn': 'fast_scnn',
    'fast-scnn': 'fast_scnn',
    'stdc1': 'stdc1',
    'stdc': 'stdc1',
    'edanet': 'edanet',
    'espnetv2': 'espnetv2',
    'dabnet': 'dabnet',
    'enet': 'enet',
    'segnet': 'segnet',
    'segformer': 'segformer',
    'unet': 'unet',
    'deeplabv3': 'deeplabv3',
}


def list_available_models():
    """Returns list of canonical baseline and proposed model names."""
    models_dir = os.path.dirname(__file__)
    return sorted([
        f[:-3] for f in os.listdir(models_dir)
        if f.endswith('.py') and f not in ('__init__.py', 'mamba_block.py', 'mambaedge.py')
    ])


def get_model(model_name: str, num_classes: int = 1, seed: int = None, input_size: int = 128, pretrained: bool = None):
    """
    Initialize model by name directly based on file name or alias.

    Args:
        model_name: Name of model architecture (e.g., 'bgfmamba', 'lcmamnet', 'mambafloodlite')
        num_classes: Number of output classes (default 1 for binary flood segmentation)
        seed: Random seed for reproducibility
        input_size: Input image dimension (default: 128)
        pretrained: Load pretrained weights if supported
    """
    if input_size % 16 != 0:
        raise ValueError(f"input_size={input_size} must be divisible by 16.")

    canonical_name = ALIAS_MAP.get(model_name.lower().strip(), model_name.lower().strip())

    if pretrained is None:
        pretrained = canonical_name in AUTO_PRETRAINED_MODELS
    elif not pretrained and canonical_name in AUTO_PRETRAINED_MODELS:
        pretrained = True

    model_file = os.path.join(os.path.dirname(__file__), f'{canonical_name}.py')
    if not os.path.exists(model_file):
        available = list_available_models()
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available models in models/: {', '.join(available)}"
        )

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    module = importlib.import_module(f'models.{canonical_name}')

    if not hasattr(module, 'build_model'):
        raise AttributeError(f"Module 'models.{canonical_name}' does not implement 'build_model()'")

    sig = inspect.signature(module.build_model)
    kwargs = {'num_classes': num_classes}
    if 'input_size' in sig.parameters:
        kwargs['input_size'] = input_size
    if 'pretrained' in sig.parameters:
        kwargs['pretrained'] = pretrained

    model = module.build_model(**kwargs)

    if seed is not None:
        print(f"[OK] Model '{canonical_name}' initialized with seed {seed} (pretrained={pretrained})")

    return model


__all__ = ['get_model', 'list_available_models', 'ALIAS_MAP']
