import os
import torch
import importlib
import inspect

AUTO_PRETRAINED_MODELS = {'deeplabv3'}


def get_model(model_name, num_classes=1, seed=None, input_size=128, pretrained=None):
    """
    Initialize model with deterministic weights.
    Supports Mamba Edge architectures (MambaFloodLite, LCMamNet, UltraLight VM-UNet, LightM-UNet)
    and standard real-time baseline models.

    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes (default 1 for binary flood segmentation)
        seed: Random seed for reproducibility
        input_size: Input image dimension (default: 128)
        pretrained: Load pretrained backbone weights if supported
    """
    if input_size % 16 != 0:
        raise ValueError(f"input_size={input_size} must be divisible by 16.")

    alias_map = {
        'fast-scnn': 'fast_scnn',
        'u_lite': 'ulite',
        'u-lite': 'ulite',
        'stdc': 'stdc1',
        'ultralight': 'ultralight_vmunet',
        'ultralight-vmunet': 'ultralight_vmunet',
        'ultralight_vm_unet': 'ultralight_vmunet',
        'lightmunet': 'lightm_unet',
        'lightm-unet': 'lightm_unet',
        'mamba_flood_lite': 'mambafloodlite',
        'mamba-flood-lite': 'mambafloodlite',
        'satreg': 'satreg_cmunet',
        'satreg-cmunet': 'satreg_cmunet',
        'satreg_cm_unet': 'satreg_cmunet',
        'cmunet': 'satreg_cmunet',
        'cm-unet': 'satreg_cmunet',
        'cm_unet': 'satreg_cmunet',
        'hybrid_seg': 'hybridseg',
        'hybrid-seg': 'hybridseg',
    }
    canonical_name = alias_map.get(model_name.lower(), model_name.lower())

    if pretrained is None:
        pretrained = canonical_name in AUTO_PRETRAINED_MODELS
    elif not pretrained and canonical_name in AUTO_PRETRAINED_MODELS:
        pretrained = True

    model_file = os.path.join(os.path.dirname(__file__), f'{canonical_name}.py')
    if not os.path.exists(model_file):
        available = [
            f.replace('.py', '')
            for f in os.listdir(os.path.dirname(__file__))
            if f.endswith('.py') and f not in ('__init__.py', 'mamba_block.py')
        ]
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available: {', '.join(sorted(available))}"
        )

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    module = importlib.import_module(f'models.{canonical_name}')

    sig = inspect.signature(module.build_model)
    kwargs = {'num_classes': num_classes}
    if 'input_size' in sig.parameters:
        kwargs['input_size'] = input_size
    if 'pretrained' in sig.parameters:
        kwargs['pretrained'] = pretrained
    if 'variant' in sig.parameters:
        if model_name.lower() in ('cmunet', 'cm-unet', 'cm_unet'):
            kwargs['variant'] = 'cmunet'
        else:
            kwargs['variant'] = 'satreg'

    model = module.build_model(**kwargs)

    if seed is not None:
        print(f"[OK] Model '{canonical_name}' initialized with seed {seed} (pretrained={pretrained})")

    return model

