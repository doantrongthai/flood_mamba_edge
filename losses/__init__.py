from .bce          import build_loss as build_bce
from .dice         import build_loss as build_dice
from .focal        import build_loss as build_focal
from .tversky      import build_loss as build_tversky
from .lovasz       import build_loss as build_lovasz
from .lovasz_focal import build_loss as build_lovasz_focal
from .ce           import build_ce_loss as build_ce, build_ce_dice_loss as build_ce_dice
import inspect


def get_loss(loss_name, num_classes=1):
    all_losses = {
        'bce':          build_bce,
        'dice':         build_dice,
        'focal':        build_focal,
        'tversky':      build_tversky,
        'lovasz':       build_lovasz,
        'lovasz_focal': build_lovasz_focal,
        'ce':           build_ce,
        'ce_dice':      build_ce_dice,
        'cross_entropy': build_ce,
    }

    loss_name_clean = loss_name.lower().replace('-', '_')
    if loss_name_clean not in all_losses:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. "
            f"Available: {list(all_losses.keys())}"
        )

    tag = "BINARY" if num_classes == 1 else f"{num_classes}-class"
    print(f"[OK] Using {loss_name.upper()} for {tag}")

    build_fn = all_losses[loss_name_clean]
    if 'num_classes' in inspect.signature(build_fn).parameters:
        return build_fn(num_classes=num_classes)
    return build_fn()
