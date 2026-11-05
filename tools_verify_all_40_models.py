import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from models import get_model

def test_all():
    models_to_test = [f"m11_v{i}" for i in range(1, 21)] + [f"m12_v{i}" for i in range(1, 21)]
    print(f"Total models to verify: {len(models_to_test)}\n")

    results = []
    failed = []

    for name in models_to_test:
        try:
            m = get_model(name, num_classes=1, input_size=128)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)

            # Test forward in eval mode
            m.eval()
            x = torch.randn(2, 3, 128, 128)
            with torch.no_grad():
                out_eval = m(x)

            # Test forward in train mode
            m.train()
            out_train = m(x)
            if isinstance(out_train, (tuple, list)):
                out_shape = f"Main: {list(out_train[0].shape)}, Aux: {list(out_train[1].shape)}"
            else:
                out_shape = f"Main: {list(out_train.shape)}"

            status = "PASS"
            if not (200000 <= params <= 300000):
                status = f"FAIL (OUT OF BUDGET: {params:,})"
                failed.append((name, status))

            print(f"[{status}] {name:<10}: {params:>8,} params | {out_shape}")
            results.append((name, params, status))
        except Exception as e:
            print(f"[ERROR] {name:<10}: {e}")
            failed.append((name, str(e)))

    print("\n" + "=" * 60)
    print(f"VERIFICATION SUMMARY: {len(results) - len(failed)} / {len(models_to_test)} PASSED")
    if failed:
        print("FAILURES DETECTED:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("ALL 40 MODELS STRICTLY COMPLY WITH 200K-300K BUDGET & PASS FORWARD PASS!")
        print("=" * 60)

if __name__ == "__main__":
    test_all()

