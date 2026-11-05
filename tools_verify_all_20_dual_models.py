"""
Comprehensive Verification Script for all 20 Decoupled Dual-Stream Models (dual_v1 to dual_v20).
Checks:
- Parameter count strictly in [200,000, 300,000]
- Train forward pass with aux_head: torch.Size([2, 1, 128, 128])
- Eval forward pass: torch.Size([2, 1, 128, 128])
- Backward gradient computation
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import get_model

def main():
    print("=" * 80)
    print("VERIFYING ALL 20 DECOUPLED DUAL-STREAM MODELS (dual_v1 -> dual_v20)")
    print("Target Budget: 200,000 to 300,000 parameters | Image Size: 128x128")
    print("=" * 80)

    all_passed = True
    x = torch.randn(2, 3, 128, 128)

    for i in range(1, 21):
        m_name = f"dual_v{i}"
        try:
            model = get_model(m_name, num_classes=1, input_size=128)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Check parameter range
            if not (200000 <= params <= 300000):
                print(f"[FAIL] {m_name}: Params {params:,} outside [200k, 300k]!")
                all_passed = False
                continue

            # Check train mode forward pass
            model.train()
            out_tr, aux_tr = model(x)
            assert out_tr.shape == (2, 1, 128, 128), f"Wrong train out shape: {out_tr.shape}"
            assert aux_tr.shape == (2, 1, 128, 128), f"Wrong train aux shape: {aux_tr.shape}"

            # Check backward pass
            loss = out_tr.sum() + 0.4 * aux_tr.sum()
            loss.backward()

            # Check eval mode forward pass
            model.eval()
            with torch.no_grad():
                out_ev = model(x)
            assert out_ev.shape == (2, 1, 128, 128), f"Wrong eval out shape: {out_ev.shape}"

            print(f"[PASS] {m_name:<10}: {params:,} params | Train/Eval OK | Grad OK")

        except Exception as e:
            print(f"[ERROR] {m_name}: {e}")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("ALL 20 MODELS PASSED 100%! READY FOR BENCHMARKING!")
    else:
        print("SOME MODELS FAILED!")
    print("=" * 80)

if __name__ == "__main__":
    main()

