import os
import argparse
import torch
import torch.nn as nn
from models import get_model


class DeployableModel(nn.Module):
    """
    Strips auxiliary heads and ensures single-tensor output for 
    clean ONNX / TFLite / Edge deployment.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def export_onnx(model_name, weights_path, output_path, input_size=128, num_classes=1, opset=13):
    print("=" * 70)
    print("FLOOD MAMBA EDGE EDGE DEPLOYMENT EXPORTER (ONNX)")
    print("=" * 70)
    print(f"Model Architecture : {model_name}")
    print(f"Weights Checkpoint : {weights_path}")
    print(f"Input Resolution   : {input_size}x{input_size}")
    print(f"Output ONNX Path   : {output_path}")
    print(f"ONNX Opset         : {opset}")
    print("=" * 70)

    # 1. Load Model and inspect checkpoint config
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if isinstance(checkpoint, dict):
            ckpt_num_classes = checkpoint.get('config', {}).get('num_classes')
            if ckpt_num_classes is not None:
                num_classes = ckpt_num_classes
        model = get_model(model_name, num_classes=num_classes, input_size=input_size)
        model.load_state_dict(state_dict)
        print(f"[OK] Checkpoint loaded successfully (num_classes={num_classes}) from: {weights_path}")
    else:
        model = get_model(model_name, num_classes=num_classes, input_size=input_size)
        print(f"[INFO] No valid checkpoint provided. Exporting with initialized weights (num_classes={num_classes}).")

    model.eval()
    deploy_model = DeployableModel(model)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # 2. Export to ONNX
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    try:
        torch.onnx.export(
            deploy_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamo=False
        )
        print(f"[OK] ONNX model exported: {output_path}")

        # 3. Verify ONNX model if onnx package is installed
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("[OK] ONNX model successfully verified with onnx.checker!")
        except Exception:
            pass

        file_size_kb = os.path.getsize(output_path) / 1024.0
        print(f"Model File Size: {file_size_kb:.2f} KB\n")

    except (ModuleNotFoundError, ImportError) as e:
        print(f"[INFO] Missing dependency for ONNX export ({e}).")
        print("       Run 'pip install onnx onnxscript' to enable ONNX export.")
    except Exception as e:
        print(f"[WARN] Export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="FLOOD MAMBA EDGE ONNX Exporter")
    parser.add_argument('--model', type=str, default='mambafloodlite', help="Model architecture")
    parser.add_argument('--weights', type=str, default=None, help="Path to trained .pth checkpoint")
    parser.add_argument('--size', type=int, default=128, help="Input size (default: 128)")
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--output', type=str, default='outputs/mambafloodlite.onnx')
    parser.add_argument('--opset', type=int, default=13)
    args = parser.parse_args()

    export_onnx(
        model_name=args.model,
        weights_path=args.weights,
        output_path=args.output,
        input_size=args.size,
        num_classes=args.num_classes,
        opset=args.opset
    )


if __name__ == '__main__':
    main()
