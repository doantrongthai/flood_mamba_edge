import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

# Add repo path
project_root = r"D:\AI Project\flood_mamba_edge"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.bgfmamba import build_model as build_mambaedge
build_m12_v16 = build_mambaedge

class DeployableModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

tasks = [
    {
        "name": "mambaedge_aifloodsense",
        "builder": build_mambaedge,
        "ckpt": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\pytorch_seeds\mambaedge\mambaedge_lovasz_focal_aifloodsense_s42.pth",
        "out_onnx": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\onnx\mambaedge_aifloodsense.onnx"
    },
    {
        "name": "mambaedge_floodkaggle",
        "builder": build_mambaedge,
        "ckpt": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\pytorch_seeds\mambaedge\mambaedge_lovasz_focal_floodkaggle_s42.pth",
        "out_onnx": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\onnx\mambaedge_floodkaggle.onnx"
    },
    {
        "name": "m12_v16_aifloodsense",
        "builder": build_m12_v16,
        "ckpt": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\pytorch_seeds\m12_v16\m12_v16_lovasz_focal_aifloodsense_s42.pth",
        "out_onnx": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\onnx\m12_v16_aifloodsense.onnx"
    },
    {
        "name": "m12_v16_floodkaggle",
        "builder": build_m12_v16,
        "ckpt": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\pytorch_seeds\m12_v16\m12_v16_lovasz_focal_floodkaggle_s42.pth",
        "out_onnx": r"D:\AI Project\FloodEdgeNet\jetson_deploy\weights\onnx\m12_v16_floodkaggle.onnx"
    }
]

dummy_input = torch.randn(1, 3, 128, 128)

print("=" * 80)
print("EXPORTING PARALLEL SCAN MAMBAEDGE MODELS TO ONNX")
print("=" * 80)

for t in tasks:
    name = t["name"]
    ckpt_path = t["ckpt"]
    out_path = t["out_onnx"]
    builder = t["builder"]

    print(f"\n[Processing] {name}...")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = builder(num_classes=1, input_size=128)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    model.eval()

    deploy_model = DeployableModel(model)

    with torch.no_grad():
        torch_out = deploy_model(dummy_input)

    # Export ONNX
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        deploy_model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )
    sz = os.path.getsize(out_path)
    print(f"  -> Exported successfully: {out_path} ({sz:,} bytes)")

    # Verify with ONNX Runtime
    sess = ort.InferenceSession(out_path)
    ort_out = sess.run(None, {'input': dummy_input.numpy()})[0]
    diff = np.abs(torch_out.numpy() - ort_out)
    max_d = diff.max()
    print(f"  -> ONNX Runtime verification max difference: {max_d:.2e}")
    assert max_d < 1e-4, f"Tolerance exceeded on {name}: {max_d}"
    print(f"  -> [OK] Numerical match confirmed (< 1e-4)!")

print("\n" + "=" * 80)
print("ALL MAMBAEDGE & M12_V16 ONNX MODELS EXPORTED AND VERIFIED SUCCESSFULLY!")
print("=" * 80)

