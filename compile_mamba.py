import os, sys, subprocess, shutil, pathlib, sysconfig
import torch

major, minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (7, 5)
arch = f"compute_{major}{minor},code=sm_{major}{minor}"

env = os.environ.copy()
env["CUDA_HOME"] = "/usr/local/cuda"
env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja", "packaging", "einops"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "causal-conv1d", "--no-build-isolation", "--no-deps"], env=env, check=True)

if os.path.exists("/content/mamba"):
    shutil.rmtree("/content/mamba")
subprocess.run(["git", "clone", "https://github.com/state-spaces/mamba", "/content/mamba"], check=True)

mamba_dir = pathlib.Path("/content/mamba")
(mamba_dir / "pyproject.toml").unlink(missing_ok=True)

setup_code = f'''import os
os.environ["CUDA_HOME"] = "/usr/local/cuda"
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

setup(
    name="mamba_ssm",
    version="2.3.2.post1",
    packages=find_packages(include=["mamba_ssm", "mamba_ssm.*"]),
    ext_modules=[
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_fwd_fp32.cu",
                "csrc/selective_scan/selective_scan_fwd_fp16.cu",
                "csrc/selective_scan/selective_scan_fwd_bf16.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
                "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
            ],
            extra_compile_args={{
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-lineinfo",
                    "--threads", "4",
                    "-gencode", "arch={arch}",
                ],
            }},
            include_dirs=[str(Path("csrc/selective_scan").resolve())],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
'''

(mamba_dir / "setup.py").write_text(setup_code.strip(), encoding="utf-8")
subprocess.run([sys.executable, "setup.py", "install"], cwd="/content/mamba", env=env, check=True)

import selective_scan_cuda
from mamba_ssm import Mamba

layer = Mamba(d_model=64).cuda()
x = torch.randn(2, 16, 64, device="cuda")
out = layer(x)
print("SUCCESS:", out.shape)

sp = sysconfig.get_paths()["purelib"]
os.system(f"zip -r /content/pack_colab.zip {sp}/mamba_ssm* {sp}/selective_scan_cuda* {sp}/causal_conv1d*")

from google.colab import files
files.download("/content/pack_colab.zip")
