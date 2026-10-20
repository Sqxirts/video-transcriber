#!/usr/bin/env bash
# Launcher for the GPU transcriber.
#
# ctranslate2 does not find the pip-installed CUDA libraries on its own — cuBLAS,
# cuDNN and nvrtc ship inside the venv under site-packages/nvidia/*/lib, which is
# not on the loader path. Without this export you get a CUDA-unavailable fallback
# to CPU, or an outright "libcudnn.so.9 not found". Set it once here so every run
# is correct rather than depending on the caller's shell.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="$HERE/venv/lib/python3.12/site-packages"

export LD_LIBRARY_PATH="$SP/nvidia/cublas/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"

exec "$HERE/venv/bin/python" "$HERE/transcribe.py" "$@"
