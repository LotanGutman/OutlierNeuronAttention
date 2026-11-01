#!/bin/bash
set -e

echo "[INFO] Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel ninja packaging

echo "[INFO] Detecting GPU architecture for PyTorch..."
IS_BLACKWELL=false
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    if echo "$GPU_NAME" | grep -qiE "RTX 50|B100|B200|Blackwell"; then
        IS_BLACKWELL=true
    fi
fi

if [ "$IS_BLACKWELL" = true ]; then
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip uninstall -y causal-conv1d mamba-ssm || true

echo "[INFO] Installing Flash Linear Attention WITHOUT letting it touch torch or triton..."
python -m pip install -U "fla-core==0.5.1" "flash-linear-attention[cuda]==0.5.1" --no-deps
python -m pip install einops  # fla's own light deps, installed manually so torch stays untouched

echo "[INFO] Confirming torch wasn't silently swapped..."
python -c "import torch; print('PyTorch OK:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "[INFO] Installing causal-conv1d and mamba-ssm"
export TORCH_CUDA_ARCH_LIST="native"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install causal-conv1d --no-build-isolation --no-cache-dir
# pip install mamba-ssm --no-build-isolation --no-cache-dir --no-deps
pip install "mamba-ssm==2.2.5" --no-build-isolation --no-cache-dir --no-deps

echo "[INFO] Installing remaining requirements..."
python -m pip install "datasets>=5.0.0" matplotlib==3.11.0 numpy==2.4.6 seaborn==0.13.2 tiktoken==0.13.0 tqdm==4.67.3 transformers==4.52.4 lm-eval accelerate

echo "[INFO] uninstalling torchvision (doesn't work well with our version combination)"
pip uninstall torchvision -y

echo "============================================="
echo "[INFO] Verifying core imports..."
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "
from fla.ops.gla import chunk_gla
print('chunk_gla import OK')
"
python -c "
import torch
from fla.layers import Mamba2
m = Mamba2(hidden_size=128, state_size=16, conv_kernel=4, expand=2, head_dim=64).cuda()
x = torch.randn(2, 64, 128, device='cuda', requires_grad=True)
y, *_ = m(x)
y.sum().backward()
print('Mamba2 forward+backward OK')
"
echo "============================================="