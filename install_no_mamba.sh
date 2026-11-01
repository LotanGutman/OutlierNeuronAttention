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
    echo "[INFO] Blackwell GPU detected ($GPU_NAME). Installing PyTorch with CUDA 12.8 (sm_120 support)..."
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    echo "[INFO] Standard GPU detected. Installing PyTorch 2.6.0 + CUDA 12.4..."
    python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

echo "[INFO] Removing any stray/mismatched causal-conv1d or mamba-ssm..."
python -m pip uninstall -y causal-conv1d mamba-ssm || true

echo "[INFO] Installing Flash Linear Attention WITHOUT letting it touch torch or triton..."
python -m pip install -U "fla-core==0.5.1" "flash-linear-attention[cuda]==0.5.1" --no-deps
python -m pip install einops  # fla's own light deps, installed manually so torch stays untouched

echo "[INFO] Installing remaining requirements..."
python -m pip install "datasets>=5.0.0" matplotlib==3.11.0 numpy==2.4.6 seaborn==0.13.2 tiktoken==0.13.0 tqdm==4.67.3 transformers==4.52.4 lm-eval accelerate

echo "[INFO] uninstalling torchvision (doesn't work well with our version combination)"
pip uninstall torchvision -y

echo "============================================="
echo "[INFO] Verifying core imports..."
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "
try:
    from fla.ops.gla import chunk_gla
    print('chunk_gla import OK')
except Exception as e:
    print('Error occurred checking chunk_gla:', e)
"