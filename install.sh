#!/bin/bash
set -e

python -m pip install --upgrade "pip<26.3" setuptools wheel ninja packaging
python -m pip uninstall -y torchvision torchaudio causal-conv1d mamba-ssm tensorflow tensorflow-cpu tf-keras || true

IS_BLACKWELL=false
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    if echo "$GPU_NAME" | grep -qiE "RTX 50|B100|B200|Blackwell"; then
        IS_BLACKWELL=true
    fi
fi

if [ "$IS_BLACKWELL" = true ]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
else
    python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1" --no-deps
python -m pip install einops

export TORCH_CUDA_ARCH_LIST="native"
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
python -m pip install causal-conv1d --no-build-isolation --no-cache-dir
python -m pip install "mamba-ssm==2.2.5" --no-build-isolation --no-cache-dir --no-deps

python -m pip install \
    "datasets>=5.0.0" \
    matplotlib==3.11.0 \
    numpy==2.4.6 \
    seaborn==0.13.2 \
    tiktoken==0.13.0 \
    tqdm==4.67.3 \
    transformers==4.52.4 \
    lm-eval \
    accelerate

python -c "import torch, fla, mamba_ssm; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| FLA + Mamba OK')"
