#!/bin/bash
set -e

echo "============================================="
echo "[INFO] Installing Python Dependencies... (note - we use python 3.11)"
echo "============================================="

# ------------------------------------------------------------
# 1. Upgrade pip and build tools
# ------------------------------------------------------------
echo "[INFO] Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel ninja packaging

# ------------------------------------------------------------
# 2. Install PyTorch 2.6.0 + CUDA 12.4
# ------------------------------------------------------------
echo "[INFO] Installing PyTorch 2.6.0 + CUDA 12.4..."
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 3. Install Precompiled Mamba & Causal Conv1d
# ------------------------------------------------------------
echo "[INFO] Installing Mamba & Causal Conv1d (Precompiled Wheels)..."
python -m pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python -m pip install https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# ------------------------------------------------------------
# 4. Install Flash Linear Attention
# ------------------------------------------------------------
echo "[INFO] Installing Flash Linear Attention..."
FLASH_LINEAR_ATTENTION_VERSION=0.5.1
python -m pip install fla-core==${FLASH_LINEAR_ATTENTION_VERSION} --no-deps
python -m pip install flash-linear-attention[cuda]==${FLASH_LINEAR_ATTENTION_VERSION} --no-deps

# ------------------------------------------------------------
# 5. Install remaining requirements
# ------------------------------------------------------------
echo "[INFO] Installing remaining requirements..."
python -m pip install \
    einops \
    datasets>=5.0.0 \
    matplotlib==3.11.0 \
    numpy==2.4.6 \
    seaborn==0.13.2 \
    tiktoken==0.13.0 \
    tqdm==4.67.3 \
    transformers==4.52.4 \
    lm-eval \
    accelerate \
    torch==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# 6. Upgrade Triton (Done last to prevent pip from overwriting PyTorch)
# ------------------------------------------------------------
echo "[INFO] Upgrading Triton (ignoring dependencies)..."
python -m pip install "triton>=3.1.0" --upgrade --no-deps

# ------------------------------------------------------------
# 7. Verification
# ------------------------------------------------------------
echo ""
echo "============================================="
echo "✅ Installation complete!"
echo "Verifying PyTorch CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torch version: {torch.__version__}')"
echo "============================================="