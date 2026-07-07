#!/bin/bash
set -e

echo "============================================="
echo "[INFO] Installing Python Dependencies..."
echo "============================================="

# ------------------------------------------------------------
# 1. Upgrade pip and build tools
# ------------------------------------------------------------
echo "[INFO] Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel ninja packaging

# ------------------------------------------------------------
# 2. Install PyTorch 2.4.0 + CUDA 12.1
# ------------------------------------------------------------
echo "[INFO] Installing PyTorch 2.4.0 + CUDA 12.1..."
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# 3. Install Mamba & Causal Conv1d
# ------------------------------------------------------------
echo "[INFO] Installing Mamba & Causal Conv1d..."
pip install causal-conv1d>=1.4.0 --no-build-isolation --no-deps
pip install mamba-ssm --no-build-isolation --no-deps

# ------------------------------------------------------------
# 4. Install Flash Linear Attention
# ------------------------------------------------------------
echo "[INFO] Installing Flash Linear Attention..."
FLASH_LINEAR_ATTENTION_VERSION=0.5.1
pip install fla-core==${FLASH_LINEAR_ATTENTION_VERSION} --no-deps
pip install flash-linear-attention[cuda]==${FLASH_LINEAR_ATTENTION_VERSION} --no-deps

# ------------------------------------------------------------
# 5. Install remaining requirements
# ------------------------------------------------------------
echo "[INFO] Installing remaining requirements..."
pip install \
    einops \
    datasets>=5.0.0 \
    matplotlib==3.11.0 \
    numpy==2.5.0 \
    seaborn==0.13.2 \
    tiktoken==0.13.0 \
    tqdm==4.67.3 \
    transformers==4.52.4 \
    lm-eval \
    accelerate \
    torch==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# 6. Upgrade Triton (Done last to prevent pip from overwriting PyTorch)
# ------------------------------------------------------------
echo "[INFO] Upgrading Triton (ignoring dependencies)..."
pip install "triton>=3.1.0" --upgrade --no-deps

# ------------------------------------------------------------
# 7. Verification
# ------------------------------------------------------------
echo ""
echo "============================================="
echo "✅ Installation complete!"
echo "Verifying PyTorch CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torch version: {torch.__version__}')"
echo "============================================="