#!/bin/bash
set -e

pip install --upgrade pip

pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu124

pip install causal-conv1d>=1.4.0 --no-build-isolation

pip install mamba-ssm --no-build-isolation

FLASH_LINEAR_ATTENTION_VERSION=0.5.1

FLASH_LINEAR_ATTENTION_VERSION=0.5.1

pip install fla-core==${FLASH_LINEAR_ATTENTION_VERSION}
pip install flash-linear-attention[cuda]==${FLASH_LINEAR_ATTENTION_VERSION} --no-deps

pip install \
    datasets>=5.0.0 \
    matplotlib==3.11.0 \
    ninja>=1.11 \
    numpy==2.5.0 \
    seaborn==0.13.2 \
    tiktoken==0.13.0 \
    tqdm==4.67.3 \
    transformers==4.52.4