FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.11 via apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    wget \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install/upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Install PyTorch CUDA first so pip won't download a CPU torch as a transitive dependency
# The TIRA platform injects the torch wheel via --from=torch_pkg at build time.
# For local builds, comment out the line below and install torch manually:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
COPY --from=torch_pkg . /usr/local/lib/python3.11/dist-packages/

# Copy project
COPY . /opt/pan26_detector
WORKDIR /opt/pan26_detector

# Install remaining dependencies (torch already present, pip will skip it)
RUN python3.11 -m pip install -r requirements.txt

# Make torch find its bundled CUDA libs
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib:\
/usr/local/lib/python3.11/dist-packages/nvidia/nvtx/lib

# NLTK tokenizer data
RUN python3.11 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab', quiet=True)"

# Flash attention (optional — improves ModernBERT throughput on A100/H100)
RUN python3.11 -m pip install --no-cache --no-build-isolation flash-attn || true

# Verify all model files are present before committing the image
RUN python3.11 -c "from pathlib import Path; \
p=Path('/opt/pan26_detector/models'); \
assert (p/'deberta').exists(); \
assert (p/'tfidf_pipeline.pkl').exists(); \
assert (p/'lgbm_model.pkl').exists(); \
assert (p/'ensemble_config.pkl').exists(); \
print('OK')"

# Disable HuggingFace network access at inference time (all models are local)
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

ENTRYPOINT ["python3.11", "/opt/pan26_detector/predict.py"]