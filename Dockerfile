# Dockerfile - GPU-optimized for NVIDIA Blackwell (compute capability 12.0)
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Keep Python output unbuffered and make /app importable everywhere
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.12, FFmpeg and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    libass-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libx264-dev \
    libmp3lame-dev \
    libopus-dev \
    libvpx-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3 and add python symlink
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip (ignore-installed to avoid debian package conflict)
RUN python3 -m pip install --no-cache-dir --ignore-installed pip --break-system-packages

# Install PyTorch nightly with CUDA 12.8 support (includes Blackwell sm_120)
RUN pip install --no-cache-dir --break-system-packages \
    --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install TensorFlow with CUDA support
# Note: TensorFlow will JIT compile kernels for Blackwell (sm_120) on first run
RUN pip install --no-cache-dir --break-system-packages \
    tensorflow[and-cuda]>=2.18.0 \
    tensorflow-hub>=0.15.0

# Install remaining Python dependencies
RUN pip install --no-cache-dir --break-system-packages \
    opencv-python \
    "Pillow>=10.0.0" \
    "numpy>=1.26,<2.0" \
    "torchfile>=0.1.0"

# Create standard runtime directories
RUN mkdir -p /app/work /app/input /app/output /app/models

CMD ["bash"]
