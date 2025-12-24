# Dockerfile - NGC TensorFlow with Blackwell GPU optimization
# Using NVIDIA's pre-optimized TensorFlow container for sm_120 support
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3

# Keep Python output unbuffered and make /app importable everywhere
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install FFmpeg and media codecs (NGC container has most deps already)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libass-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libx264-dev \
    libmp3lame-dev \
    libopus-dev \
    libvpx-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch nightly with CUDA 12.8 support (includes Blackwell sm_120)
# NGC TensorFlow container doesn't include PyTorch
RUN pip install --no-cache-dir \
    --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install TensorFlow Hub (NGC container has TensorFlow but not Hub)
RUN pip install --no-cache-dir \
    tensorflow-hub>=0.15.0

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    opencv-python \
    "Pillow>=10.0.0" \
    "numpy>=1.26,<2.0" \
    "torchfile>=0.1.0"

# Create standard runtime directories
RUN mkdir -p /app/work /app/input /app/output /app/models

CMD ["bash"]
