# Dockerfile
FROM python:3.12-slim

# Keep Python output unbuffered and make /app importable everywhere
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install FFmpeg and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libass-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libx264-dev \
    libmp3lame-dev \
    libopus-dev \
    libvpx-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create standard runtime directories
RUN mkdir -p /app/work /app/input /app/output /app/models

# If you prefer baking the code into the image, uncomment the next line
# When using docker-compose with "- ./:/app", leave it commented
# COPY . .

CMD ["bash"]