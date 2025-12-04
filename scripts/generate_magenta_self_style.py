#!/usr/bin/env python3
"""
Generate Magenta Self-Style Samples

Uses the original image as both content and style for Magenta arbitrary style transfer.
This creates a unique "enhanced" version of each image using its own patterns.

Run inside Docker:
    docker-compose run --rm web bash -lc "python /web/generate_magenta_self_style.py"
"""

import os
import sys
import subprocess
import random
from pathlib import Path

# Paths
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))
PRESET_SAMPLES_DIR = WEB_DIR / "static" / "preset_samples"
OUTPUT_DIR = WEB_DIR / "static" / "magenta_self_style"
APP_DIR = Path("/app")

# Magenta settings
MAGENTA_TILE = 512
MAGENTA_OVERLAP = 64
SCALE = 720
BLEND = 0.95


def get_random_images(n=30, seed=42):
    """Get n random images from preset_samples."""
    all_images = list(PRESET_SAMPLES_DIR.glob("*.jpg")) + list(PRESET_SAMPLES_DIR.glob("*.png"))

    if len(all_images) < n:
        print(f"Warning: Only found {len(all_images)} images, using all")
        n = len(all_images)

    random.seed(seed)
    return random.sample(all_images, n)


def generate_self_style(input_image, output_path):
    """
    Run Magenta style transfer using the same image as both content and style.
    """
    # Build pipeline command - use image as both input and style
    cmd = [
        "python3", "/app/pipeline.py",
        "--input_image", str(input_image),
        "--output_image", str(output_path),
        "--model_type", "magenta",
        "--magenta_style", str(input_image),  # Same image as style!
        "--magenta_tile", str(MAGENTA_TILE),
        "--magenta_overlap", str(MAGENTA_OVERLAP),
        "--scale", str(SCALE),
        "--blend", str(BLEND),
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False

    return output_path.exists()


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get random images
    images = get_random_images(n=30, seed=42)
    print(f"Processing {len(images)} images with Magenta self-style...")

    success = 0
    for i, img_path in enumerate(images, 1):
        # Output filename
        stem = img_path.stem
        output_name = f"selfstyle_{stem}.jpg"
        output_path = OUTPUT_DIR / output_name

        if output_path.exists():
            print(f"[{i}/{len(images)}] Skipping (exists): {output_name}")
            success += 1
            continue

        print(f"[{i}/{len(images)}] Processing: {img_path.name}")

        if generate_self_style(img_path, output_path):
            print(f"  -> Created: {output_name}")
            success += 1
        else:
            print(f"  -> FAILED")

    print(f"\nComplete! Generated {success}/{len(images)} self-styled images")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
