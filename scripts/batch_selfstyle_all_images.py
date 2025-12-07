#!/usr/bin/env python3
"""
Batch Self-Style Processing for Images 1-7
Generates multiple styled versions with different tile sizes for each image.

Run inside Docker:
    docker-compose run --rm style bash -lc "python /app/scripts/batch_selfstyle_all_images.py"
"""

import os
import subprocess
from pathlib import Path

# Paths (Docker paths)
INPUT_DIR = Path("/app/input/self_style_samples")
OUTPUT_DIR = Path("/app/output/batch_selfstyle")

# Tile sizes with 12.5% overlap ratio (proven to work well)
TILE_CONFIGS = [
    (128, 16),   # tile128_overlap16
    (160, 20),   # tile160_overlap20
    (192, 24),   # tile192_overlap24
    (224, 28),   # tile224_overlap28
    (256, 32),   # tile256_overlap32
    (384, 48),   # tile384_overlap48
    (512, 64),   # tile512_overlap64
]

# Style transfer settings
HIGH_RES_SCALE = 1440  # Output resolution for high detail
BLEND = 0.95


def find_images_in_folder(folder_path):
    """Find Raw/Final Image and Style Image in a folder."""
    content_image = None  # "final image" - cropped content to style
    style_image = None    # "style image" - style reference
    raw_image = None      # "raw image" - original (for reference)

    for f in folder_path.iterdir():
        name_lower = f.name.lower()
        if name_lower.startswith('final image') or name_lower.startswith('final_image'):
            content_image = f
        elif name_lower.startswith('style image') or name_lower.startswith('style_image') or name_lower.startswith('styled image'):
            style_image = f
        elif name_lower.startswith('raw image') or name_lower.startswith('raw_image'):
            raw_image = f

    return content_image, style_image, raw_image


def run_magenta_style_transfer(content_path, style_path, output_path, tile, overlap, scale=HIGH_RES_SCALE, blend=BLEND):
    """
    Run Magenta arbitrary style transfer.
    """
    cmd = [
        "python3", "/app/pipeline.py",
        "--input_image", str(content_path),
        "--output_image", str(output_path),
        "--model_type", "magenta",
        "--magenta_style", str(style_path),
        "--magenta_tile", str(tile),
        "--magenta_overlap", str(overlap),
        "--scale", str(scale),
        "--blend", str(blend),
    ]

    print(f"    Running: tile={tile}, overlap={overlap}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:300]}")
        return False

    return output_path.exists()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch self-style processing for Images 1-7')
    parser.add_argument('--force', action='store_true', help='Regenerate existing files')
    parser.add_argument('--scale', type=int, default=HIGH_RES_SCALE, help=f'Output scale (default: {HIGH_RES_SCALE})')
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all Image folders
    image_folders = []
    for folder in sorted(INPUT_DIR.iterdir()):
        if folder.is_dir() and folder.name.startswith('Image '):
            image_folders.append(folder)

    print(f"Found {len(image_folders)} image folders to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tile configurations: {len(TILE_CONFIGS)}")
    print(f"Scale: {args.scale}p")
    print()

    total_generated = 0
    total_skipped = 0

    for folder in image_folders:
        print(f"\n{'='*60}")
        print(f"Processing: {folder.name}")
        print(f"{'='*60}")

        content_image, style_image, raw_image = find_images_in_folder(folder)

        if not content_image:
            print(f"  WARNING: No 'final image' found in {folder.name}")
            continue
        if not style_image:
            print(f"  WARNING: No 'style image' found in {folder.name}")
            continue

        print(f"  Content: {content_image.name}")
        print(f"  Style: {style_image.name}")
        if raw_image:
            print(f"  Raw: {raw_image.name}")

        # Extract image number from folder name
        image_num = folder.name.replace('Image ', '')

        # Generate output for each tile config
        for tile_size, overlap in TILE_CONFIGS:
            output_name = f"img{image_num}_tile{tile_size}_overlap{overlap}.jpg"
            output_path = OUTPUT_DIR / output_name

            if output_path.exists() and not args.force:
                print(f"  Skipping (exists): {output_name}")
                total_skipped += 1
                continue

            success = run_magenta_style_transfer(
                content_image, style_image, output_path,
                tile_size, overlap, scale=args.scale
            )

            if success:
                print(f"    -> Created: {output_name}")
                total_generated += 1
            else:
                print(f"    -> FAILED: {output_name}")

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"Generated: {total_generated} new images")
    print(f"Skipped: {total_skipped} existing images")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
