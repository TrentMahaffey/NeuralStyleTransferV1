#!/usr/bin/env python3
"""
Generate Self-Style Preset Samples

Creates sample images for Self Style presets by using source images
as both content AND style with Magenta arbitrary style transfer.

For enhanced detail: runs at high resolution (1080-2K) on center crop (~20% of image),
then resizes back to full dimensions to show fine style details blown up.

Output shows styled image with small original thumbnail in top-left corner.

Run inside Docker:
    docker-compose run --rm web bash -lc "python /app/scripts/generate_style_selfstyle.py"
"""

import os
import sys
import subprocess
import random
import sqlite3
import shutil
from pathlib import Path

try:
    from PIL import Image, ExifTags, ImageDraw
    from PIL.ImageOps import exif_transpose
except ImportError:
    Image = None
    ExifTags = None
    ImageDraw = None
    exif_transpose = None

# Paths
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))
INPUT_DIR = Path("/app/input/self_style_samples")  # Dedicated Self Style source images
INPUT_DIR_FALLBACK = Path("/app/input")  # Fallback to general input if dedicated is empty
OUTPUT_DIR = WEB_DIR / "static" / "preset_samples"
DB_PATH = WEB_DIR / "presets.db"
APP_DIR = Path("/app")

# Center crop settings for enhanced detail
CENTER_CROP_RATIO = 0.45  # Crop center ~45% of image (square crop from center)
HIGH_RES_SCALE = 1440     # Run style transfer at this resolution for detail
FINAL_OUTPUT_SIZE = 720   # Final output longest edge
THUMBNAIL_RATIO = 0.15    # Thumbnail is 15% of output width
THUMBNAIL_PADDING = 10    # Padding from corner in pixels
THUMBNAIL_BORDER = 3      # White border around thumbnail


def get_self_style_presets():
    """Get all Self Style presets from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get category_id for Self Style
    cursor.execute("SELECT id FROM categories WHERE name = 'Self Style'")
    row = cursor.fetchone()
    if not row:
        print("Self Style category not found in database")
        return []

    category_id = row['id']

    # Get all presets in Self Style category
    cursor.execute("""
        SELECT id, name, magenta_tile, magenta_overlap, blend, sample_image_path
        FROM presets
        WHERE category_id = ? AND magenta_style = 'SELF_STYLE'
        ORDER BY name
    """, (category_id,))

    presets = []
    for row in cursor.fetchall():
        # Convert preset name to filename
        filename = row['name'].lower().replace(' ', '_') + '.jpg'
        presets.append({
            'id': row['id'],
            'name': row['name'],
            'filename': filename,
            'tile': row['magenta_tile'] or 512,
            'overlap': row['magenta_overlap'] or 64,
            'blend': row['blend'] or 0.95,
            'has_sample': bool(row['sample_image_path']),
        })

    conn.close()
    return presets


def update_sample_path(preset_id, sample_path):
    """Update the sample_image_path in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE presets
        SET sample_image_path = ?, has_sample_image = 1
        WHERE id = ?
    """, (sample_path, preset_id))
    conn.commit()
    conn.close()


def get_source_images():
    """Get source images from dedicated self_style_samples directory, fallback to general input."""
    images = []

    # First try the dedicated self_style_samples folder
    if INPUT_DIR.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(INPUT_DIR.glob(ext))
        images = [f for f in images if not f.name.startswith('.')]

    # If no images in dedicated folder, fall back to general input
    if not images and INPUT_DIR_FALLBACK.exists():
        print(f"No images in {INPUT_DIR}, falling back to {INPUT_DIR_FALLBACK}")
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(INPUT_DIR_FALLBACK.glob(ext))
        images = [f for f in images if not f.name.startswith('.')]

    return images


def get_premade_image_folders():
    """
    Get pre-made image folders from self_style_samples directory.

    Looks for folders named "Image X" containing:
    - "final image.*" or "Final Image.*" - The CROPPED CONTENT to style (input)
    - "style image.*" or "Style Image.*" - The STYLE REFERENCE for Magenta
    - "raw image.*" or "Raw Image.*" - Original full image (for thumbnail overlay)

    Returns list of dicts with 'folder', 'content', 'style', 'raw' paths.
    """
    folders = []

    if not INPUT_DIR.exists():
        return folders

    # Look for "Image X" folders
    for folder in sorted(INPUT_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.'):
            continue

        # Find files in folder (case-insensitive matching)
        content_image = None  # "final image" = cropped content to style
        style_image = None    # "style image" = style reference
        raw_image = None      # "raw image" = original for thumbnail

        for f in folder.iterdir():
            name_lower = f.name.lower()
            if name_lower.startswith('final image') or name_lower.startswith('final_image'):
                content_image = f
            elif name_lower.startswith('style image') or name_lower.startswith('style_image'):
                style_image = f
            elif name_lower.startswith('raw image') or name_lower.startswith('raw_image'):
                raw_image = f

        # Need both content and style to run Magenta
        if content_image and style_image:
            folders.append({
                'folder': folder,
                'content': content_image,  # What to style
                'style': style_image,      # Style reference
                'raw': raw_image,          # For thumbnail (optional)
            })

    return folders


def extract_center_crop(input_path, output_path, crop_ratio=CENTER_CROP_RATIO):
    """
    Extract center crop from an image.

    Args:
        input_path: Source image path
        output_path: Where to save the crop
        crop_ratio: Fraction of image to keep (0.45 = center 45%)

    Returns:
        True if successful, False otherwise
    """
    if Image is None:
        print("  PIL not available for center crop")
        return False

    try:
        img = Image.open(input_path)

        # Apply EXIF orientation
        if exif_transpose is not None:
            img = exif_transpose(img)

        width, height = img.size

        # Calculate crop box for center region
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # Crop and save
        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path, quality=95)

        return True
    except Exception as e:
        print(f"  Error extracting center crop: {e}")
        return False


def generate_self_style(input_image, output_path, tile, overlap, blend, scale=HIGH_RES_SCALE):
    """
    Run Magenta style transfer using the same image as both content and style.

    Args:
        input_image: Path to input image (may be a center crop)
        output_path: Where to save styled output
        tile: Magenta tile size
        overlap: Magenta overlap size
        blend: Style blend strength
        scale: Output resolution (higher = more detail)
    """
    cmd = [
        "python3", "/app/pipeline.py",
        "--input_image", str(input_image),
        "--output_image", str(output_path),
        "--model_type", "magenta",
        "--magenta_style", str(input_image),  # Same image as style!
        "--magenta_tile", str(tile),
        "--magenta_overlap", str(overlap),
        "--scale", str(scale),
        "--blend", str(blend),
    ]

    print(f"  Running: tile={tile}, overlap={overlap}, blend={blend}, scale={scale}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False

    return output_path.exists()


def add_thumbnail_overlay(styled_path, original_path, output_path, style_path=None):
    """
    Add thumbnail overlays to the styled image.

    For Self Style presets with separate content and style images:
    - Top-left: Content image (what's being styled)
    - Top-right: Style image (the style reference)

    For single-image presets:
    - Top-left: Original image only

    Args:
        styled_path: Path to the styled image (main display)
        original_path: Path to the content/original image (for left thumbnail)
        output_path: Where to save the final composited image
        style_path: Optional path to style image (for right thumbnail)
    """
    if Image is None:
        print("  PIL not available, skipping thumbnail overlay")
        return False

    try:
        styled = Image.open(styled_path)
        content_img = Image.open(original_path)

        # Apply EXIF orientation to content
        if exif_transpose is not None:
            content_img = exif_transpose(content_img)

        # Resize styled image to final output size
        styled_ratio = styled.width / styled.height
        if styled.width > styled.height:
            final_width = FINAL_OUTPUT_SIZE
            final_height = int(FINAL_OUTPUT_SIZE / styled_ratio)
        else:
            final_height = FINAL_OUTPUT_SIZE
            final_width = int(FINAL_OUTPUT_SIZE * styled_ratio)

        styled = styled.resize((final_width, final_height), Image.LANCZOS)

        # Calculate thumbnail size (15% of output width)
        thumb_width = int(final_width * THUMBNAIL_RATIO)

        # Create content thumbnail (left side)
        content_ratio = content_img.width / content_img.height
        content_thumb_height = int(thumb_width / content_ratio)
        content_thumb = content_img.resize((thumb_width, content_thumb_height), Image.LANCZOS)

        # Create bordered content thumbnail
        bordered_width = thumb_width + 2 * THUMBNAIL_BORDER
        bordered_height = content_thumb_height + 2 * THUMBNAIL_BORDER
        bordered_content = Image.new('RGB', (bordered_width, bordered_height), (255, 255, 255))
        bordered_content.paste(content_thumb, (THUMBNAIL_BORDER, THUMBNAIL_BORDER))

        # Paste content thumbnail in top-left corner
        styled.paste(bordered_content, (THUMBNAIL_PADDING, THUMBNAIL_PADDING))

        # If style image provided, add it to top-right
        if style_path and Path(style_path).exists():
            style_img = Image.open(style_path)
            if exif_transpose is not None:
                style_img = exif_transpose(style_img)

            style_ratio = style_img.width / style_img.height
            style_thumb_height = int(thumb_width / style_ratio)
            style_thumb = style_img.resize((thumb_width, style_thumb_height), Image.LANCZOS)

            # Create bordered style thumbnail
            style_bordered_width = thumb_width + 2 * THUMBNAIL_BORDER
            style_bordered_height = style_thumb_height + 2 * THUMBNAIL_BORDER
            bordered_style = Image.new('RGB', (style_bordered_width, style_bordered_height), (255, 255, 255))
            bordered_style.paste(style_thumb, (THUMBNAIL_BORDER, THUMBNAIL_BORDER))

            # Paste style thumbnail in top-right corner
            right_x = final_width - style_bordered_width - THUMBNAIL_PADDING
            styled.paste(bordered_style, (right_x, THUMBNAIL_PADDING))

        styled.save(output_path, quality=90)
        return True
    except Exception as e:
        print(f"  Error adding thumbnail overlay: {e}")
        return False


def create_comparison(original_path, styled_path, output_path):
    """
    DEPRECATED: Now uses thumbnail overlay instead of side-by-side.
    This function is kept for backwards compatibility but calls add_thumbnail_overlay.
    """
    return add_thumbnail_overlay(styled_path, original_path, output_path)


def run_magenta_style_transfer(content_path, style_path, output_path, tile, overlap, blend, scale=HIGH_RES_SCALE):
    """
    Run Magenta arbitrary style transfer with separate content and style images.

    Args:
        content_path: Path to the content image (what to style)
        style_path: Path to the style image (style reference)
        output_path: Where to save styled output
        tile: Magenta tile size
        overlap: Magenta overlap size
        blend: Style blend strength
        scale: Output resolution

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "python3", "/app/pipeline.py",
        "--input_image", str(content_path),
        "--output_image", str(output_path),
        "--model_type", "magenta",
        "--magenta_style", str(style_path),  # Style image (different from content!)
        "--magenta_tile", str(tile),
        "--magenta_overlap", str(overlap),
        "--scale", str(scale),
        "--blend", str(blend),
    ]

    print(f"  Running Magenta: tile={tile}, overlap={overlap}, blend={blend}, scale={scale}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False

    return output_path.exists()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="""
Generate Self Style preset samples with enhanced detail.

Two modes:
1. PRE-MADE: If "Image X" folders exist with "final image" files, use those directly
2. AUTO-GENERATE: Otherwise, extract center crop and run Magenta self-style

Pre-made folder structure:
  self_style_samples/
    Image 1/
      final image.png    <- Used as preset sample
      raw image.jpg      <- Original (reference only)
      style image.png    <- Cropped region (reference only)
    Image 2/
      ...
    """)
    parser.add_argument('--force', action='store_true', help='Regenerate all samples')
    parser.add_argument('--crop-ratio', type=float, default=CENTER_CROP_RATIO,
                        help=f'Center crop ratio (default: {CENTER_CROP_RATIO})')
    parser.add_argument('--high-res', type=int, default=HIGH_RES_SCALE,
                        help=f'High-res scale for style transfer (default: {HIGH_RES_SCALE})')
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for pre-made image folders first
    premade_folders = get_premade_image_folders()

    # Get all Self Style presets from database
    presets = get_self_style_presets()
    if not presets:
        print("No Self Style presets found in database")
        return

    print(f"Found {len(presets)} Self Style presets")

    if premade_folders:
        print(f"Found {len(premade_folders)} pre-made image folders with content+style!")
        print("Will run Magenta style transfer on each.")
        print()

        success = 0
        for idx, preset in enumerate(presets):
            # Cycle through pre-made folders
            premade = premade_folders[idx % len(premade_folders)]
            output_path = OUTPUT_DIR / preset['filename']

            if output_path.exists() and not args.force:
                print(f"Skipping (exists): {preset['filename']}")
                if not preset['has_sample']:
                    update_sample_path(preset['id'], f"preset_samples/{preset['filename']}")
                success += 1
                continue

            print(f"Styling: {preset['name']} (from {premade['folder'].name})")
            print(f"  Content: {premade['content'].name}")
            print(f"  Style:   {premade['style'].name}")

            # Run Magenta style transfer with content and style images
            temp_styled = OUTPUT_DIR / f"_temp_styled_{idx}.jpg"
            if run_magenta_style_transfer(
                premade['content'], premade['style'], temp_styled,
                preset['tile'], preset['overlap'], preset['blend'],
                scale=args.high_res
            ):
                # Add thumbnail overlays: content (left) + style (right)
                if add_thumbnail_overlay(temp_styled, premade['content'], output_path, style_path=premade['style']):
                    print(f"  -> Created: {preset['filename']} (styled + dual thumbnails)")
                else:
                    # Overlay failed - just use styled output
                    shutil.move(str(temp_styled), str(output_path))
                    print(f"  -> Created: {preset['filename']} (styled only)")

                update_sample_path(preset['id'], f"preset_samples/{preset['filename']}")
                success += 1
            else:
                print(f"  -> FAILED (style transfer)")

            # Clean up temp file
            if temp_styled.exists():
                temp_styled.unlink()

        print(f"\nComplete! Styled {success}/{len(presets)} samples from pre-made content+style")
        print(f"Output directory: {OUTPUT_DIR}")
        return

    # Fall back to auto-generation mode
    source_images = get_source_images()
    if not source_images:
        print("No source images found in /app/input/")
        return

    # Shuffle source images for variety
    random.seed(42)
    random.shuffle(source_images)
    print(f"Found {len(source_images)} source images to choose from")
    print(f"Settings: center crop={args.crop_ratio*100:.0f}%, high-res={args.high_res}p")
    print()

    success = 0
    for idx, preset in enumerate(presets):
        # Use a DIFFERENT source image for each preset!
        # Cycle through available images
        source_img = source_images[idx % len(source_images)]

        output_path = OUTPUT_DIR / preset['filename']

        if output_path.exists() and not args.force:
            print(f"Skipping (exists): {preset['filename']}")
            # Update database if not already set
            if not preset['has_sample']:
                update_sample_path(preset['id'], f"preset_samples/{preset['filename']}")
            success += 1
            continue

        print(f"Generating: {preset['name']} (using {source_img.name})")

        # Step 1: Extract center crop for enhanced detail
        temp_crop = OUTPUT_DIR / f"_temp_crop_{idx}.jpg"
        print(f"  Extracting center {args.crop_ratio*100:.0f}% crop...")
        if not extract_center_crop(source_img, temp_crop, args.crop_ratio):
            print(f"  -> FAILED (crop extraction)")
            continue

        # Step 2: Run self-style at high resolution on the crop
        temp_styled = OUTPUT_DIR / f"_temp_styled_{idx}.jpg"
        if generate_self_style(
            temp_crop, temp_styled,
            preset['tile'], preset['overlap'], preset['blend'],
            scale=args.high_res
        ):
            # Step 3: Add thumbnail overlay of original in top-left
            if add_thumbnail_overlay(temp_styled, source_img, output_path):
                print(f"  -> Created: {preset['filename']} (with thumbnail)")
            else:
                # Fall back to just the styled image if overlay fails
                shutil.move(str(temp_styled), str(output_path))
                print(f"  -> Created: {preset['filename']} (styled only)")

            # Update database
            update_sample_path(preset['id'], f"preset_samples/{preset['filename']}")
            success += 1
        else:
            print(f"  -> FAILED (style transfer)")

        # Clean up temp files
        for temp_file in [temp_crop, temp_styled]:
            if temp_file.exists():
                temp_file.unlink()

    print(f"\nComplete! Generated {success}/{len(presets)} self-style samples")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nEach sample shows: CENTER CROP STYLED @ {args.high_res}p with original thumbnail")


if __name__ == "__main__":
    main()
