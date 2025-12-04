#!/usr/bin/env python3
"""
Generate sample images for all presets using randomized input images.
Run this inside Docker to generate the preset gallery.

Usage:
    docker-compose run --rm style bash -lc "python /app/web/generate_preset_samples.py"
"""

import os
import sys
import json
import sqlite3
import subprocess
import shutil
import random
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Paths for presets (now from SQLite)
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))
PRESETS_DIR = WEB_DIR / "presets"
PRESET_SAMPLES_DIR = WEB_DIR / "static" / "preset_samples"
PRESETS_DB = WEB_DIR / "presets.db"


def load_presets_from_db():
    """Load all presets from SQLite database (source of truth)."""
    if not PRESETS_DB.exists():
        print(f"ERROR: presets.db not found at {PRESETS_DB}")
        sys.exit(1)

    presets = []
    conn = sqlite3.connect(str(PRESETS_DB))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            p.id,
            p.name,
            p.description,
            c.name as category,
            p.model_type,
            p.model_path,
            p.io_preset,
            p.magenta_style,
            p.magenta_tile,
            p.magenta_overlap,
            p.blend,
            p.smooth_alpha,
            p.smooth_lightness,
            p.smooth_chroma,
            p.chroma_alpha,
            p.flow_ema,
            p.flow_alpha,
            p.model_b_type,
            p.model_b_path,
            p.io_preset_b,
            p.magenta_style_b,
            p.magenta_tile_b,
            p.magenta_overlap_b,
            p.model_c_type,
            p.model_c_path,
            p.io_preset_c,
            p.model_d_type,
            p.model_d_path,
            p.io_preset_d,
            p.blend_models_weights,
            p.blend_models_lab,
            p.blend_models_lab_weights,
            p.region_mode,
            p.region_count,
            p.region_feather,
            p.region_blend_spec,
            p.region_morph,
            p.region_rotate,
            p.region_sizes,
            p.sample_image_path,
            GROUP_CONCAT(t.name) as tags
        FROM presets p
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN preset_tags pt ON p.id = pt.preset_id
        LEFT JOIN tags t ON pt.tag_id = t.id
        GROUP BY p.id
        ORDER BY p.id
    """)

    for row in cursor.fetchall():
        # Build params dict from individual columns
        params = {}
        if row['model_type']:
            params['model_type'] = row['model_type']
        if row['model_path']:
            params['model'] = row['model_path']
        if row['io_preset']:
            params['io_preset'] = row['io_preset']
        if row['magenta_style']:
            params['magenta_style'] = row['magenta_style']
        if row['magenta_tile']:
            params['magenta_tile'] = row['magenta_tile']
        if row['magenta_overlap'] is not None:
            params['magenta_overlap'] = row['magenta_overlap']
        if row['blend'] is not None:
            params['blend'] = row['blend']
        if row['smooth_alpha'] is not None:
            params['smooth_alpha'] = row['smooth_alpha']
        if row['smooth_lightness']:
            params['smooth_lightness'] = bool(row['smooth_lightness'])
        if row['smooth_chroma']:
            params['smooth_chroma'] = bool(row['smooth_chroma'])
        if row['chroma_alpha'] is not None:
            params['chroma_alpha'] = row['chroma_alpha']
        if row['flow_ema']:
            params['flow_ema'] = bool(row['flow_ema'])
        if row['flow_alpha'] is not None:
            params['flow_alpha'] = row['flow_alpha']
        # Secondary model
        if row['model_b_type']:
            params['model_b_type'] = row['model_b_type']
        if row['model_b_path']:
            params['model_b'] = row['model_b_path']
        if row['io_preset_b']:
            params['io_preset_b'] = row['io_preset_b']
        if row['magenta_style_b']:
            params['magenta_style_b'] = row['magenta_style_b']
        if row['magenta_tile_b']:
            params['magenta_tile_b'] = row['magenta_tile_b']
        if row['magenta_overlap_b'] is not None:
            params['magenta_overlap_b'] = row['magenta_overlap_b']
        # Third model
        if row['model_c_type']:
            params['model_c_type'] = row['model_c_type']
        if row['model_c_path']:
            params['model_c'] = row['model_c_path']
        if row['io_preset_c']:
            params['io_preset_c'] = row['io_preset_c']
        # Fourth model
        if row['model_d_type']:
            params['model_d_type'] = row['model_d_type']
        if row['model_d_path']:
            params['model_d'] = row['model_d_path']
        if row['io_preset_d']:
            params['io_preset_d'] = row['io_preset_d']
        # Blend weights
        if row['blend_models_weights']:
            params['blend_models_weights'] = row['blend_models_weights']
        if row['blend_models_lab']:
            params['blend_models_lab'] = bool(row['blend_models_lab'])
        if row['blend_models_lab_weights']:
            params['blend_models_lab_weights'] = row['blend_models_lab_weights']
        # Region settings
        if row['region_mode']:
            params['region_mode'] = row['region_mode']
        if row['region_count']:
            params['region_count'] = row['region_count']
        if row['region_feather']:
            params['region_feather'] = row['region_feather']
        if row['region_blend_spec']:
            params['region_blend_spec'] = row['region_blend_spec']
        if row['region_morph']:
            params['region_morph'] = row['region_morph']
        if row['region_rotate'] is not None:
            params['region_rotate'] = row['region_rotate']
        if row['region_sizes']:
            params['region_sizes'] = row['region_sizes']

        preset = {
            'id': row['id'],
            'name': row['name'],
            'description': row['description'],
            'category': row['category'],
            'params': params,
            'tags': row['tags'].split(',') if row['tags'] else [],
        }
        if row['sample_image_path']:
            preset['sample_image'] = row['sample_image_path']
        presets.append(preset)

    conn.close()
    return presets


# Load presets from database
PRESETS = load_presets_from_db()

# Paths (configurable via environment)
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", "/app"))
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))

PIPELINE = PIPELINE_DIR / "pipeline.py"
WORK_DIR = PIPELINE_DIR / "_work" / "preset_samples"
# Use timestamped subfolder for extracted frames to avoid conflicts with concurrent jobs
JOB_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
EXTRACTED_FRAMES_DIR = WORK_DIR / "extracted_frames" / f"job_{JOB_TIMESTAMP}"

# Output resolution for consistent sample image sizes (longest edge)
SAMPLE_SCALE = 720

# Input directories (from pipeline)
INPUT_IMAGES_DIR = PIPELINE_DIR / "input" / "images"
INPUT_VIDEOS_DIR = PIPELINE_DIR / "input_videos"

# Retry settings
MAX_RETRIES = 3  # Maximum number of retries per preset if it fails


def extract_random_frame(video_path: Path, output_path: Path) -> bool:
    """Extract a random frame from a video file."""
    try:
        # Get video duration using ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False

        duration = float(result.stdout.strip())
        if duration <= 0:
            return False

        # Pick a random timestamp (avoid first/last 10% of video)
        start_time = duration * 0.1
        end_time = duration * 0.9
        random_time = random.uniform(start_time, end_time)

        # Extract frame at that timestamp
        extract_cmd = [
            "ffmpeg", "-y", "-ss", str(random_time),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_path)
        ]
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and output_path.exists()

    except Exception as e:
        print(f"    Error extracting frame from {video_path.name}: {e}")
        return False


def collect_all_sample_images(videos_only: bool = True) -> list:
    """Collect sample images extracted from videos only (for consistency).

    Args:
        videos_only: If True, only use frames extracted from videos (default).
                    If False, also include static images from input/images.
    """
    all_images = []

    # Extract frames from videos (primary source)
    if INPUT_VIDEOS_DIR.exists():
        EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
        videos = []
        for ext in video_extensions:
            videos.extend(list(INPUT_VIDEOS_DIR.glob(ext)))

        print(f"Found {len(videos)} videos in {INPUT_VIDEOS_DIR}")

        for video in videos:
            # Extract 3-5 frames per video for more variety
            num_frames = random.randint(3, 5)
            for i in range(num_frames):
                frame_name = f"{video.stem}_frame_{i}.jpg"
                frame_path = EXTRACTED_FRAMES_DIR / frame_name

                # Skip if already extracted
                if frame_path.exists():
                    all_images.append(frame_path)
                    continue

                print(f"  Extracting frame {i+1}/{num_frames} from {video.name}...")
                if extract_random_frame(video, frame_path):
                    all_images.append(frame_path)

    # Optionally include static images (disabled by default)
    if not videos_only:
        if INPUT_IMAGES_DIR.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                all_images.extend(list(INPUT_IMAGES_DIR.glob(ext)))
            print(f"Also found {len(all_images)} static images in {INPUT_IMAGES_DIR}")

        # Also check input/ directly
        input_dir = PIPELINE_DIR / "input"
        if input_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                for img in input_dir.glob(ext):
                    if img not in all_images:
                        all_images.append(img)

    return all_images


def get_random_image(available_images: list, used_images: set = None) -> Path:
    """Get a random image, preferring unused images for variety."""
    if not available_images:
        return None

    if used_images is None:
        used_images = set()

    # Try to find an unused image first
    unused = [img for img in available_images if img not in used_images]

    if unused:
        return random.choice(unused)
    else:
        # All images used, just pick random
        return random.choice(available_images)


def build_command(preset: dict, input_image: Path, output_image: Path, scale: int = SAMPLE_SCALE) -> list:
    """Build pipeline.py command from preset params.

    Args:
        preset: Preset configuration dict
        input_image: Path to input image
        output_image: Path for output image
        scale: Output resolution (longest edge). Default SAMPLE_SCALE (720) for consistent sample sizes.
    """
    params = preset["params"]
    cmd = [
        "python3", str(PIPELINE),
        "--input_image", str(input_image),
        "--output_image", str(output_image),
        "--work_dir", str(WORK_DIR),
        "--scale", str(scale),  # Consistent output resolution
    ]

    # Model settings
    if params.get("model"):
        cmd += ["--model", params["model"]]
    if params.get("model_type"):
        cmd += ["--model_type", params["model_type"]]
    if params.get("io_preset"):
        cmd += ["--io_preset", params["io_preset"]]

    # Magenta settings
    if params.get("magenta_style"):
        cmd += ["--magenta_style", params["magenta_style"]]
    if params.get("magenta_tile"):
        cmd += ["--magenta_tile", str(params["magenta_tile"])]
    if params.get("magenta_overlap"):
        cmd += ["--magenta_overlap", str(params["magenta_overlap"])]

    # Secondary models
    for letter in ["b", "c", "d"]:
        model_key = f"model_{letter}"
        if params.get(model_key):
            cmd += [f"--model_{letter}", params[model_key]]
        type_key = f"model_{letter}_type"
        if params.get(type_key):
            cmd += [f"--model_{letter}_type", params[type_key]]
        preset_key = f"io_preset_{letter}"
        if params.get(preset_key):
            cmd += [f"--io_preset_{letter}", params[preset_key]]
        style_key = f"magenta_style_{letter}"
        if params.get(style_key):
            cmd += [f"--magenta_style_{letter}", params[style_key]]

    # Blending
    if params.get("blend") is not None:
        cmd += ["--blend", str(params["blend"])]
    if params.get("blend_models_weights"):
        cmd += ["--blend_models_weights", params["blend_models_weights"]]
    if params.get("blend_models_lab"):
        cmd += ["--blend_models_lab"]
    if params.get("blend_models_lab_weights"):
        cmd += ["--blend_models_lab_weights", params["blend_models_lab_weights"]]

    # Region settings (for region presets - works on single images too)
    if params.get("region_mode"):
        cmd += ["--region_mode", params["region_mode"]]
    if params.get("region_count"):
        cmd += ["--region_count", str(params["region_count"])]
    if params.get("region_feather"):
        cmd += ["--region_feather", str(params["region_feather"])]
    if params.get("region_blend_spec"):
        cmd += ["--region_blend_spec", params["region_blend_spec"]]
    if params.get("region_spin"):
        cmd += ["--region_spin", str(params["region_spin"])]
    # Note: region_morph is for video animation, skip for static images

    # Additional model support (e, f, g, h for multi-region presets)
    for letter in ["e", "f", "g", "h"]:
        model_key = f"model_{letter}"
        if params.get(model_key):
            cmd += [f"--model_{letter}", params[model_key]]
        type_key = f"model_{letter}_type"
        if params.get(type_key):
            cmd += [f"--model_{letter}_type", params[type_key]]
        preset_key = f"io_preset_{letter}"
        if params.get(preset_key):
            cmd += [f"--io_preset_{letter}", params[preset_key]]
        style_key = f"magenta_style_{letter}"
        if params.get(style_key):
            cmd += [f"--magenta_style_{letter}", params[style_key]]

    return cmd


def generate_sample(preset: dict, input_image: Path, output_dir: Path, skip_existing: bool = True, scale: int = SAMPLE_SCALE) -> tuple:
    """Generate a sample image for a preset.

    Args:
        preset: Preset configuration dict
        input_image: Path to input image
        output_dir: Directory to save output
        skip_existing: Skip if output already exists
        scale: Output resolution (longest edge)

    Returns:
        tuple: (success: bool, input_image_used: Path or None)
    """
    safe_name = preset["name"].lower().replace(" ", "_").replace("+", "_")
    output_image = output_dir / f"{safe_name}.jpg"

    # Skip if already exists
    if skip_existing and output_image.exists():
        print(f"  [SKIP] {preset['name']} - already exists")
        return True, None

    print(f"  [GEN] {preset['name']} using {input_image.name} @ {scale}p...")

    # Build command
    cmd = build_command(preset, input_image, output_image, scale=scale)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0 and output_image.exists():
            print(f"  [OK] {preset['name']} -> {output_image.name}")
            return True, input_image
        else:
            print(f"  [FAIL] {preset['name']}: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            return False, input_image

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {preset['name']}")
        return False, input_image
    except Exception as e:
        print(f"  [ERROR] {preset['name']}: {e}")
        return False, input_image


def generate_all_samples(input_image: Path = None, force: bool = False, randomize: bool = True, max_retries: int = MAX_RETRIES, regions_only: bool = False, include_regions: bool = True, category_filter: str = None, scale: int = SAMPLE_SCALE):
    """Generate samples for all presets with retry logic for failures.

    Args:
        input_image: Specific image to use (overrides randomization)
        force: Force regenerate all samples
        randomize: Use different random images for each preset (default True)
        max_retries: Maximum number of retries per preset if it fails
        regions_only: Only generate samples for Region presets
        include_regions: Include Region presets (default True, set False for non-region only)
        category_filter: Only generate samples for presets in this category (case-insensitive)
        scale: Output resolution (longest edge). Default 720 for consistent sample sizes.
    """
    # Ensure directories exist
    PRESET_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Note: With --force, we now regenerate and overwrite in place rather than
    # deleting the entire directory first. This preserves any files not in the
    # current preset list (e.g., manually added samples, different generators).
    # The skip_existing flag in generate_sample handles the per-file logic.

    # Collect available images (only from videos by default)
    if input_image is not None:
        # User specified a single image
        available_images = [input_image]
        randomize = False
        print(f"Using specified image: {input_image}")
    else:
        print("Extracting sample frames from videos only...")
        available_images = collect_all_sample_images(videos_only=True)

        if not available_images:
            print("ERROR: No sample images found!")
            print("Please add videos to /app/input_videos/")
            sys.exit(1)

        print(f"\nTotal available sample frames: {len(available_images)}")
        print("Mode: RANDOMIZE with RETRY - each preset uses a different frame, retries on failure")

    print(f"Output directory: {PRESET_SAMPLES_DIR}")
    print(f"Max retries per preset: {max_retries}")

    # Filter presets based on options
    # NOTE: Self Style presets are ALWAYS excluded from this generator because
    # they require special handling (center crop + thumbnail overlay) via
    # generate_style_selfstyle.py
    EXCLUDED_CATEGORIES = ["Self Style"]

    if category_filter:
        # Filter by specific category (case-insensitive)
        # Allow explicit Self Style category filter for debugging
        image_presets = [p for p in PRESETS if p["category"].lower() == category_filter.lower()]
        print(f"Generating {len(image_presets)} {category_filter} preset samples...\n")
    elif regions_only:
        # Only Region presets
        image_presets = [p for p in PRESETS if p["category"] == "Regions"]
        print(f"Generating {len(image_presets)} REGION preset samples...\n")
    elif include_regions:
        # All presets including Regions, but excluding Self Style
        image_presets = [p for p in PRESETS if p["category"] not in EXCLUDED_CATEGORIES]
        print(f"Generating {len(image_presets)} preset samples (excluding Self Style)...\n")
    else:
        # Exclude Region presets and Self Style
        image_presets = [p for p in PRESETS if p["category"] not in ["Regions"] + EXCLUDED_CATEGORIES]
        print(f"Generating {len(image_presets)} preset samples (excluding regions, Self Style)...\n")

    success_count = 0
    fail_count = 0
    used_images = set()
    failed_images = set()  # Track images that caused failures
    image_mapping = {}  # Track which image was used for each preset

    # Shuffle presets for variety if randomizing
    if randomize:
        random.shuffle(image_presets)

    for i, preset in enumerate(image_presets, 1):
        print(f"[{i}/{len(image_presets)}] {preset['name']}")

        # Try with different images until success or max retries
        success = False
        attempts = 0
        used_img = None

        while not success and attempts < max_retries:
            attempts += 1

            # Select an image, avoiding known problematic ones
            good_images = [img for img in available_images if img not in failed_images]
            if not good_images:
                # All images have failed at some point, try any unused image
                good_images = [img for img in available_images if img not in used_images]
            if not good_images:
                # Fall back to any image
                good_images = available_images

            if randomize:
                selected_image = get_random_image(good_images, used_images)
            else:
                selected_image = available_images[0]

            if attempts > 1:
                print(f"    [RETRY {attempts}/{max_retries}] Trying with {selected_image.name}...")

            success, used_img = generate_sample(preset, selected_image, PRESET_SAMPLES_DIR, skip_existing=not force, scale=scale)

            if not success and used_img:
                # Mark this image as problematic for this attempt
                failed_images.add(used_img)
                print(f"    [INFO] Marked {used_img.name} as problematic, will try another image")

        if success:
            success_count += 1
            if used_img:
                used_images.add(used_img)
                image_mapping[preset["name"]] = str(used_img.name)
                # Remove from failed if it succeeded (might have been a model issue)
                failed_images.discard(used_img)
        else:
            fail_count += 1
            print(f"    [FAILED] {preset['name']} after {max_retries} attempts")

    print(f"\n{'='*50}")
    print(f"Complete! Success: {success_count}, Failed: {fail_count}")

    if failed_images:
        print(f"\nProblematic images that caused failures:")
        for img in sorted(failed_images):
            print(f"  - {img.name}")

    # Save image mapping for reference
    if image_mapping:
        mapping_file = PRESET_SAMPLES_DIR / "image_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(image_mapping, f, indent=2)
        print(f"Saved image mapping to {mapping_file}")

    # Also save the preset metadata alongside samples
    save_preset_metadata()

    return success_count, fail_count


def save_preset_metadata():
    """Save preset metadata JSON for the web UI."""
    metadata = []
    for preset in PRESETS:
        safe_name = preset["name"].lower().replace(" ", "_").replace("+", "_")
        sample_image_path = PRESET_SAMPLES_DIR / f"{safe_name}.jpg"
        sample_gif_path = PRESET_SAMPLES_DIR / f"{safe_name}.gif"
        sample_video_path = PRESET_SAMPLES_DIR / f"{safe_name}.mp4"
        # Also check for morph videos
        sample_morph_path = PRESET_SAMPLES_DIR / f"{safe_name}_morph.mp4"

        # Prefer video over image for preview
        sample_video = None
        if sample_video_path.exists():
            sample_video = f"preset_samples/{safe_name}.mp4"
        elif sample_morph_path.exists():
            sample_video = f"preset_samples/{safe_name}_morph.mp4"

        # Determine sample image: prefer GIF (for morph), then JPG
        sample_image = None
        if sample_gif_path.exists():
            sample_image = f"preset_samples/{safe_name}.gif"
        elif sample_image_path.exists():
            sample_image = f"preset_samples/{safe_name}.jpg"

        metadata.append({
            "name": preset["name"],
            "description": preset["description"],
            "category": preset["category"],
            "tags": preset["tags"],
            "sample_image": sample_image,
            "sample_video": sample_video,
            "params": preset["params"],
        })

    # Save metadata
    metadata_file = PRESET_SAMPLES_DIR / "presets.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved preset metadata to {metadata_file}")

    # Also save individual preset files
    for preset in PRESETS:
        safe_name = preset["name"].lower().replace(" ", "_").replace("+", "_")
        preset_file = PRESETS_DIR / f"{safe_name}.json"
        with open(preset_file, "w") as f:
            json.dump(preset, f, indent=2)

    print(f"Saved {len(PRESETS)} individual preset files to {PRESETS_DIR}")


def copy_sample_to_output(input_image: Path):
    """Copy the original sample image to output for comparison."""
    output = PRESET_SAMPLES_DIR / "original.jpg"
    if not output.exists():
        # Convert to jpg if needed
        from PIL import Image
        img = Image.open(input_image)
        img = img.convert("RGB")
        # Resize to reasonable size for web
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        img.save(output, "JPEG", quality=90)
        print(f"Saved original sample to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate preset sample images")
    parser.add_argument("--input", "-i", type=str, help="Input sample image path (disables randomization)")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate all samples")
    parser.add_argument("--no-randomize", action="store_true", help="Use same image for all presets")
    parser.add_argument("--max-retries", "-r", type=int, default=MAX_RETRIES,
                        help=f"Max retries per preset on failure (default: {MAX_RETRIES})")
    parser.add_argument("--list", "-l", action="store_true", help="List all presets")
    parser.add_argument("--regions-only", action="store_true", help="Only generate samples for Region presets")
    parser.add_argument("--no-regions", action="store_true", help="Exclude Region presets (legacy behavior)")
    parser.add_argument("--category", "-c", type=str, help="Only generate samples for this category (e.g., Magenta, ReCoNet)")
    parser.add_argument("--scale", "-s", type=int, default=SAMPLE_SCALE,
                        help=f"Output resolution for samples (longest edge, default: {SAMPLE_SCALE})")

    args = parser.parse_args()

    if args.list:
        print(f"Total presets: {len(PRESETS)}\n")
        for i, preset in enumerate(PRESETS, 1):
            print(f"{i:2}. [{preset['category']:12}] {preset['name']}")
            print(f"     {preset['description']}")
            print(f"     Tags: {', '.join(preset['tags'])}")
            print()
        sys.exit(0)

    input_image = Path(args.input) if args.input else None
    randomize = not args.no_randomize
    include_regions = not args.no_regions
    generate_all_samples(
        input_image=input_image,
        force=args.force,
        randomize=randomize,
        max_retries=args.max_retries,
        regions_only=args.regions_only,
        include_regions=include_regions,
        category_filter=args.category,
        scale=args.scale
    )
