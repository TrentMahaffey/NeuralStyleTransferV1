#!/usr/bin/env python3
"""
Generate animated GIF samples for Morph presets.

Morph presets animate the boundaries between style regions over time,
so they need animated previews rather than static images.

Usage:
    docker-compose run --rm web bash -lc "python /web/generate_morph_samples.py"
    docker-compose run --rm web bash -lc "python /web/generate_morph_samples.py --force"
    docker-compose run --rm web bash -lc "python /web/generate_morph_samples.py --list"
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
import argparse

# Paths
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", "/app"))
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))

PIPELINE = PIPELINE_DIR / "pipeline.py"
WORK_DIR = PIPELINE_DIR / "_work" / "morph_samples"

# Input/Output directories
INPUT_DIR = PIPELINE_DIR / "input"
INPUT_VIDEOS_DIR = PIPELINE_DIR / "input_videos"
OUTPUT_DIR = WEB_DIR / "static" / "preset_samples"
PRESETS_JSON = OUTPUT_DIR / "presets.json"

# GIF settings
GIF_FRAMES = 30  # Number of frames to generate
GIF_SCALE = 320  # Width for GIF preview
GIF_FPS = 10  # Frames per second for GIF

# Morph presets to generate samples for
MORPH_PRESETS = [
    {
        "name": "Spiral Dreams",
        "output_name": "spiral_dreams",
        "params": {
            "model": "/app/models/pytorch/rain_princess.pth",
            "model_type": "transformer",
            "io_preset": "raw_255",
            "model_b": "/app/models/torch/starry_night_eccv16.t7",
            "model_b_type": "torch7",
            "model_c": "/app/models/pytorch/udnie.pth",
            "model_c_type": "transformer",
            "io_preset_c": "raw_255",
            "region_mode": "spiral",
            "region_count": 3,
            "region_feather": 40,
            "region_blend_spec": "A|B|C",
            "region_morph": "blob",
            "blend": 1.0
        }
    },
    {
        "name": "Wave Morph",
        "output_name": "wave_morph",
        "params": {
            "model": "/app/models/pytorch/rain_princess.pth",
            "model_type": "transformer",
            "io_preset": "raw_255",
            "model_b": "/app/models/torchAll/the_wave_eccv16.t7",
            "model_b_type": "torch7",
            "region_mode": "waves",
            "region_count": 2,
            "region_feather": 60,
            "region_blend_spec": "A|B",
            "region_morph": "tentacle",
            "blend": 1.0
        }
    },
    {
        "name": "Blob Morph Trio",
        "output_name": "blob_morph_trio",
        "params": {
            "model": "/app/models/pytorch/candy.pth",
            "model_type": "transformer",
            "io_preset": "raw_255",
            "model_b": "/app/models/torch/starry_night_eccv16.t7",
            "model_b_type": "torch7",
            "model_c": "/app/models/pytorch/mosaic.pth",
            "model_c_type": "transformer",
            "io_preset_c": "raw_255",
            "region_mode": "voronoi",
            "region_count": 3,
            "region_feather": 50,
            "region_blend_spec": "A|B|C",
            "region_morph": "blob",
            "blend": 0.95
        }
    },
    {
        "name": "Tentacle Morph Quad",
        "output_name": "tentacle_morph_quad",
        "params": {
            "model": "/app/models/pytorch/rain_princess.pth",
            "model_type": "transformer",
            "io_preset": "raw_255",
            "model_b": "/app/models/torch/la_muse_eccv16.t7",
            "model_b_type": "torch7",
            "model_c": "/app/models/pytorch/udnie.pth",
            "model_c_type": "transformer",
            "io_preset_c": "raw_255",
            "model_d": "/app/models/torch/composition_vii_eccv16.t7",
            "model_d_type": "torch7",
            "region_mode": "voronoi",
            "region_count": 4,
            "region_feather": 40,
            "region_blend_spec": "A|B|C|D",
            "region_morph": "1.0,0.15,3.0,tentacle",
            "blend": 0.9
        }
    },
    {
        "name": "Wave Morph Duo",
        "output_name": "wave_morph_duo",
        "params": {
            "model": "magenta",
            "model_type": "magenta",
            "magenta_style": "/app/models/magenta_styles/canyon.jpg",
            "model_b": "magenta",
            "model_b_type": "magenta",
            "magenta_style_b": "/app/models/magenta_styles/style_rainforest.jpg",
            "magenta_tile": 512,
            "magenta_overlap": 64,
            "region_mode": "voronoi",
            "region_count": 2,
            "region_feather": 60,
            "region_blend_spec": "A|B",
            "region_morph": "wave",
            "blend": 0.9
        }
    },
    {
        "name": "Pulse Morph Center",
        "output_name": "pulse_morph_center",
        "params": {
            "model": "/app/models/pytorch/candy.pth",
            "model_type": "transformer",
            "io_preset": "raw_255",
            "model_b": "/app/models/torch/the_scream.t7",
            "model_b_type": "torch7",
            "model_c": "/app/models/pytorch/mosaic.pth",
            "model_c_type": "transformer",
            "io_preset_c": "raw_255",
            "region_mode": "radial",
            "region_count": 3,
            "region_feather": 45,
            "region_blend_spec": "A|B|C",
            "region_morph": "pulse",
            "blend": 0.9
        }
    },
]


def find_input_image() -> Path:
    """Find a sample image to use."""
    # Look for sample image
    sample = INPUT_DIR / "sample.jpg"
    if sample.exists():
        return sample

    # Try mask_samples directory
    mask_samples = INPUT_DIR / "mask_samples"
    if mask_samples.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(mask_samples.glob(ext))
            if images:
                return images[0]

    # Try input/images
    images_dir = INPUT_DIR / "images"
    if images_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(images_dir.glob(ext))
            if images:
                return images[0]

    # Try input directory directly
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images = list(INPUT_DIR.glob(ext))
        if images:
            return images[0]

    return None


def find_input_video() -> Path:
    """Find a sample video to use."""
    if INPUT_VIDEOS_DIR.exists():
        for ext in ["*.mp4", "*.avi", "*.mov", "*.MP4"]:
            videos = list(INPUT_VIDEOS_DIR.glob(ext))
            if videos:
                return videos[0]
    return None


def create_synthetic_video(image_path: Path, output_path: Path, num_frames: int = GIF_FRAMES) -> bool:
    """Create a synthetic video from an image (static frames for morph animation)."""
    try:
        # Create a short video from static image - morph will animate the boundaries
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-t", str(num_frames / GIF_FPS),
            "-r", str(GIF_FPS),
            "-vf", f"scale={GIF_SCALE}:-2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"    Error creating synthetic video: {e}")
        return False


def extract_video_clip(video_path: Path, output_path: Path, num_frames: int = GIF_FRAMES) -> bool:
    """Extract a short clip from a video."""
    try:
        duration = num_frames / GIF_FPS
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-t", str(duration),
            "-r", str(GIF_FPS),
            "-vf", f"scale={GIF_SCALE}:-2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"    Error extracting clip: {e}")
        return False


def video_to_gif(video_path: Path, gif_path: Path) -> bool:
    """Convert video to optimized GIF."""
    try:
        # Create palette for better GIF quality
        palette = video_path.parent / "palette.png"

        # Generate palette
        palette_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={GIF_FPS},scale={GIF_SCALE}:-1:flags=lanczos,palettegen",
            str(palette)
        ]
        subprocess.run(palette_cmd, capture_output=True, timeout=60)

        # Create GIF with palette
        if palette.exists():
            gif_cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(palette),
                "-lavfi", f"fps={GIF_FPS},scale={GIF_SCALE}:-1:flags=lanczos[x];[x][1:v]paletteuse",
                str(gif_path)
            ]
        else:
            # Fallback without palette
            gif_cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"fps={GIF_FPS},scale={GIF_SCALE}:-1:flags=lanczos",
                str(gif_path)
            ]

        result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=120)

        # Cleanup palette
        if palette.exists():
            palette.unlink()

        return result.returncode == 0 and gif_path.exists()
    except Exception as e:
        print(f"    Error converting to GIF: {e}")
        return False


def build_pipeline_command(params: dict, input_video: Path, output_video: Path) -> list:
    """Build pipeline.py command from preset params."""
    cmd = [
        "python3", str(PIPELINE),
        "--input_video", str(input_video),
        "--output_video", str(output_video),
        "--work_dir", str(WORK_DIR),
        "--scale", str(GIF_SCALE),
        "--max_frames", str(GIF_FRAMES),
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

    # Region settings
    if params.get("region_mode"):
        cmd += ["--region_mode", params["region_mode"]]
    if params.get("region_count"):
        cmd += ["--region_count", str(params["region_count"])]
    if params.get("region_feather"):
        cmd += ["--region_feather", str(params["region_feather"])]
    if params.get("region_blend_spec"):
        cmd += ["--region_blend_spec", params["region_blend_spec"]]
    if params.get("region_morph"):
        cmd += ["--region_morph", str(params["region_morph"])]

    # Blending
    if params.get("blend") is not None:
        cmd += ["--blend", str(params["blend"])]

    return cmd


def generate_morph_sample(preset: dict, input_video: Path, force: bool = False) -> bool:
    """Generate a GIF sample for a morph preset."""
    output_name = preset["output_name"]
    output_gif = OUTPUT_DIR / f"{output_name}.gif"

    if not force and output_gif.exists():
        print(f"  [SKIP] {preset['name']} - already exists")
        return True

    print(f"  [GEN] {preset['name']}...")

    work = WORK_DIR / output_name
    work.mkdir(parents=True, exist_ok=True)

    try:
        # Run pipeline to generate styled video
        output_video = work / "styled.mp4"
        cmd = build_pipeline_command(preset["params"], input_video, output_video)

        print(f"    Running pipeline...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0 or not output_video.exists():
            error_msg = result.stderr[:300] if result.stderr else "Unknown error"
            print(f"  [FAIL] {preset['name']}: {error_msg}")
            return False

        # Convert to GIF
        print(f"    Converting to GIF...")
        if video_to_gif(output_video, output_gif):
            size_kb = output_gif.stat().st_size / 1024
            print(f"  [OK] {preset['name']} -> {output_gif.name} ({size_kb:.1f} KB)")

            # Cleanup work directory
            shutil.rmtree(work, ignore_errors=True)
            return True
        else:
            print(f"  [FAIL] {preset['name']}: GIF conversion failed")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {preset['name']}")
        return False
    except Exception as e:
        print(f"  [ERROR] {preset['name']}: {e}")
        return False


def update_presets_json():
    """Update presets.json to point to the GIF files."""
    if not PRESETS_JSON.exists():
        print("Warning: presets.json not found")
        return

    with open(PRESETS_JSON, "r") as f:
        presets = json.load(f)

    # Map preset names to output files
    morph_files = {p["name"]: p["output_name"] for p in MORPH_PRESETS}

    updated = 0
    for preset in presets:
        if preset["name"] in morph_files:
            output_name = morph_files[preset["name"]]
            gif_file = OUTPUT_DIR / f"{output_name}.gif"

            if gif_file.exists():
                preset["sample_image"] = f"preset_samples/{output_name}.gif"
                updated += 1
                print(f"  Updated {preset['name']} -> {output_name}.gif")

    if updated > 0:
        with open(PRESETS_JSON, "w") as f:
            json.dump(presets, f, indent=2)
        print(f"\nUpdated {updated} presets in presets.json")


def main():
    parser = argparse.ArgumentParser(description="Generate morph preset GIF samples")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate all samples")
    parser.add_argument("--list", "-l", action="store_true", help="List morph presets")
    parser.add_argument("--preset", "-p", type=str, help="Generate only specific preset by name")
    args = parser.parse_args()

    if args.list:
        print("Morph presets:")
        for p in MORPH_PRESETS:
            print(f"  - {p['name']} ({p['output_name']})")
        return 0

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Find input source
    input_video = find_input_video()
    input_image = find_input_image()

    # Prepare input video
    sample_video = WORK_DIR / "sample_input.mp4"

    if input_video:
        print(f"Using input video: {input_video}")
        if not extract_video_clip(input_video, sample_video):
            print("ERROR: Failed to extract video clip")
            return 1
    elif input_image:
        print(f"Using input image: {input_image} (creating synthetic video)")
        if not create_synthetic_video(input_image, sample_video):
            print("ERROR: Failed to create synthetic video from image")
            return 1
    else:
        print("ERROR: No input video or image found!")
        print("Please place a video in /app/input_videos/ or an image in /app/input/")
        return 1

    # Filter presets if specific one requested
    presets_to_generate = MORPH_PRESETS
    if args.preset:
        presets_to_generate = [p for p in MORPH_PRESETS if args.preset.lower() in p["name"].lower()]
        if not presets_to_generate:
            print(f"No preset matching '{args.preset}' found")
            return 1

    print(f"\nGenerating {len(presets_to_generate)} morph GIF samples...\n")

    success = 0
    failed = 0

    for i, preset in enumerate(presets_to_generate, 1):
        print(f"[{i}/{len(presets_to_generate)}] {preset['name']}")
        if generate_morph_sample(preset, sample_video, force=args.force):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Complete! Success: {success}, Failed: {failed}")

    # Update presets.json with new GIF paths
    if success > 0:
        print("\nUpdating presets.json...")
        update_presets_json()

    print(f"\nGIF samples saved to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
