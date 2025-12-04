#!/usr/bin/env python3
"""
Generate video sample previews for presets.

Two types of previews:
1. Region presets: Process 2-3 seconds of video to show region morphing/animation
2. Regular presets: Create morph-style transition (original → styled → original)

Usage:
    docker-compose run --rm web bash -lc "python /web/generate_video_samples.py"
    docker-compose run --rm web bash -lc "python /web/generate_video_samples.py --regions-only"
    docker-compose run --rm web bash -lc "python /web/generate_video_samples.py --morph-only"
"""

import os
import sys
import subprocess
import shutil
import random
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from presets import PRESETS, PRESET_SAMPLES_DIR

# Paths (configurable via environment)
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", "/app"))
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))

PIPELINE = PIPELINE_DIR / "pipeline.py"
WORK_DIR = PIPELINE_DIR / "_work" / "video_samples"

# Input directories
INPUT_VIDEOS_DIR = PIPELINE_DIR / "input_videos"
INPUT_IMAGES_DIR = PIPELINE_DIR / "input" / "images"

# Video sample settings
SAMPLE_FRAMES = 20  # max frames to process
SAMPLE_SCALE = 480  # height for previews (smaller for faster generation)
# FPS is NOT hardcoded - let the pipeline use its defaults or preset settings
# This allows previews to showcase different frame rates (e.g. flow_ema effects)

# Morph preview timing
MORPH_FPS = 10  # fps for morph videos only (static image transitions)
MORPH_HOLD = 0.6  # seconds to hold each frame in morph
MORPH_TRANS = 0.4  # transition time in morph


def find_sample_video() -> Path:
    """Find a sample video to use for region previews."""
    if INPUT_VIDEOS_DIR.exists():
        for ext in ["*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV"]:
            videos = list(INPUT_VIDEOS_DIR.glob(ext))
            if videos:
                # Prefer shorter videos
                return random.choice(videos)
    return None


def find_sample_image() -> Path:
    """Find a sample image for morph previews."""
    if INPUT_IMAGES_DIR.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            images = list(INPUT_IMAGES_DIR.glob(ext))
            if images:
                return random.choice(images)

    # Check input/ directly
    input_dir = PIPELINE_DIR / "input"
    if input_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(input_dir.glob(ext))
            if images:
                return random.choice(images)

    return None


def extract_video_clip(video_path: Path, output_path: Path, max_frames: int = SAMPLE_FRAMES, start_pct: float = 0.2) -> bool:
    """Extract a short clip from a video starting at start_pct of the way through.

    Preserves the original fps - the pipeline will limit frames with --max_frames.
    """
    try:
        # Get video duration and fps
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=r_frame_rate",
            "-of", "json",
            str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False

        import json
        probe_data = json.loads(result.stdout)
        total_duration = float(probe_data["format"]["duration"])

        # Get fps from stream (format: "30/1" or "30000/1001")
        fps_str = probe_data["streams"][0]["r_frame_rate"] if probe_data.get("streams") else "24/1"
        fps_parts = fps_str.split("/")
        source_fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

        # Calculate duration needed for max_frames at source fps
        clip_duration = max_frames / source_fps
        start_time = total_duration * start_pct

        # Don't go past the end
        if start_time + clip_duration > total_duration:
            start_time = max(0, total_duration - clip_duration - 0.5)

        # Extract clip - preserve original fps, just scale
        extract_cmd = [
            "ffmpeg", "-y", "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(clip_duration),
            "-vf", f"scale=-2:{SAMPLE_SCALE}",
            "-c:v", "libx264", "-preset", "fast",
            "-an",  # No audio
            str(output_path)
        ]
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists()

    except Exception as e:
        print(f"    Error extracting clip: {e}")
        return False


def is_region_preset(preset: dict) -> bool:
    """Check if a preset uses region-based effects (needs video preview)."""
    params = preset.get("params", {})
    return any(k.startswith("region_") for k in params.keys())


def get_region_presets() -> list:
    """Get all presets that use region effects."""
    return [p for p in PRESETS if is_region_preset(p)]


def get_non_region_presets() -> list:
    """Get all presets that don't use region effects."""
    return [p for p in PRESETS if not is_region_preset(p)]


def build_pipeline_command(preset: dict, input_path: Path, output_path: Path, is_video: bool = True) -> list:
    """Build pipeline.py command from preset params."""
    params = preset["params"]

    cmd = ["python3", str(PIPELINE)]

    if is_video:
        cmd += ["--input_video", str(input_path)]
        cmd += ["--output_video", str(output_path)]
    else:
        cmd += ["--input_image", str(input_path)]
        cmd += ["--output_image", str(output_path)]

    cmd += ["--work_dir", str(WORK_DIR)]
    cmd += ["--scale", str(SAMPLE_SCALE)]
    # Don't hardcode fps - let pipeline use source fps or preset defaults
    # This allows flow_ema, blend, etc. to showcase their actual behavior
    if is_video:
        cmd += ["--max_frames", str(SAMPLE_FRAMES)]

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

    # Secondary models (B, C, D)
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

    # Region settings (the key ones for video)
    if params.get("region_mode"):
        cmd += ["--region_mode", params["region_mode"]]
    if params.get("region_count"):
        cmd += ["--region_count", str(params["region_count"])]
    if params.get("region_feather"):
        cmd += ["--region_feather", str(params["region_feather"])]
    if params.get("region_blend_spec"):
        cmd += ["--region_blend_spec", params["region_blend_spec"]]
    if params.get("region_morph"):
        cmd += ["--region_morph", params["region_morph"]]
    if params.get("region_rotate"):
        cmd += ["--region_rotate", str(params["region_rotate"])]

    # Blending
    if params.get("blend") is not None:
        cmd += ["--blend", str(params["blend"])]

    return cmd


def extract_first_frame(video_path: Path, image_path: Path) -> bool:
    """Extract the first frame from a video as a thumbnail."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(image_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and image_path.exists()
    except Exception as e:
        print(f"    Error extracting thumbnail: {e}")
        return False


def generate_region_preview(preset: dict, input_clip: Path, output_dir: Path, skip_existing: bool = True) -> bool:
    """Generate a video preview for a region preset."""
    safe_name = preset["name"].lower().replace(" ", "_").replace("+", "_")
    output_video = output_dir / f"{safe_name}.mp4"
    output_thumb = output_dir / f"{safe_name}.jpg"

    if skip_existing and output_video.exists() and output_thumb.exists():
        print(f"  [SKIP] {preset['name']} - already exists")
        return True

    print(f"  [GEN] {preset['name']} (region video)...")

    cmd = build_pipeline_command(preset, input_clip, output_video, is_video=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0 and output_video.exists():
            # Extract first frame as thumbnail
            if extract_first_frame(output_video, output_thumb):
                print(f"  [OK] {preset['name']} -> {output_video.name} + {output_thumb.name}")
            else:
                print(f"  [OK] {preset['name']} -> {output_video.name} (no thumbnail)")
            return True
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            print(f"  [FAIL] {preset['name']}: {error_msg}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {preset['name']}")
        return False
    except Exception as e:
        print(f"  [ERROR] {preset['name']}: {e}")
        return False


def generate_morph_preview(preset: dict, input_image: Path, output_dir: Path, skip_existing: bool = True) -> bool:
    """Generate a morph-style video preview (original → styled → original)."""
    safe_name = preset["name"].lower().replace(" ", "_").replace("+", "_")
    output_video = output_dir / f"{safe_name}_morph.mp4"
    output_thumb = output_dir / f"{safe_name}.jpg"  # Thumbnail matches the styled look

    if skip_existing and output_video.exists() and output_thumb.exists():
        print(f"  [SKIP] {preset['name']} morph - already exists")
        return True

    print(f"  [GEN] {preset['name']} (morph video)...")

    work = WORK_DIR / f"morph_{safe_name}"
    work.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate styled image
        styled_image = work / "styled.jpg"
        cmd = build_pipeline_command(preset, input_image, styled_image, is_video=False)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not styled_image.exists():
            print(f"  [FAIL] {preset['name']}: Could not generate styled image")
            return False

        # Copy styled image as thumbnail (so thumbnail matches the styled part of the morph)
        shutil.copy(styled_image, output_thumb)

        # Step 2: Create original still clip
        orig_clip = work / "orig.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1",
            "-t", str(MORPH_HOLD + MORPH_TRANS),
            "-r", str(MORPH_FPS),
            "-i", str(input_image),
            "-vf", f"scale=-2:{SAMPLE_SCALE},format=yuv420p",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(orig_clip)
        ], capture_output=True, timeout=60)

        # Step 3: Create styled still clip
        styled_clip = work / "styled.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1",
            "-t", str(MORPH_HOLD + MORPH_TRANS),
            "-r", str(MORPH_FPS),
            "-i", str(styled_image),
            "-vf", f"scale=-2:{SAMPLE_SCALE},format=yuv420p",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(styled_clip)
        ], capture_output=True, timeout=60)

        # Step 4: Create second original clip for the return transition
        orig2_clip = work / "orig2.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1",
            "-t", str(MORPH_HOLD),
            "-r", str(MORPH_FPS),
            "-i", str(input_image),
            "-vf", f"scale=-2:{SAMPLE_SCALE},format=yuv420p",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(orig2_clip)
        ], capture_output=True, timeout=60)

        # Step 5: Crossfade orig → styled
        fade1 = work / "fade1.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(orig_clip),
            "-i", str(styled_clip),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={MORPH_TRANS}:offset={MORPH_HOLD},format=yuv420p[v]",
            "-map", "[v]",
            "-r", str(MORPH_FPS),
            str(fade1)
        ], capture_output=True, timeout=60)

        # Step 6: Crossfade (orig→styled) → orig2
        fade1_dur = MORPH_HOLD * 2 + MORPH_TRANS  # approximate
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(fade1),
            "-i", str(orig2_clip),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={MORPH_TRANS}[a];"
            f"[1:v]tpad=stop_mode=clone:stop_duration={MORPH_TRANS}[b];"
            f"[a][b]xfade=transition=fade:duration={MORPH_TRANS}:offset={MORPH_HOLD + MORPH_TRANS + MORPH_HOLD - 0.1},format=yuv420p[v]",
            "-map", "[v]",
            "-r", str(MORPH_FPS),
            str(output_video)
        ], capture_output=True, timeout=60)

        if output_video.exists():
            print(f"  [OK] {preset['name']} -> {output_video.name} + {output_thumb.name}")
            # Cleanup work dir
            shutil.rmtree(work, ignore_errors=True)
            return True
        else:
            print(f"  [FAIL] {preset['name']}: Final video not created")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {preset['name']}")
        return False
    except Exception as e:
        print(f"  [ERROR] {preset['name']}: {e}")
        return False


def generate_region_samples(force: bool = False) -> tuple:
    """Generate video samples for all region presets."""
    PRESET_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Find sample video
    sample_video = find_sample_video()
    if not sample_video:
        print("ERROR: No sample video found!")
        print("Please place a video in /app/input_videos/")
        return 0, 0

    print(f"Using sample video: {sample_video}")

    # Extract a short clip for processing (enough frames for the preview)
    input_clip = WORK_DIR / "sample_clip.mp4"
    print(f"Extracting clip with ~{SAMPLE_FRAMES} frames...")
    if not extract_video_clip(sample_video, input_clip, max_frames=SAMPLE_FRAMES):
        print("ERROR: Failed to extract video clip")
        return 0, 0

    region_presets = get_region_presets()
    print(f"Generating {len(region_presets)} region preset videos...\n")

    success_count = 0
    fail_count = 0

    for i, preset in enumerate(region_presets, 1):
        print(f"[{i}/{len(region_presets)}] {preset['name']}")
        if generate_region_preview(preset, input_clip, PRESET_SAMPLES_DIR, skip_existing=not force):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Region samples complete! Success: {success_count}, Failed: {fail_count}")
    return success_count, fail_count


def generate_morph_samples(force: bool = False, limit: int = 0) -> tuple:
    """Generate morph-style video samples for non-region presets."""
    PRESET_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Find sample image
    sample_image = find_sample_image()
    if not sample_image:
        print("ERROR: No sample image found!")
        print("Please place an image in /app/input/images/")
        return 0, 0

    print(f"Using sample image: {sample_image}")

    non_region_presets = get_non_region_presets()
    if limit > 0:
        non_region_presets = non_region_presets[:limit]

    print(f"Generating {len(non_region_presets)} morph videos...\n")

    success_count = 0
    fail_count = 0

    for i, preset in enumerate(non_region_presets, 1):
        print(f"[{i}/{len(non_region_presets)}] {preset['name']}")
        if generate_morph_preview(preset, sample_image, PRESET_SAMPLES_DIR, skip_existing=not force):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Morph samples complete! Success: {success_count}, Failed: {fail_count}")
    return success_count, fail_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate video sample previews")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate all samples")
    parser.add_argument("--regions-only", action="store_true", help="Only generate region preset videos")
    parser.add_argument("--morph-only", action="store_true", help="Only generate morph videos")
    parser.add_argument("--morph-limit", type=int, default=0, help="Limit number of morph videos to generate")
    parser.add_argument("--list", "-l", action="store_true", help="List presets by type")

    args = parser.parse_args()

    if args.list:
        region_presets = get_region_presets()
        non_region_presets = get_non_region_presets()

        print(f"Region presets ({len(region_presets)}):")
        for p in region_presets:
            print(f"  - {p['name']}")

        print(f"\nNon-region presets ({len(non_region_presets)}):")
        for p in non_region_presets:
            print(f"  - {p['name']}")

        sys.exit(0)

    total_success = 0
    total_fail = 0

    if args.morph_only:
        s, f = generate_morph_samples(force=args.force, limit=args.morph_limit)
        total_success += s
        total_fail += f
    elif args.regions_only:
        s, f = generate_region_samples(force=args.force)
        total_success += s
        total_fail += f
    else:
        # Generate both
        print("=" * 60)
        print("PHASE 1: Region preset videos")
        print("=" * 60)
        s, f = generate_region_samples(force=args.force)
        total_success += s
        total_fail += f

        print("\n")
        print("=" * 60)
        print("PHASE 2: Morph preview videos")
        print("=" * 60)
        s, f = generate_morph_samples(force=args.force, limit=args.morph_limit)
        total_success += s
        total_fail += f

    print(f"\n{'='*60}")
    print(f"TOTAL: Success: {total_success}, Failed: {total_fail}")
    print(f"Videos saved to: {PRESET_SAMPLES_DIR}")
