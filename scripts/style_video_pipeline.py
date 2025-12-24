#!/usr/bin/env python3
"""
Full pipeline for styling a video with multi-model blend.
Extracts frames, styles with udnie/mosaic/tenharmsel, creates walk files, composes video.

Run inside Docker container:
  docker compose run --rm -v /path/to/project:/project style \
    python3 /app/scripts/style_video_pipeline.py \
    --video /project/input.mp4 \
    --output_dir /project/output \
    --output_name my_video
"""

import subprocess
import json
import argparse
import sys
from pathlib import Path
import shutil

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_model_path(model_name):
    """Get model path inside container."""
    # Check tenharmsel models
    tenharmsel_path = Path(f"/app/models/tenharmsel/{model_name}.pth")
    if tenharmsel_path.exists():
        return str(tenharmsel_path)

    # Check pytorch models
    pytorch_path = Path(f"/app/models/pytorch/{model_name}.pth")
    if pytorch_path.exists():
        return str(pytorch_path)

    # Check base models
    base_path = Path(f"/app/models/{model_name}.pth")
    if base_path.exists():
        return str(base_path)

    return None

def run_style(input_path, output_path, model_name, scale=1080):
    """Run single style operation using pipeline.py."""
    model_path = get_model_path(model_name)
    if not model_path:
        print(f"      Model not found: {model_name}")
        return False

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "/app/pipeline.py",
        "--input_image", str(input_path),
        "--output_image", str(output_path),
        "--model_type", "transformer",
        "--model", model_path,
        "--io_preset", "auto",
        "--scale", str(scale),
        "--inference_res", "720",
        "--blend", "0.9",
        "--device", "cuda",
        "--clean_work_dir"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def extract_frames(video_path, output_dir, fps=8):
    """Extract frames from video at given fps using ffmpeg."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Extracting frames at {fps}fps ===")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_dir / "frame_%04d_original.jpg")
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return 0

    frame_count = len(list(output_dir.glob("frame_*_original.jpg")))
    print(f"  Extracted {frame_count} frames")
    return frame_count

def style_frames(input_dir, output_dir, style_name, weights, frame_start, frame_end, scale=1080):
    """Style frames with all weights for a style."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = frame_end - frame_start + 1
    total_ops = total_frames * len(weights)
    completed = 0

    print(f"\n=== Styling {style_name}: frames {frame_start}-{frame_end} ===")
    print(f"  {total_frames} frames x {len(weights)} weights = {total_ops} operations")
    print(f"  Output: {output_dir}")

    for frame_num in range(frame_start, frame_end + 1):
        frame_name = f"frame_{frame_num:04d}"
        input_path = input_dir / f"{frame_name}_original.jpg"

        if not input_path.exists():
            print(f"  Missing input: {input_path}")
            continue

        # Copy original to output dir for blending
        orig_dest = output_dir / f"{frame_name}_original.jpg"
        if not orig_dest.exists():
            shutil.copy(input_path, orig_dest)

        for weight in weights:
            output_path = output_dir / f"{frame_name}_{weight}.jpg"

            if output_path.exists():
                completed += 1
                continue

            completed += 1
            print(f"  [{completed}/{total_ops}] {frame_name} -> {weight}")

            success = run_style(input_path, output_path, weight, scale)
            if not success:
                print(f"      FAILED: {frame_name} {weight}")

    print(f"  Done! {completed}/{total_ops} operations")

def create_walk_file(output_dir, style_name, weights, frame_start, frame_end):
    """Create walk.json file for a style."""
    output_dir = Path(output_dir)
    num_frames = frame_end - frame_start + 1

    # Simple walk - all zeros for single weight, or random walk for multiple
    if len(weights) == 1:
        walk = [0] * num_frames
    else:
        # Create a gentle walk through weights
        import random
        walk = []
        pos = len(weights) // 2  # Start in middle
        for _ in range(num_frames):
            walk.append(pos)
            # Random walk with bounds
            pos += random.choice([-1, 0, 0, 1])  # Bias toward staying
            pos = max(0, min(len(weights) - 1, pos))

    walk_data = {
        "walk": walk,
        "weights": weights,
        "frame_start": frame_start,
        "frame_end": frame_end
    }

    walk_file = output_dir / f"walk_{style_name}.json"
    with open(walk_file, 'w') as f:
        json.dump(walk_data, f)

    print(f"  Created {walk_file}")

def run_pipeline(
    video_path,
    output_dir,
    output_name,
    styled_start=17,       # First styled frame (skip 2s at 8fps = 16 frames)
    styled_end=160,        # Last styled frame (20s at 8fps = 160)
    fps_extract=8,
    size=1080,
    include_tenharmsel=True,
    udnie_weights=None,
    mosaic_weights=None,
    tenharmsel_weights=None
):
    """Run the full styling pipeline."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Output directories
    base_output = output_dir / output_name
    frames_dir = base_output / "frames"
    styled_udnie = base_output / "styled_udnie"
    styled_mosaic = base_output / "styled_mosaic"
    styled_tenharmsel = base_output / "styled_tenharmsel" if include_tenharmsel else None

    # Default weights
    if udnie_weights is None:
        udnie_weights = [f"udnie_style{i}e10" for i in range(5, 10)]  # 5-9 for stronger effect
    if mosaic_weights is None:
        mosaic_weights = ["mosaic_style5e10"]
    if tenharmsel_weights is None:
        tenharmsel_weights = [f"tenharmsel_style{i}e10" for i in range(1, 10)]

    print(f"\n{'='*60}")
    print(f"MULTI-MODEL STYLE PIPELINE")
    print(f"{'='*60}")
    print(f"  Video: {video_path}")
    print(f"  Output: {base_output}")
    print(f"  Styled frames: {styled_start}-{styled_end}")
    print(f"{'='*60}")

    # Step 1: Extract frames
    frame_count = extract_frames(video_path, frames_dir, fps_extract)
    if frame_count == 0:
        print("ERROR: No frames extracted")
        return False

    # Adjust styled_end if video is shorter
    if styled_end > frame_count:
        styled_end = frame_count
        print(f"  Adjusted styled_end to {styled_end} (video length)")

    # Step 2: Style with udnie
    style_frames(frames_dir, styled_udnie, "udnie", udnie_weights, styled_start, styled_end, size)
    create_walk_file(styled_udnie, "udnie", udnie_weights, styled_start, styled_end)

    # Step 3: Style with mosaic
    style_frames(frames_dir, styled_mosaic, "mosaic", mosaic_weights, styled_start, styled_end, size)
    create_walk_file(styled_mosaic, "mosaic", mosaic_weights, styled_start, styled_end)

    # Step 4: Style with tenharmsel (optional)
    if include_tenharmsel:
        style_frames(frames_dir, styled_tenharmsel, "tenharmsel", tenharmsel_weights, styled_start, styled_end, size)
        create_walk_file(styled_tenharmsel, "tenharmsel", tenharmsel_weights, styled_start, styled_end)

    # Step 5: Copy original frames for intro/outro blending
    print("\n=== Copying original frames for blending ===")
    for frame_num in range(1, styled_start):
        frame_name = f"frame_{frame_num:04d}"
        src = frames_dir / f"{frame_name}_original.jpg"
        dst = styled_udnie / f"{frame_name}_original.jpg"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    for frame_num in range(styled_end + 1, frame_count + 1):
        frame_name = f"frame_{frame_num:04d}"
        src = frames_dir / f"{frame_name}_original.jpg"
        dst = styled_udnie / f"{frame_name}_original.jpg"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    print(f"\n{'='*60}")
    print("STYLING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTo create video, run:")
    print(f"  python3 /app/scripts/multi_model_video.py \\")
    print(f"    --orig_dir {frames_dir} \\")
    print(f"    --styled_udnie {styled_udnie} \\")
    print(f"    --styled_mosaic {styled_mosaic} \\")
    if include_tenharmsel:
        print(f"    --styled_tenharmsel {styled_tenharmsel} \\")
    print(f"    --output {output_dir}/{output_name}.mp4 \\")
    print(f"    --styled_start {styled_start} --styled_end {styled_end}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model video styling pipeline")
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--output_name', required=True, help='Output folder/video name')
    parser.add_argument('--styled_start', type=int, default=17, help='First frame to style (default: 17 = skip 2s at 8fps)')
    parser.add_argument('--styled_end', type=int, default=160, help='Last frame to style (default: 160 = 20s at 8fps)')
    parser.add_argument('--fps_extract', type=int, default=8, help='FPS for frame extraction')
    parser.add_argument('--size', type=int, default=1080, help='Output size')
    parser.add_argument('--no_tenharmsel', action='store_true', help='Skip tenharmsel styling')
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        output_name=args.output_name,
        styled_start=args.styled_start,
        styled_end=args.styled_end,
        fps_extract=args.fps_extract,
        size=args.size,
        include_tenharmsel=not args.no_tenharmsel
    )
