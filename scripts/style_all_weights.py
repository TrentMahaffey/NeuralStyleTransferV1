#!/usr/bin/env python3
"""
Style frames with ALL weights for a given model.
Ensures smooth ramping is possible since every weight is available for each frame.

Run inside Docker container:
  docker compose run --rm -v /path/to/project:/project style \
    python3 /app/scripts/style_all_weights.py \
    --style udnie \
    --frame_start 17 --frame_end 160 \
    --input_dir /project/frames \
    --output_dir /project/styled_udnie
"""

import subprocess
from pathlib import Path
import argparse
import shutil

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

def style_all_weights(style_name, weights, frame_start, frame_end, input_dir, output_dir, scale=1080):
    """Style all frames with all weights."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = frame_end - frame_start + 1
    total_ops = total_frames * len(weights)
    completed = 0

    print(f"\nStyling {style_name}: frames {frame_start}-{frame_end}")
    print(f"  {total_frames} frames x {len(weights)} weights = {total_ops} operations")
    print(f"  Output: {output_dir}\n")

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

    print(f"\nDone! {completed}/{total_ops} operations completed")

# Define weight sets for each style
WEIGHT_SETS = {
    'udnie': [f"udnie_style{i}e10" for i in range(1, 10)],
    'udnie_strong': [f"udnie_style{i}e10" for i in range(5, 10)],  # Only stronger weights
    'udnie_e11': [f"udnie_style{i}e10" for i in range(5, 10)] + [f"udnie_style{i}e11" for i in range(1, 6)],  # e10 strong + e11
    'candy': [f"candy_style{i}e10" for i in range(1, 10)],
    'mosaic': ["mosaic_style5e10"],  # Only style5e10 available
    'tenharmsel': [f"tenharmsel_style{i}e10" for i in range(1, 10)],
    'tenharmsel_strong': [f"tenharmsel_style{i}e10" for i in range(5, 10)],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style frames with all weights for a model")
    parser.add_argument('--style', required=True, help=f'Style name: {", ".join(WEIGHT_SETS.keys())}')
    parser.add_argument('--frame_start', type=int, required=True, help='First frame number')
    parser.add_argument('--frame_end', type=int, required=True, help='Last frame number')
    parser.add_argument('--input_dir', required=True, help='Directory with original frames')
    parser.add_argument('--output_dir', required=True, help='Output directory for styled frames')
    parser.add_argument('--scale', type=int, default=1080, help='Output resolution')
    args = parser.parse_args()

    if args.style not in WEIGHT_SETS:
        print(f"Unknown style: {args.style}")
        print(f"Available: {list(WEIGHT_SETS.keys())}")
        exit(1)

    weights = WEIGHT_SETS[args.style]

    style_all_weights(
        style_name=args.style,
        weights=weights,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scale=args.scale
    )
