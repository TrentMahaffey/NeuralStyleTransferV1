#!/usr/bin/env python3
"""
Style Showcase - Generate morph videos showcasing neural style transfers.

Takes input images and creates a video that transitions through:
  original → each style → style blends → original

With optional motion effects: zoom in/out, pan left/right/up/down, ken burns.

Usage:
    docker-compose run --rm style bash -lc "python /app/web/style_showcase.py"
    docker-compose run --rm style bash -lc "python /app/web/style_showcase.py --motion zoom_in"
    docker-compose run --rm style bash -lc "python /app/web/style_showcase.py --motion ken_burns --styles candy,mosaic"

Environment variables:
    IN_DIR          Input directory (default: /app/input)
    OUT_DIR         Output directory (default: /app/output)
    SCALE           Output height (default: 720)
    FPS             Frames per second (default: 24)
    HOLD_MODEL      Seconds to hold each style (default: 1.5)
    HOLD_ORIG       Seconds to hold original (default: 2.0)
    TRANS           Transition duration (default: 1.0)
    TRANSITION      FFmpeg xfade transition (default: fade)
    MAX_MODELS      Max styles per image (default: 10)
    MOTION          Motion effect: none, zoom_in, zoom_out, pan_left, pan_right, ken_burns
"""

import os
import sys
import json
import subprocess
import shutil
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

# Paths
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", "/app"))
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))
PIPELINE = PIPELINE_DIR / "pipeline.py"


@dataclass
class Config:
    """Configuration for style showcase generation."""
    in_dir: Path = Path(os.environ.get("IN_DIR", "/app/input"))
    out_dir: Path = Path(os.environ.get("OUT_DIR", "/app/output"))
    work_dir: Path = Path(os.environ.get("TMP", "/app/_work/showcase"))

    scale: int = int(os.environ.get("SCALE", "720"))
    fps: int = int(os.environ.get("FPS", "24"))

    hold_model: float = float(os.environ.get("HOLD_MODEL", "1.5"))
    hold_orig_start: float = float(os.environ.get("HOLD_ORIG_START", "2.0"))
    hold_orig_end: float = float(os.environ.get("HOLD_ORIG_END", "2.0"))
    trans_duration: float = float(os.environ.get("TRANS", "1.0"))
    transition: str = os.environ.get("TRANSITION", "fade")

    max_models: int = int(os.environ.get("MAX_MODELS", "10"))
    max_blends: int = int(os.environ.get("MAX_BLENDS", "5"))
    include_blends: bool = os.environ.get("INCLUDE_BLENDS", "1") == "1"

    motion: str = os.environ.get("MOTION", "none")
    motion_strength: float = float(os.environ.get("MOTION_STRENGTH", "0.1"))

    jpeg_quality: int = int(os.environ.get("IMG_Q", "90"))
    crf: int = int(os.environ.get("CRF", "18"))
    preset: str = os.environ.get("PRESET", "slow")

    magenta_style_dir: Path = Path(os.environ.get("MAGENTA_STYLE_DIR", "/app/models/magenta_styles"))
    magenta_tile: int = int(os.environ.get("MAGENTA_TILE", "512"))
    magenta_overlap: int = int(os.environ.get("MAGENTA_OVERLAP", "64"))


@dataclass
class StyleModel:
    """A style model definition."""
    name: str
    model_type: str  # transformer, torch7, magenta
    path: str
    io_preset: str = "imagenet_255"


# Available style models
STYLE_MODELS: Dict[str, StyleModel] = {
    # PyTorch Transformer models
    "candy": StyleModel("candy", "transformer", "/app/models/pytorch/candy.pth"),
    "mosaic": StyleModel("mosaic", "transformer", "/app/models/pytorch/mosaic.pth"),
    "udnie": StyleModel("udnie", "transformer", "/app/models/pytorch/udnie.pth"),
    "rain_princess": StyleModel("rain_princess", "transformer", "/app/models/pytorch/rain_princess.pth"),

    # Torch7 models
    "composition_vii": StyleModel("composition_vii", "torch7", "/app/models/torch/composition_vii_eccv16.t7"),
    "la_muse": StyleModel("la_muse", "torch7", "/app/models/torch/la_muse_eccv16.t7"),
    "starry_night": StyleModel("starry_night", "torch7", "/app/models/torch/starry_night_eccv16.t7"),
    "the_scream": StyleModel("the_scream", "torch7", "/app/models/torch/the_scream.t7"),
    "the_wave": StyleModel("the_wave", "torch7", "/app/models/torch/the_wave_eccv16.t7"),
}


def find_images(directory: Path) -> List[Path]:
    """Find all images in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    images = []
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            images.append(f)
    return sorted(images)


def find_magenta_styles(style_dir: Path) -> List[str]:
    """Find all magenta style images."""
    if not style_dir.exists():
        return []

    extensions = {'.jpg', '.jpeg', '.png'}
    styles = []
    for f in style_dir.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            styles.append(f.stem)
    return sorted(styles)


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 0.0


def style_image_transformer(src: Path, model: StyleModel, output: Path, config: Config) -> bool:
    """Style image using transformer model via pipeline."""
    work = config.work_dir / "transformer_tmp"
    work.mkdir(parents=True, exist_ok=True)

    # Create 1-frame video from image
    tmp_video = work / "in.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", "0.04", "-r", str(config.fps),
        "-i", str(src),
        "-vf", f"scale={config.scale}:-2:flags=bicubic,format=yuv420p",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(tmp_video)
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        return False

    # Run pipeline
    out_video = work / "out.mp4"
    cmd = [
        "python3", str(PIPELINE),
        "--model_type", "transformer",
        "--model", model.path,
        "--io_preset", model.io_preset,
        "--input_video", str(tmp_video),
        "--output_video", str(out_video),
        "--work_dir", str(work),
        "--max_frames", "1",
        "--fps", str(config.fps),
        "--scale", str(config.scale),
        "--image_ext", "jpg",
        "--jpeg_quality", str(config.jpeg_quality)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    Pipeline failed: {result.stderr[:200]}")
        return False

    # Extract styled frame
    frames_dir = work / "frames"
    styled_frame = None
    for pattern in ["styled_frame_0001.jpg", "styled_frame_001.jpg", "styled_frame_1.jpg"]:
        candidate = frames_dir / pattern
        if candidate.exists():
            styled_frame = candidate
            break

    if styled_frame and styled_frame.exists():
        shutil.copy(styled_frame, output)
        return True

    return False


def style_image_torch7(src: Path, model: StyleModel, output: Path, config: Config) -> bool:
    """Style image using Torch7 model via OpenCV DNN."""
    import cv2

    try:
        # Read image
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return False

        # Scale if needed
        h, w = img.shape[:2]
        if h > config.scale:
            new_w = int(round(w * (config.scale / float(h))))
            img = cv2.resize(img, (new_w, config.scale), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        # Caffe-style preprocessing
        MEAN_BGR = (103.939, 116.779, 123.680)
        blob = cv2.dnn.blobFromImage(
            image=img,
            scalefactor=1.0,
            size=(w, h),
            mean=MEAN_BGR,
            swapRB=False,
            crop=False
        )

        # Load and run model
        net = cv2.dnn.readNetFromTorch(model.path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        net.setInput(blob)
        out = net.forward()

        if out is None or out.size == 0:
            return False

        # Convert output
        out = out.squeeze().transpose(1, 2, 0)
        out = out + np.array(MEAN_BGR, dtype=np.float32)[None, None, :]
        out = np.clip(out, 0, 255).astype(np.uint8)

        cv2.imwrite(str(output), out, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
        return True

    except Exception as e:
        print(f"    Torch7 error: {e}")
        return False


def style_image_magenta(src: Path, style_name: str, output: Path, config: Config) -> bool:
    """Style image using Magenta TF-Hub."""
    style_path = config.magenta_style_dir / f"{style_name}.jpg"
    if not style_path.exists():
        style_path = config.magenta_style_dir / f"{style_name}.png"
    if not style_path.exists():
        print(f"    Magenta style not found: {style_name}")
        return False

    cmd = [
        "python3", str(PIPELINE),
        "--model_type", "magenta",
        "--magenta_style", str(style_path),
        "--magenta_tile", str(config.magenta_tile),
        "--magenta_overlap", str(config.magenta_overlap),
        "--input_image", str(src),
        "--output_image", str(output),
        "--image_ext", "jpg",
        "--jpeg_quality", str(config.jpeg_quality),
        "--device", "cpu"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0 and output.exists()


def blend_images(img_a: Path, img_b: Path, output: Path, ratio: float = 0.5) -> bool:
    """Blend two images together."""
    try:
        a = Image.open(img_a).convert('RGB')
        b = Image.open(img_b).convert('RGB')

        # Match sizes
        if b.size != a.size:
            b = b.resize(a.size, Image.BICUBIC)

        # Blend
        blended = Image.blend(a, b, ratio)

        # Ensure even dimensions
        w, h = blended.size
        w2, h2 = w - (w % 2), h - (h % 2)
        if (w2, h2) != (w, h):
            blended = blended.crop((0, 0, w2, h2))

        blended.save(output, format='JPEG', quality=90, subsampling=0)
        return True
    except Exception as e:
        print(f"    Blend error: {e}")
        return False


def create_motion_clip(
    image: Path,
    output: Path,
    duration: float,
    config: Config,
    motion: str = "none"
) -> bool:
    """Create a video clip from image with optional motion effect."""

    # Base filter for scaling
    filters = [f"scale={config.scale}:-2:flags=bicubic"]

    # Add motion effects
    if motion == "zoom_in":
        # Slow zoom in (1.0 to 1.0 + strength)
        strength = config.motion_strength
        filters.append(
            f"zoompan=z='1+{strength}*in/({config.fps}*{duration})':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )
    elif motion == "zoom_out":
        # Slow zoom out (1.0 + strength to 1.0)
        strength = config.motion_strength
        filters.append(
            f"zoompan=z='{1+strength}-{strength}*in/({config.fps}*{duration})':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )
    elif motion == "pan_left":
        # Pan from right to left
        filters.append(
            f"zoompan=z='1.1':x='iw*0.1*(1-in/({config.fps}*{duration}))':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )
    elif motion == "pan_right":
        # Pan from left to right
        filters.append(
            f"zoompan=z='1.1':x='iw*0.1*in/({config.fps}*{duration})':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )
    elif motion == "pan_up":
        # Pan from bottom to top
        filters.append(
            f"zoompan=z='1.1':y='ih*0.1*(1-in/({config.fps}*{duration}))':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )
    elif motion == "pan_down":
        # Pan from top to bottom
        filters.append(
            f"zoompan=z='1.1':y='ih*0.1*in/({config.fps}*{duration})':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )
    elif motion == "ken_burns":
        # Random ken burns effect (zoom + pan)
        # Alternate between zoom in/out with slight pan
        strength = config.motion_strength
        direction = random.choice(['in', 'out'])
        pan_x = random.uniform(-0.05, 0.05)
        pan_y = random.uniform(-0.05, 0.05)

        if direction == 'in':
            zoom = f"1+{strength}*in/({config.fps}*{duration})"
        else:
            zoom = f"{1+strength}-{strength}*in/({config.fps}*{duration})"

        filters.append(
            f"zoompan=z='{zoom}':x='iw*{pan_x}*in/({config.fps}*{duration})':y='ih*{pan_y}*in/({config.fps}*{duration})':d={int(config.fps * duration)}:s={config.scale}x-2:fps={config.fps}"
        )

    filters.append("format=yuv420p")
    filter_str = ",".join(filters)

    # For motion effects, we need different approach
    if motion != "none":
        # zoompan creates its own frames, so we just input the image once
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(image),
            "-vf", filter_str,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(config.crf), "-preset", config.preset,
            str(output)
        ]
    else:
        # No motion - simple loop
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-t", str(duration), "-r", str(config.fps),
            "-i", str(image),
            "-vf", filter_str,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(config.crf), "-preset", config.preset,
            str(output)
        ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0 and output.exists()


def crossfade_videos(video_a: Path, video_b: Path, output: Path, config: Config) -> bool:
    """Crossfade two videos together."""
    dur_a = get_video_duration(video_a)
    offset = max(0, dur_a - config.trans_duration)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_a),
        "-i", str(video_b),
        "-filter_complex",
        f"[0:v]tpad=stop_mode=clone:stop_duration={config.trans_duration}[a];"
        f"[1:v]tpad=stop_mode=clone:stop_duration={config.trans_duration}[b];"
        f"[a][b]xfade=transition={config.transition}:duration={config.trans_duration}:offset={offset},format=yuv420p[v]",
        "-map", "[v]",
        "-r", str(config.fps),
        "-pix_fmt", "yuv420p",
        "-crf", str(config.crf), "-preset", config.preset,
        str(output)
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    return result.returncode == 0 and output.exists()


def build_showcase(
    src_image: Path,
    config: Config,
    selected_styles: Optional[List[str]] = None,
    include_magenta: bool = True
) -> Optional[Path]:
    """Build a style showcase video for a single image."""

    base_name = src_image.stem
    work = config.work_dir / base_name
    stills_dir = work / "stills"
    clips_dir = work / "clips"

    # Clean and create directories
    if work.exists():
        shutil.rmtree(work)
    stills_dir.mkdir(parents=True)
    clips_dir.mkdir(parents=True)

    print(f"\n[showcase] Processing: {src_image.name}")

    # Determine which styles to use
    available_styles = list(STYLE_MODELS.keys())

    # Add magenta styles if available
    magenta_styles = []
    if include_magenta:
        magenta_styles = find_magenta_styles(config.magenta_style_dir)
        print(f"  Found {len(magenta_styles)} Magenta styles")

    # Select styles
    if selected_styles:
        styles_to_use = [s for s in selected_styles if s in available_styles or s in magenta_styles]
    else:
        # Random selection up to max_models
        all_styles = available_styles + [f"magenta:{s}" for s in magenta_styles]
        random.shuffle(all_styles)
        styles_to_use = all_styles[:config.max_models]

    print(f"  Using {len(styles_to_use)} styles: {', '.join(styles_to_use[:5])}...")

    # 1. Copy original
    orig_still = stills_dir / "orig.jpg"
    img = Image.open(src_image).convert('RGB')
    # Scale to target height
    w, h = img.size
    if h > config.scale:
        new_w = int(w * config.scale / h)
        img = img.resize((new_w, config.scale), Image.LANCZOS)
    # Ensure even dimensions
    w, h = img.size
    if w % 2 or h % 2:
        img = img.crop((0, 0, w - w % 2, h - h % 2))
    img.save(orig_still, quality=config.jpeg_quality)
    print(f"  [orig] Saved original")

    # Also save to output
    shutil.copy(orig_still, config.out_dir / f"{base_name}_orig.jpg")

    # 2. Generate styled stills
    styled_stills = {"orig": orig_still}

    for style in styles_to_use:
        if style.startswith("magenta:"):
            style_name = style.replace("magenta:", "")
            safe_name = f"magenta_{style_name}"
            still_path = stills_dir / f"{safe_name}.jpg"
            print(f"  [style] {style_name} (magenta)...", end=" ", flush=True)
            if style_image_magenta(src_image, style_name, still_path, config):
                styled_stills[safe_name] = still_path
                shutil.copy(still_path, config.out_dir / f"{base_name}_{safe_name}.jpg")
                print("OK")
            else:
                print("FAILED")
        elif style in STYLE_MODELS:
            model = STYLE_MODELS[style]
            still_path = stills_dir / f"{style}.jpg"
            print(f"  [style] {style} ({model.model_type})...", end=" ", flush=True)

            if model.model_type == "transformer":
                success = style_image_transformer(src_image, model, still_path, config)
            elif model.model_type == "torch7":
                success = style_image_torch7(src_image, model, still_path, config)
            else:
                success = False

            if success and still_path.exists():
                styled_stills[style] = still_path
                shutil.copy(still_path, config.out_dir / f"{base_name}_{style}.jpg")
                print("OK")
            else:
                print("FAILED")

    if len(styled_stills) < 2:
        print("  ERROR: Not enough styled images generated")
        return None

    # 3. Generate blends (optional)
    blend_stills = {}
    if config.include_blends:
        style_names = [s for s in styled_stills.keys() if s != "orig"]
        if len(style_names) >= 2:
            # Generate random pairs
            pairs = []
            for i, a in enumerate(style_names):
                for b in style_names[i+1:]:
                    pairs.append((a, b))

            random.shuffle(pairs)
            pairs = pairs[:config.max_blends]

            for a, b in pairs:
                blend_name = f"{a}_{b}"
                blend_path = stills_dir / f"{blend_name}.jpg"
                print(f"  [blend] {a} + {b}...", end=" ", flush=True)
                if blend_images(styled_stills[a], styled_stills[b], blend_path):
                    blend_stills[blend_name] = blend_path
                    shutil.copy(blend_path, config.out_dir / f"{base_name}_{blend_name}.jpg")
                    print("OK")
                else:
                    print("FAILED")

    # 4. Build sequence: orig → styles → blends → orig
    sequence = ["orig"]
    sequence.extend([s for s in styled_stills.keys() if s != "orig"])
    sequence.extend(blend_stills.keys())
    sequence.append("orig")

    all_stills = {**styled_stills, **blend_stills}

    print(f"  [sequence] {len(sequence)} clips: {' → '.join(sequence[:5])}...")

    # 5. Create video clips with motion effects
    clips = []
    motion_options = ["zoom_in", "zoom_out", "pan_left", "pan_right", "ken_burns"]

    for i, name in enumerate(sequence):
        if name not in all_stills:
            continue

        still = all_stills[name]
        clip_path = clips_dir / f"{i:03d}_{name}.mp4"

        # Determine hold duration
        if name == "orig":
            if i == 0:
                hold = config.hold_orig_start
            else:
                hold = config.hold_orig_end
        else:
            hold = config.hold_model

        # Add transition time to clip
        duration = hold + config.trans_duration

        # Select motion effect
        if config.motion == "random":
            motion = random.choice(motion_options)
        elif config.motion == "ken_burns":
            motion = "ken_burns"
        else:
            motion = config.motion

        if create_motion_clip(still, clip_path, duration, config, motion):
            clips.append(clip_path)
        else:
            print(f"  WARNING: Failed to create clip for {name}")

    if len(clips) < 2:
        print("  ERROR: Not enough clips generated")
        return None

    # 6. Chain clips with crossfades
    print(f"  [chain] Crossfading {len(clips)} clips...")

    accum = clips[0]
    for i, clip in enumerate(clips[1:], 1):
        next_accum = clips_dir / f"accum_{i}.mp4"
        if crossfade_videos(accum, clip, next_accum, config):
            accum = next_accum
        else:
            print(f"  WARNING: Crossfade failed at clip {i}")

    # 7. Move final video to output
    output_video = config.out_dir / f"{base_name}_showcase.mp4"
    shutil.move(accum, output_video)

    duration = get_video_duration(output_video)
    print(f"  [done] {output_video.name} ({duration:.1f}s)")

    return output_video


def main():
    parser = argparse.ArgumentParser(description="Generate style showcase videos")
    parser.add_argument("--input", "-i", type=str, help="Input directory or single image")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    parser.add_argument("--styles", "-s", type=str, help="Comma-separated list of styles to use")
    parser.add_argument("--motion", "-m", type=str, default="none",
                        choices=["none", "zoom_in", "zoom_out", "pan_left", "pan_right",
                                "pan_up", "pan_down", "ken_burns", "random"],
                        help="Motion effect for clips")
    parser.add_argument("--motion-strength", type=float, default=0.1,
                        help="Strength of motion effect (0.05-0.2)")
    parser.add_argument("--max-models", type=int, default=10, help="Max styles per image")
    parser.add_argument("--no-blends", action="store_true", help="Skip style blends")
    parser.add_argument("--no-magenta", action="store_true", help="Skip Magenta styles")
    parser.add_argument("--transition", type=str, default="fade",
                        help="FFmpeg xfade transition type")
    parser.add_argument("--list", "-l", action="store_true", help="List available styles")

    args = parser.parse_args()

    if args.list:
        print("Available transformer/torch7 styles:")
        for name, model in STYLE_MODELS.items():
            status = "OK" if Path(model.path).exists() else "MISSING"
            print(f"  {name} ({model.model_type}) [{status}]")

        config = Config()
        magenta = find_magenta_styles(config.magenta_style_dir)
        if magenta:
            print(f"\nMagenta styles in {config.magenta_style_dir}:")
            for s in magenta[:20]:
                print(f"  magenta:{s}")
            if len(magenta) > 20:
                print(f"  ... and {len(magenta) - 20} more")
        return 0

    # Build config
    config = Config()

    if args.input:
        config.in_dir = Path(args.input)
    if args.output:
        config.out_dir = Path(args.output)
    if args.motion:
        config.motion = args.motion
    if args.motion_strength:
        config.motion_strength = args.motion_strength
    if args.max_models:
        config.max_models = args.max_models
    if args.no_blends:
        config.include_blends = False
    if args.transition:
        config.transition = args.transition

    # Parse selected styles
    selected_styles = None
    if args.styles:
        selected_styles = [s.strip() for s in args.styles.split(",")]

    # Create output directory
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    if config.in_dir.is_file():
        images = [config.in_dir]
    else:
        images = find_images(config.in_dir)

    if not images:
        print(f"ERROR: No images found in {config.in_dir}")
        return 1

    print(f"Style Showcase Generator")
    print(f"  Input: {config.in_dir}")
    print(f"  Output: {config.out_dir}")
    print(f"  Images: {len(images)}")
    print(f"  Motion: {config.motion}")
    print(f"  Max styles: {config.max_models}")
    print(f"  Blends: {'yes' if config.include_blends else 'no'}")
    print(f"  Transition: {config.transition} ({config.trans_duration}s)")

    # Process each image
    success = 0
    failed = 0

    for img in images:
        result = build_showcase(
            img, config,
            selected_styles=selected_styles,
            include_magenta=not args.no_magenta
        )
        if result:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Complete! Success: {success}, Failed: {failed}")
    print(f"Output: {config.out_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
