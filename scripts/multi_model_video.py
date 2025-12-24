#!/usr/bin/env python3
"""
Multi-model video compositor.
Creates video with multiple style segments, blending styles with Gaussian pulses.

Run inside Docker container:
  docker compose run --rm -v /path/to/project:/project style \
    python3 /app/scripts/multi_model_video.py \
    --orig_dir /project/output/my_video/frames \
    --styled_udnie /project/output/my_video/styled_udnie \
    --styled_mosaic /project/output/my_video/styled_mosaic \
    --output /project/output/my_video.mp4
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import subprocess
import os
from datetime import datetime

def load_resize(path, size=1080, portrait=True):
    """Load and resize image for portrait 9:16."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if portrait:
        # Scale width to size, height proportional
        scale = size / w
        new_w = size
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Scale to fit
        if w > h:
            scale = size / w
        else:
            scale = size / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def load_walk(walk_file):
    """Load walk from JSON file."""
    with open(walk_file) as f:
        data = json.load(f)
    return data['walk'], data['weights']

def smooth_walk_ema(walk, alpha=0.05):
    """Apply EMA smoothing to walk."""
    smoothed = [float(walk[0])]
    for i in range(1, len(walk)):
        smoothed.append(alpha * walk[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def get_styled_frame(styled_dir, frame_name, weights, weight_pos, size, portrait, orig_blend=0.4):
    """Get interpolated styled frame at given weight position."""
    # Load original
    orig_path = styled_dir / f"{frame_name}_original.jpg"
    orig = load_resize(orig_path, size, portrait)
    if orig is None:
        return None
    orig = orig.astype(np.float32)

    # Interpolate between weights
    idx_low = int(weight_pos)
    idx_high = min(idx_low + 1, len(weights) - 1)
    blend_factor = weight_pos - idx_low

    # Load low weight
    weight_low = weights[idx_low]
    style_low_path = styled_dir / f"{frame_name}_{weight_low}.jpg"
    style_low = load_resize(style_low_path, size, portrait)

    if style_low is None:
        # Fallback to any available weight
        for w in weights:
            p = styled_dir / f"{frame_name}_{w}.jpg"
            if p.exists():
                style_low = load_resize(p, size, portrait)
                break

    if style_low is None:
        return orig.astype(np.uint8)

    style_low = style_low.astype(np.float32)

    # Blend with high weight if needed
    if blend_factor > 0.01 and idx_high != idx_low:
        weight_high = weights[idx_high]
        style_high_path = styled_dir / f"{frame_name}_{weight_high}.jpg"
        style_high = load_resize(style_high_path, size, portrait)
        if style_high is not None:
            style_high = style_high.astype(np.float32)
            style_frame = style_low * (1 - blend_factor) + style_high * blend_factor
        else:
            style_frame = style_low
    else:
        style_frame = style_low

    # Blend with original
    frame = orig * orig_blend + style_frame * (1 - orig_blend)
    return np.clip(frame, 0, 255).astype(np.uint8)

def smoothstep(t):
    """Smooth interpolation."""
    return t * t * (3 - 2 * t)

def adjust_saturation(img, factor=1.3):
    """Adjust color saturation. factor > 1 increases, < 1 decreases."""
    img_float = img.astype(np.float32)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Boost saturation channel
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    # Convert back to BGR
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result

def gaussian_pulse(t, num_pulses=4, width=0.15):
    """Create Gaussian pulses over normalized time t (0-1).
    Returns blend factor 0-1 with multiple pulses."""
    import math
    total = 0
    for i in range(num_pulses):
        center = (i + 0.5) / num_pulses
        total += math.exp(-((t - center) ** 2) / (2 * width ** 2))
    return min(1.0, total)


def save_run_log(output_path, params, styled_dirs, total_frames, duration_sec):
    """Save a JSON log of the run parameters and results."""
    log_path = output_path.parent / f"{output_path.stem}_run.json"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "script": "multi_model_video.py",
        "output_video": str(output_path),
        "duration_seconds": round(duration_sec, 2),
        "total_frames": total_frames,
        "parameters": {
            "orig_dir": str(params.get('orig_dir', '')),
            "styled_start": params.get('styled_start'),
            "styled_end": params.get('styled_end'),
            "fps": params.get('fps'),
            "size": params.get('size'),
            "portrait": params.get('portrait'),
            "orig_blend": params.get('orig_blend'),
            "crossfade_frames": params.get('crossfade_frames'),
            "hold_frames": params.get('hold_frames'),
            "saturation": params.get('saturation'),
            "smoothness": params.get('smoothness'),
        },
        "styled_dirs": {k: str(v) for k, v in styled_dirs.items()},
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  [log] Saved run parameters to {log_path.name}")
    return log_path

def create_multi_model_video(
    orig_dir,
    styled_dirs,  # dict: {'udnie': path, 'tenharmsel': path, 'mosaic': path}
    output_path,
    styled_start=17,
    styled_end=160,
    total_frames=None,
    fps=24,
    size=1080,
    portrait=True,
    orig_blend=0.4,
    crossfade_frames=48,  # 2 seconds at 24fps
    smoothness=0.05,
    hold_frames=3,  # 8fps extraction, 24fps output
    saturation=1.3  # Saturation boost for styled frames (1.0 = no change)
):
    """Create multi-model video with crossfades and Gaussian pulsing blend."""

    orig_dir = Path(orig_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load walks for each style
    walks = {}
    weights = {}
    for style_name, style_dir in styled_dirs.items():
        style_dir = Path(style_dir)
        walk_file = style_dir / f"walk_{style_name}.json"
        if not walk_file.exists():
            walk_file = style_dir / "walk.json"

        if walk_file.exists():
            with open(walk_file) as f:
                data = json.load(f)
            w = data['walk']
            wts = data['weights']
            walks[style_name] = smooth_walk_ema(w, alpha=smoothness)
            weights[style_name] = wts
            print(f"  Loaded walk for {style_name}: {len(w)} positions")

    # Determine video dimensions from first frame
    first_frames = list(orig_dir.glob("frame_0001_original.jpg"))
    if not first_frames:
        first_frames = list(orig_dir.glob("frame_*_original.jpg"))[:1]
    if not first_frames:
        print("ERROR: No frames found")
        return False

    test_img = load_resize(first_frames[0], size, portrait)
    video_h, video_w = test_img.shape[:2]
    print(f"  Video dimensions: {video_w}x{video_h}")

    # Calculate total frames if not provided
    if total_frames is None:
        total_frames = len(list(orig_dir.glob("frame_*_original.jpg")))

    # Build segments: original intro -> styled -> original outro
    segments = [
        (1, styled_start - 1, None),        # Original intro
        (styled_start, styled_end, 'udnie'), # Styled section with blending
        (styled_end + 1, total_frames, None) # Original outro
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))

    print(f"\nCreating multi-model video:")
    print(f"  {len(segments)} segments")
    print(f"  Crossfade: {crossfade_frames} frames ({crossfade_frames/fps:.1f}s)")

    written = 0

    for seg_idx, (start, end, style_name) in enumerate(segments):
        if start > end:
            continue

        is_last_seg = (seg_idx == len(segments) - 1)
        next_seg = segments[seg_idx + 1] if not is_last_seg else None

        style_dir = Path(styled_dirs.get(style_name, orig_dir)) if style_name else orig_dir
        style_weights = weights.get(style_name, None)

        print(f"  Segment {seg_idx+1}: frames {start}-{end}, style={style_name or 'original'}")

        for src_idx, frame_num in enumerate(range(start, end + 1)):
            frame_name = f"frame_{frame_num:04d}"

            # Get the frame
            if style_name and style_weights:
                # Slow step-wise ramp through weights
                segment_frames = end - start + 1

                # Skip weak weights for udnie (start at style5e10 = index 0 since we only have 5-9)
                start_weight_idx = 0
                usable_weights = len(style_weights)

                frames_per_weight = max(1, segment_frames // usable_weights)
                weight_idx = min(src_idx // frames_per_weight, usable_weights - 1)
                weight_progress = (src_idx % frames_per_weight) / frames_per_weight
                weight_pos = start_weight_idx + weight_idx + weight_progress * 0.5

                frame = get_styled_frame(style_dir, frame_name, style_weights, weight_pos, size, portrait, orig_blend)

                # Blend all three styles with offset Gaussian pulsing
                if style_name == 'udnie':
                    segment_progress = src_idx / max(1, (end - start))
                    frame_f = frame.astype(np.float32) if frame is not None else None

                    # Offset pulses for each style so they interweave
                    mosaic_blend = gaussian_pulse(segment_progress, num_pulses=4, width=0.10) * 0.5
                    tenharmsel_blend = gaussian_pulse(segment_progress + 0.125, num_pulses=4, width=0.10) * 0.5

                    # Load and blend mosaic
                    if mosaic_blend > 0.01 and 'mosaic' in styled_dirs and frame_f is not None:
                        mosaic_dir = Path(styled_dirs['mosaic'])
                        mosaic_weights = weights.get('mosaic', ['mosaic_style5e10'])
                        mosaic_frame = get_styled_frame(mosaic_dir, frame_name, mosaic_weights, 0, size, portrait, orig_blend)
                        if mosaic_frame is not None:
                            frame_f = frame_f * (1 - mosaic_blend) + mosaic_frame.astype(np.float32) * mosaic_blend

                    # Load and blend tenharmsel
                    if tenharmsel_blend > 0.01 and 'tenharmsel' in styled_dirs and frame_f is not None:
                        tenharmsel_dir = Path(styled_dirs['tenharmsel'])
                        tenharmsel_weights = weights.get('tenharmsel', None)
                        if tenharmsel_weights:
                            tenharmsel_frame = get_styled_frame(tenharmsel_dir, frame_name, tenharmsel_weights, 5, size, portrait, orig_blend)
                            if tenharmsel_frame is not None:
                                frame_f = frame_f * (1 - tenharmsel_blend) + tenharmsel_frame.astype(np.float32) * tenharmsel_blend

                    if frame_f is not None:
                        frame = np.clip(frame_f, 0, 255).astype(np.uint8)
            else:
                # Original frame
                orig_path = orig_dir / f"{frame_name}_original.jpg"
                frame = load_resize(orig_path, size, portrait)

            if frame is None:
                continue

            frame = frame.astype(np.float32)

            # Apply saturation boost to styled frames
            if style_name and saturation != 1.0:
                frame = adjust_saturation(np.clip(frame, 0, 255).astype(np.uint8), saturation).astype(np.float32)

            # Handle fade-in from original at START of styled segments
            # Convert crossfade_frames (output fps) to source frames
            crossfade_src_frames = crossfade_frames // hold_frames
            if style_name and src_idx < crossfade_src_frames:
                fade_in_progress = src_idx / crossfade_src_frames
                fade_in_t = smoothstep(fade_in_progress)

                orig_path = orig_dir / f"{frame_name}_original.jpg"
                orig_frame = load_resize(orig_path, size, portrait)
                if orig_frame is not None:
                    orig_frame = orig_frame.astype(np.float32)
                    frame = orig_frame * (1 - fade_in_t) + frame * fade_in_t

            # Handle crossfade to next segment
            frames_from_end = end - frame_num
            if not is_last_seg and frames_from_end < crossfade_frames:
                fade_progress = 1 - (frames_from_end / crossfade_frames)
                fade_t = smoothstep(fade_progress)

                next_start, next_end, next_style = next_seg
                if next_style is None:
                    next_path = orig_dir / f"{frame_name}_original.jpg"
                    next_frame = load_resize(next_path, size, portrait)
                else:
                    next_style_dir = Path(styled_dirs.get(next_style, orig_dir))
                    next_weights = weights.get(next_style, None)
                    if next_weights:
                        next_frame = get_styled_frame(next_style_dir, frame_name, next_weights, 0, size, portrait, orig_blend)
                    else:
                        next_path = orig_dir / f"{frame_name}_original.jpg"
                        next_frame = load_resize(next_path, size, portrait)

                if next_frame is not None:
                    next_frame = next_frame.astype(np.float32)
                    frame = frame * (1 - fade_t) + next_frame * fade_t

            frame = np.clip(frame, 0, 255).astype(np.uint8)

            # Write hold_frames copies for smooth playback
            for _ in range(hold_frames):
                out.write(frame)
                written += 1

    out.release()

    # Convert to H.264
    h264_path = str(output_path).replace('.mp4', '_h264.mp4')
    print(f"\n  Converting to H.264...")
    subprocess.run([
        'ffmpeg', '-y', '-i', str(output_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        h264_path
    ], capture_output=True)

    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
        os.remove(str(output_path))
        os.rename(h264_path, str(output_path))

    duration = written / fps
    print(f"\n  Done! {written} frames, {duration:.1f}s ({duration/60:.1f} min)")

    # Save run log
    params = {
        'orig_dir': orig_dir,
        'styled_start': styled_start,
        'styled_end': styled_end,
        'fps': fps,
        'size': size,
        'portrait': portrait,
        'orig_blend': orig_blend,
        'crossfade_frames': crossfade_frames,
        'hold_frames': hold_frames,
        'saturation': saturation,
        'smoothness': smoothness,
    }
    save_run_log(output_path, params, styled_dirs, written, duration)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model video compositor")
    parser.add_argument('--orig_dir', required=True, help='Directory with original frames')
    parser.add_argument('--styled_udnie', required=True, help='Directory with udnie styled frames')
    parser.add_argument('--styled_mosaic', help='Directory with mosaic styled frames')
    parser.add_argument('--styled_tenharmsel', help='Directory with tenharmsel styled frames')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--styled_start', type=int, default=17, help='First styled frame')
    parser.add_argument('--styled_end', type=int, default=160, help='Last styled frame')
    parser.add_argument('--total_frames', type=int, help='Total frames (auto-detected if not set)')
    parser.add_argument('--fps', type=int, default=24, help='Output FPS')
    parser.add_argument('--size', type=int, default=1080, help='Output size')
    parser.add_argument('--portrait', action='store_true', default=True)
    parser.add_argument('--orig_blend', type=float, default=0.4, help='Original blend ratio')
    parser.add_argument('--crossfade_seconds', type=float, default=2.0, help='Crossfade duration')
    parser.add_argument('--hold_frames', type=int, default=3, help='Frames to hold per source frame')
    parser.add_argument('--saturation', type=float, default=1.3, help='Saturation boost for styled frames (1.0 = no change)')
    args = parser.parse_args()

    styled_dirs = {'udnie': args.styled_udnie}
    if args.styled_mosaic:
        styled_dirs['mosaic'] = args.styled_mosaic
    if args.styled_tenharmsel:
        styled_dirs['tenharmsel'] = args.styled_tenharmsel

    create_multi_model_video(
        orig_dir=args.orig_dir,
        styled_dirs=styled_dirs,
        output_path=args.output,
        styled_start=args.styled_start,
        styled_end=args.styled_end,
        total_frames=args.total_frames,
        fps=args.fps,
        size=args.size,
        portrait=args.portrait,
        orig_blend=args.orig_blend,
        crossfade_frames=int(args.crossfade_seconds * args.fps),
        hold_frames=args.hold_frames,
        saturation=args.saturation
    )
