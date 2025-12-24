#!/usr/bin/env python3
"""
Style Morph - Multi-image style morphing with weight flow across images.
Styles gradually ramp up/down over multiple images for fluid flow.
Run inside Docker.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import random
import math

# Weight ladders for each style family (ordered low to high)
CANDY_LADDER = ['candy', 'candy_style1e9', 'candy_style5e9', 'candy_style1e10', 'candy_style5e10', 'candy_style1e11', 'candy_style5e11', 'candy_style1e12']
UDNIE_LADDER = ['udnie', 'udnie_style1e9', 'udnie_style5e9', 'udnie_style1e10', 'udnie_style5e10', 'udnie_style1e11', 'udnie_style5e11', 'udnie_style1e12']
MOSAIC_LADDER = ['mosaic', 'mosaic_style1e9', 'mosaic_style5e9', 'mosaic_style1e10', 'mosaic_style5e10', 'mosaic_style1e11', 'mosaic_style5e11', 'mosaic_style1e12']
RAIN_PRINCESS_LADDER = ['rain_princess', 'rain_princess_style1e9', 'rain_princess_style5e9', 'rain_princess_style1e10', 'rain_princess_style5e10', 'rain_princess_style1e11', 'rain_princess_style5e11', 'rain_princess_style1e12']

# Tenharmsel: 1e9 -> 2e9 -> ... -> 9e9 -> 1e10 -> ... -> 1e12
TENHARMSEL_LADDER = [
    'tenharmsel_style1e9', 'tenharmsel_style2e9', 'tenharmsel_style3e9', 'tenharmsel_style4e9',
    'tenharmsel_style5e9', 'tenharmsel_style6e9', 'tenharmsel_style7e9', 'tenharmsel_style8e9', 'tenharmsel_style9e9',
    'tenharmsel_style1e10', 'tenharmsel_style2e10', 'tenharmsel_style3e10', 'tenharmsel_style4e10',
    'tenharmsel_style5e10', 'tenharmsel_style6e10', 'tenharmsel_style7e10', 'tenharmsel_style8e10', 'tenharmsel_style9e10',
    'tenharmsel_style1e11', 'tenharmsel_style2e11', 'tenharmsel_style3e11', 'tenharmsel_style4e11',
    'tenharmsel_style5e11', 'tenharmsel_style6e11', 'tenharmsel_style7e11', 'tenharmsel_style8e11', 'tenharmsel_style9e11',
    'tenharmsel_style1e12',
]

ALL_LADDERS = {
    'candy': CANDY_LADDER,
    'udnie': UDNIE_LADDER,
    'mosaic': MOSAIC_LADDER,
    'rain_princess': RAIN_PRINCESS_LADDER,
    'tenharmsel': TENHARMSEL_LADDER,
}

# Gentle filters
def boost_saturation(img, factor=1.10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def warm_filter(img, strength=0.06):
    img = img.astype(np.float32)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1 + strength), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (1 + strength * 0.3), 0, 255)
    img[:, :, 0] = np.clip(img[:, :, 0] * (1 - strength * 0.3), 0, 255)
    return img.astype(np.uint8)

def vibrance(img, factor=1.10):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    boost = factor + (1 - factor) * (sat / 255)
    hsv[:, :, 1] = np.clip(sat * boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

FILTERS = [
    ('none', lambda x: x),
    ('subtle_sat', lambda x: boost_saturation(x, 1.08)),
    ('vibrance', lambda x: vibrance(x, 1.08)),
    ('warm', lambda x: warm_filter(x, 0.05)),
]

def load_resize(path, size=1080, portrait=False):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if portrait:
        # Keep aspect ratio, scale to height=size
        scale = size / h
        new_w = int(w * scale)
        new_h = size
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        # Square crop
        if w > h:
            x = (w - h) // 2
            img = img[:, x:x+h]
        elif h > w:
            y = (h - w) // 2
            img = img[y:y+w, :]
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)

def get_styled_image(styled_dir, img_name, style, size, portrait=False):
    path = styled_dir / f"{img_name}_{style}.jpg"
    if path.exists():
        return load_resize(path, size, portrait)
    return None

def load_ladder_images(styled_dir, img_name, ladder, size, portrait=False):
    """Load all available images from a weight ladder."""
    images = []
    for style in ladder:
        img = get_styled_image(styled_dir, img_name, style, size, portrait)
        if img is not None:
            images.append(img.astype(np.float32))
    return images

def interpolate_ladder(images, position):
    """Interpolate between images in a ladder. Position 0-1."""
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    idx_float = position * (len(images) - 1)
    idx_low = int(idx_float)
    idx_high = min(idx_low + 1, len(images) - 1)
    blend = idx_float - idx_low
    blend = blend * blend * (3 - 2 * blend)  # Smoothstep

    return images[idx_low] * (1 - blend) + images[idx_high] * blend

def create_video(styled_dir, output_path, frame_seconds=4.0, fps=24, size=1080, skip_first=True,
                  portrait=False, orig_blend=0.08, families=None):
    styled_dir = Path(styled_dir)

    all_files = list(styled_dir.glob("*.jpg"))
    image_names = sorted(set(f.stem.rsplit('_', 1)[0] for f in all_files if '_' in f.stem))

    if skip_first and len(image_names) > 1:
        skipped = image_names[0]
        image_names = image_names[1:]
        print(f"  Skipping first image: {skipped}")

    # Filter style families if specified
    if families:
        use_ladders = {k: v for k, v in ALL_LADDERS.items() if k in families}
    else:
        use_ladders = ALL_LADDERS

    size_str = f"{size}p" if portrait else f"{size}x{size}"
    print(f"Creating Style Morph video (cross-image weight flow)")
    print(f"  {len(image_names)} images")
    print(f"  {frame_seconds}s per image, {size_str}, {'portrait' if portrait else 'square'}")
    print(f"  Original blend: {orig_blend*100:.0f}%, Families: {list(use_ladders.keys())}")

    # Preload all ladder images
    image_data = []
    for img_name in image_names:
        orig = get_styled_image(styled_dir, img_name, 'original', size, portrait)
        if orig is None:
            continue

        ladders = {}
        for name, ladder in use_ladders.items():
            imgs = load_ladder_images(styled_dir, img_name, ladder, size, portrait)
            if imgs:
                ladders[name] = imgs

        if len(ladders) < 1:
            continue

        filter_name, filter_fn = random.choice(FILTERS)

        data = {
            'name': img_name,
            'orig': orig.astype(np.float32),
            'ladders': ladders,
            'filter_fn': filter_fn,
        }
        image_data.append(data)

    print(f"  Loaded {len(image_data)} images")

    if len(image_data) < 2:
        print("Not enough images")
        return False

    num_images = len(image_data)
    frame_count = int(fps * frame_seconds)
    crossfade_frames = int(fps * 1.5)

    # === PLAN WEIGHT TRAJECTORIES ACROSS ALL IMAGES ===
    # Each style has a slow-moving position (0-1) on its weight ladder
    # Position changes gradually over multiple images

    style_names = list(ALL_LADDERS.keys())

    # Create smooth trajectories for each style's ladder position
    # Use slow sine waves with different frequencies/phases
    def get_ladder_position(style_idx, global_t):
        """Get ladder position (0-1) for a style at global time (0-1 across video)."""
        # Different frequency and phase for each style
        freq = 0.3 + style_idx * 0.15  # Slow waves
        phase = style_idx * 0.25
        # Sine wave mapped to 0-1
        pos = 0.5 + 0.5 * math.sin((global_t * freq + phase) * math.pi * 2)
        return pos

    # Create smooth contribution weights for each style
    def get_style_weight(style_idx, global_t, num_active):
        """Get blend weight for a style at global time."""
        # Slow gaussian-like waves
        freq = 0.2 + style_idx * 0.1
        phase = style_idx * 0.3
        raw = 0.5 + 0.5 * math.sin((global_t * freq + phase) * math.pi * 2)
        return max(0.1, raw)  # Minimum presence

    # Determine video dimensions from first loaded image
    first_orig = image_data[0]['orig']
    video_h, video_w = first_orig.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))

    total_frames = 0
    total_video_frames = num_images * frame_count

    for idx, data in enumerate(image_data):
        is_first = (idx == 0)
        is_last = (idx == len(image_data) - 1)
        next_data = image_data[idx + 1] if not is_last else None

        available_styles = list(data['ladders'].keys())

        for f in range(frame_count):
            # Global progress through entire video (0-1)
            global_frame = idx * frame_count + f
            global_t = global_frame / total_video_frames

            # Local progress within this image (0-1)
            local_t = f / frame_count

            ORIG_AMOUNT = orig_blend
            STYLE_TOTAL = 1.0 - orig_blend

            # Get each style's contribution
            style_contributions = []
            for style_idx, style_name in enumerate(available_styles):
                ladder_imgs = data['ladders'][style_name]

                # Ladder position based on global time (slow drift)
                ladder_pos = get_ladder_position(style_idx, global_t)

                # Get interpolated frame from ladder
                style_frame = interpolate_ladder(ladder_imgs, ladder_pos)

                # Blend weight based on global time
                weight = get_style_weight(style_idx, global_t, len(available_styles))

                style_contributions.append((style_frame, weight))

            # Normalize weights
            total_weight = sum(w for _, w in style_contributions)
            style_contributions = [(f, STYLE_TOTAL * w / total_weight)
                                   for f, w in style_contributions]

            # Blend frame
            frame = data['orig'] * ORIG_AMOUNT
            for style_frame, weight in style_contributions:
                if style_frame is not None:
                    frame += style_frame * weight

            # === CROSSFADE TO NEXT IMAGE ===
            if not is_last and f >= frame_count - crossfade_frames:
                fade_progress = (f - (frame_count - crossfade_frames)) / crossfade_frames
                fade_t = fade_progress * fade_progress * (3 - 2 * fade_progress)

                # Next image with same global time logic
                next_global_frame = (idx + 1) * frame_count + int(fade_progress * crossfade_frames * 0.3)
                next_global_t = next_global_frame / total_video_frames

                next_available = list(next_data['ladders'].keys())
                next_contributions = []

                for style_idx, style_name in enumerate(next_available):
                    ladder_imgs = next_data['ladders'][style_name]
                    ladder_pos = get_ladder_position(style_idx, next_global_t)
                    style_frame = interpolate_ladder(ladder_imgs, ladder_pos)
                    weight = get_style_weight(style_idx, next_global_t, len(next_available))
                    next_contributions.append((style_frame, weight))

                next_total = sum(w for _, w in next_contributions)
                next_contributions = [(f, STYLE_TOTAL * w / next_total)
                                      for f, w in next_contributions]

                next_frame = next_data['orig'] * ORIG_AMOUNT
                for style_frame, weight in next_contributions:
                    if style_frame is not None:
                        next_frame += style_frame * weight

                frame = frame * (1 - fade_t) + next_frame * fade_t

            # Skip frames from previous crossfade
            if not is_first and f < crossfade_frames:
                continue

            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame = data['filter_fn'](frame)

            out.write(frame)
            total_frames += 1

        if idx % 5 == 0:
            print(f"  [{idx+1}/{num_images}] {data['name']}")

    out.release()

    h264_path = str(output_path).replace('.mp4', '_h264.mp4')
    print(f"  Converting to H.264...")
    os.system(f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset medium -crf 20 "{h264_path}" 2>/dev/null')

    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
        os.remove(str(output_path))
        os.rename(h264_path, str(output_path))

    duration = total_frames / fps
    print(f"\n  Done! {total_frames} frames, {duration:.1f}s ({duration/60:.1f} min)")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--styled_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frame_time', type=float, default=4.0)
    parser.add_argument('--size', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--no_skip_first', action='store_true')
    parser.add_argument('--portrait', action='store_true', help='Portrait mode (keep aspect ratio)')
    parser.add_argument('--orig_blend', type=float, default=0.08, help='Original blend amount (0-1)')
    parser.add_argument('--families', nargs='+', help='Style families to use (e.g., tenharmsel udnie)')

    args = parser.parse_args()

    create_video(
        args.styled_dir, args.output,
        frame_seconds=args.frame_time,
        size=args.size,
        fps=args.fps,
        skip_first=not args.no_skip_first,
        portrait=args.portrait,
        orig_blend=args.orig_blend,
        families=args.families
    )
