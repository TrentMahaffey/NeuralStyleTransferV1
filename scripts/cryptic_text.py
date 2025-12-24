#!/usr/bin/env python3
"""
cryptic_text.py - Advanced artistic text overlay for videos

Features:
- Path-based animations (orbit, wave path, diagonal sweep, edge crawl)
- Morphing distortions (ripple, melt, glitch warp, breathing)
- NST texture fills (use styled images as text texture)
- Per-letter animations with independent timing
- Dynamic positioning and movement

Usage:
    docker compose run --rm style bash -lc "python /app/scripts/cryptic_text.py \
        --input /app/input/video.mp4 \
        --phrases 'DREAM,REALITY,CHAOS' \
        --output /app/output/cryptic.mp4"
"""

import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math
from pathlib import Path
import subprocess
import glob

# Available fonts
FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
]

# Animation modes (how text moves)
ANIMATIONS = ['orbit', 'wave_path', 'diagonal', 'edge_crawl', 'float', 'zoom_travel', 'spiral_in']

# Distortion effects (how text warps)
DISTORTIONS = ['ripple', 'melt', 'breathe', 'glitch_warp', 'wave_distort', 'none']

# Fill styles
FILLS = ['gradient_fire', 'gradient_ice', 'gradient_rainbow', 'neon_glow', 'chrome', 'nst_texture']


def ease_in_out(t):
    """Smooth easing function."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_out(t):
    return 1 - pow(1 - t, 3)


def ease_in(t):
    return t * t * t


# =============================================================================
# VIDEO I/O
# =============================================================================

def read_video(path):
    """Read all frames from video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"  Read {len(frames)} frames at {fps:.1f}fps ({width}x{height})")
    return frames, fps, (width, height)


def write_video(frames, path, fps, size):
    """Write frames to H.264 video."""
    path = Path(path)
    temp_path = path.with_suffix('.temp.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, size)

    for frame in frames:
        out.write(frame)
    out.release()

    subprocess.run([
        'ffmpeg', '-y', '-i', str(temp_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(path)
    ], capture_output=True)

    temp_path.unlink(missing_ok=True)
    print(f"  Wrote {len(frames)} frames to {path}")


# =============================================================================
# TEXT RENDERING
# =============================================================================

def render_text_mask(text, font_path, font_size):
    """Render text as a grayscale mask."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Get text size
    dummy = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Add padding
    padding = font_size // 2
    img = Image.new('L', (tw + padding * 2, th + padding * 2), 0)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=255)

    return np.array(img)


def render_letter_masks(text, font_path, font_size):
    """Render each letter as separate mask with position info."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    dummy = Image.new('L', (1, 1))
    draw = ImageDraw.Draw(dummy)

    letters = []
    x_offset = 0

    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        cw = bbox[2] - bbox[0]
        ch = bbox[3] - bbox[1]

        if cw > 0 and ch > 0:
            padding = 10
            img = Image.new('L', (cw + padding * 2, ch + padding * 2), 0)
            d = ImageDraw.Draw(img)
            d.text((padding - bbox[0], padding - bbox[1]), char, font=font, fill=255)

            letters.append({
                'char': char,
                'mask': np.array(img),
                'x_offset': x_offset,
                'width': cw,
                'height': ch
            })

        # Get advance width
        adv_bbox = draw.textbbox((0, 0), char + ' ', font=font)
        x_offset += (adv_bbox[2] - adv_bbox[0]) - draw.textbbox((0, 0), ' ', font=font)[2]

    return letters


# =============================================================================
# FILL STYLES (apply color/texture to mask)
# =============================================================================

def apply_gradient_fill(mask, gradient_type, progress=0):
    """Apply a gradient fill to text mask."""
    h, w = mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)

    for x in range(w):
        ratio = x / max(1, w - 1)

        if gradient_type == 'fire':
            # Red -> Orange -> Yellow -> White
            if ratio < 0.33:
                r, g, b = 200, int(50 + ratio * 300), 0
            elif ratio < 0.66:
                r, g, b = 255, int(150 + (ratio - 0.33) * 300), int((ratio - 0.33) * 150)
            else:
                r, g, b = 255, 255, int(50 + (ratio - 0.66) * 600)

        elif gradient_type == 'ice':
            # Deep blue -> Cyan -> White
            if ratio < 0.5:
                r, g, b = int(ratio * 100), int(100 + ratio * 300), 255
            else:
                r = int(50 + (ratio - 0.5) * 400)
                g = int(250 + (ratio - 0.5) * 10)
                b = 255

        elif gradient_type == 'rainbow':
            # Animated rainbow
            hue = (ratio + progress) % 1.0
            h_val = hue * 6
            if h_val < 1:
                r, g, b = 255, int(h_val * 255), 0
            elif h_val < 2:
                r, g, b = int((2 - h_val) * 255), 255, 0
            elif h_val < 3:
                r, g, b = 0, 255, int((h_val - 2) * 255)
            elif h_val < 4:
                r, g, b = 0, int((4 - h_val) * 255), 255
            elif h_val < 5:
                r, g, b = int((h_val - 4) * 255), 0, 255
            else:
                r, g, b = 255, 0, int((6 - h_val) * 255)

        elif gradient_type == 'chrome':
            # Metallic chrome with highlights
            base = 0.5 + 0.5 * math.sin((ratio * 4 + progress * 2) * math.pi)
            r = int(180 + 75 * base)
            g = int(180 + 75 * base)
            b = int(200 + 55 * base)

        else:  # Default white
            r, g, b = 255, 255, 255

        result[:, x, 0] = min(255, r)
        result[:, x, 1] = min(255, g)
        result[:, x, 2] = min(255, b)

    result[:, :, 3] = mask
    return result


def apply_neon_glow(rgba, glow_color, glow_size=20):
    """Add neon glow effect."""
    pil_img = Image.fromarray(rgba)
    alpha = pil_img.split()[3]

    # Create outer glow
    glow = alpha.filter(ImageFilter.GaussianBlur(glow_size))
    glow2 = alpha.filter(ImageFilter.GaussianBlur(glow_size // 2))

    # Create glow layer
    glow_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    # Outer glow (color)
    glow_rgba = Image.new('RGBA', pil_img.size, glow_color + (0,))
    glow_rgba.putalpha(glow)

    # Inner glow (brighter)
    inner_color = tuple(min(255, c + 100) for c in glow_color)
    inner_rgba = Image.new('RGBA', pil_img.size, inner_color + (0,))
    inner_rgba.putalpha(glow2)

    # Composite
    result = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    result = Image.alpha_composite(result, glow_rgba)
    result = Image.alpha_composite(result, inner_rgba)
    result = Image.alpha_composite(result, pil_img)

    return np.array(result)


def apply_nst_texture(mask, texture_path, progress=0):
    """Fill text with NST-styled texture."""
    h, w = mask.shape

    # Load texture
    texture = cv2.imread(str(texture_path))
    if texture is None:
        # Fallback to gradient
        return apply_gradient_fill(mask, 'rainbow', progress)

    # Resize texture to cover mask with some animation offset
    th, tw = texture.shape[:2]

    # Calculate offset for animation
    offset_x = int((progress * tw * 2) % tw)
    offset_y = int((progress * th) % th)

    # Tile texture if needed
    if tw < w or th < h:
        tiles_x = (w // tw) + 2
        tiles_y = (h // th) + 2
        tiled = np.tile(texture, (tiles_y, tiles_x, 1))
        texture = tiled
        th, tw = texture.shape[:2]

    # Extract region with offset
    x1 = offset_x % (tw - w) if tw > w else 0
    y1 = offset_y % (th - h) if th > h else 0

    cropped = texture[y1:y1 + h, x1:x1 + w]

    # Resize if needed
    if cropped.shape[:2] != (h, w):
        cropped = cv2.resize(cropped, (w, h))

    # Convert BGR to RGB and add alpha
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, 0] = cropped[:, :, 2]  # R
    result[:, :, 1] = cropped[:, :, 1]  # G
    result[:, :, 2] = cropped[:, :, 0]  # B
    result[:, :, 3] = mask

    return result


# =============================================================================
# DISTORTION EFFECTS (warp the text)
# =============================================================================

def apply_distortion(rgba, distortion_type, progress, intensity=1.0):
    """Apply morphing distortion to text."""
    h, w = rgba.shape[:2]

    if distortion_type == 'none':
        return rgba

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    if distortion_type == 'ripple':
        # Rippling water effect
        freq = 8
        amp = 15 * intensity
        phase = progress * 4 * math.pi

        x_offset = amp * np.sin(y_coords / h * freq * math.pi + phase)
        y_offset = amp * np.sin(x_coords / w * freq * math.pi + phase * 0.7)

        map_x = x_coords + x_offset
        map_y = y_coords + y_offset

    elif distortion_type == 'melt':
        # Melting/dripping effect
        melt_amount = progress * 50 * intensity

        # More melt at bottom
        melt_factor = (y_coords / h) ** 2

        x_offset = np.sin(x_coords / 30 + progress * 5) * 10 * melt_factor * intensity
        y_offset = melt_factor * melt_amount + np.sin(x_coords / 20) * 5

        map_x = x_coords + x_offset
        map_y = y_coords + y_offset

    elif distortion_type == 'breathe':
        # Pulsing/breathing effect
        scale = 1.0 + 0.1 * math.sin(progress * 4 * math.pi) * intensity

        cx, cy = w / 2, h / 2
        map_x = cx + (x_coords - cx) * scale
        map_y = cy + (y_coords - cy) * scale

    elif distortion_type == 'glitch_warp':
        # Glitchy displacement
        map_x = x_coords.copy()
        map_y = y_coords.copy()

        # Random horizontal slices
        num_slices = 5
        for _ in range(num_slices):
            if random.random() < 0.3:
                y_start = random.randint(0, h - 20)
                y_end = y_start + random.randint(10, 30)
                shift = random.randint(-30, 30) * intensity
                map_x[y_start:y_end] += shift

    elif distortion_type == 'wave_distort':
        # Wavy distortion
        freq_x = 3 + 2 * math.sin(progress * 2)
        freq_y = 2 + math.cos(progress * 3)
        amp = 20 * intensity

        x_offset = amp * np.sin(y_coords / h * freq_x * math.pi + progress * 6)
        y_offset = amp * 0.5 * np.sin(x_coords / w * freq_y * math.pi + progress * 4)

        map_x = x_coords + x_offset
        map_y = y_coords + y_offset

    else:
        return rgba

    # Apply remapping
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    result = cv2.remap(rgba, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return result


# =============================================================================
# PATH ANIMATIONS (how text moves)
# =============================================================================

def get_animation_position(animation_type, progress, frame_size, text_size):
    """Get text position and rotation for animation."""
    w, h = frame_size
    tw, th = text_size

    # Margins
    margin = 50

    if animation_type == 'orbit':
        # Orbit around center
        angle = progress * 2 * math.pi
        radius_x = w * 0.3
        radius_y = h * 0.25

        cx = w / 2 + radius_x * math.cos(angle)
        cy = h / 2 + radius_y * math.sin(angle)
        rotation = math.degrees(angle) + 90

        return (int(cx), int(cy)), rotation, 1.0

    elif animation_type == 'wave_path':
        # Follow a wave path across screen
        x = margin + progress * (w - 2 * margin)
        wave_amp = h * 0.2
        y = h / 2 + wave_amp * math.sin(progress * 4 * math.pi)

        # Slight rotation following wave
        rotation = 15 * math.cos(progress * 4 * math.pi)

        return (int(x), int(y)), rotation, 1.0

    elif animation_type == 'diagonal':
        # Diagonal sweep with fade
        if progress < 0.5:
            # Top-left to center
            t = progress * 2
            x = margin + t * (w / 2 - margin)
            y = margin + t * (h / 2 - margin)
        else:
            # Center to bottom-right
            t = (progress - 0.5) * 2
            x = w / 2 + t * (w / 2 - margin - w / 2)
            y = h / 2 + t * (h - margin - h / 2)

        return (int(x), int(y)), -15, 1.0

    elif animation_type == 'edge_crawl':
        # Crawl along edges of frame
        perimeter = 2 * w + 2 * h
        pos = progress * perimeter

        if pos < w:  # Top edge
            x, y = pos, margin
            rotation = 0
        elif pos < w + h:  # Right edge
            x, y = w - margin, pos - w
            rotation = 90
        elif pos < 2 * w + h:  # Bottom edge
            x, y = w - (pos - w - h), h - margin
            rotation = 180
        else:  # Left edge
            x, y = margin, h - (pos - 2 * w - h)
            rotation = 270

        return (int(x), int(y)), rotation, 0.8

    elif animation_type == 'float':
        # Gentle floating motion
        x = w / 2 + 100 * math.sin(progress * 3 * math.pi)
        y = h / 2 + 50 * math.cos(progress * 2 * math.pi)
        rotation = 10 * math.sin(progress * 4 * math.pi)
        scale = 1.0 + 0.1 * math.sin(progress * 2 * math.pi)

        return (int(x), int(y)), rotation, scale

    elif animation_type == 'zoom_travel':
        # Start small and far, zoom in while traveling
        t = ease_in_out(progress)

        x = margin + t * (w - 2 * margin)
        y = h / 2 + 100 * math.sin(t * 2 * math.pi)

        # Scale from small to large
        scale = 0.3 + t * 0.7
        rotation = 360 * progress  # Spin while traveling

        return (int(x), int(y)), rotation, scale

    elif animation_type == 'spiral_in':
        # Spiral from outside to center
        max_radius = min(w, h) * 0.4
        radius = max_radius * (1 - progress)
        angle = progress * 6 * math.pi  # Multiple rotations

        x = w / 2 + radius * math.cos(angle)
        y = h / 2 + radius * math.sin(angle)

        scale = 0.5 + 0.5 * progress
        rotation = math.degrees(angle)

        return (int(x), int(y)), rotation, scale

    else:
        # Default: center
        return (w // 2, h // 2), 0, 1.0


# =============================================================================
# COMPOSITING
# =============================================================================

def rotate_and_scale_rgba(rgba, angle, scale):
    """Rotate and scale RGBA image."""
    h, w = rgba.shape[:2]

    # Calculate new size
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w < 1 or new_h < 1:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    # Scale first
    scaled = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Then rotate
    if abs(angle) > 0.1:
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounds
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        bound_w = int(new_h * sin + new_w * cos)
        bound_h = int(new_h * cos + new_w * sin)

        M[0, 2] += (bound_w - new_w) / 2
        M[1, 2] += (bound_h - new_h) / 2

        rotated = cv2.warpAffine(scaled, M, (bound_w, bound_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))
        return rotated

    return scaled


def composite_rgba_on_frame(frame, rgba, center_pos):
    """Composite RGBA image onto BGR frame at center position."""
    fh, fw = frame.shape[:2]
    th, tw = rgba.shape[:2]

    cx, cy = center_pos

    # Calculate bounds
    x1 = cx - tw // 2
    y1 = cy - th // 2
    x2 = x1 + tw
    y2 = y1 + th

    # Clip to frame
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = tw - max(0, x2 - fw)
    src_y2 = th - max(0, y2 - fh)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(fw, x2)
    dst_y2 = min(fh, y2)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return frame

    result = frame.copy()

    # Get regions
    text_region = rgba[src_y1:src_y2, src_x1:src_x2]
    frame_region = result[dst_y1:dst_y2, dst_x1:dst_x2]

    # Alpha blend (RGBA to BGR)
    alpha = text_region[:, :, 3:4].astype(float) / 255
    text_bgr = text_region[:, :, :3][:, :, ::-1]  # RGB to BGR

    blended = frame_region * (1 - alpha) + text_bgr * alpha
    result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

    return result


# =============================================================================
# MAIN EFFECT PIPELINE
# =============================================================================

def apply_text_to_frame(frame, text, font_path, font_size,
                        animation, distortion, fill_style,
                        progress, texture_path=None):
    """Apply complete text effect to a frame."""
    h, w = frame.shape[:2]

    # Fade in/out
    if progress < 0.1:
        alpha = ease_out(progress / 0.1)
    elif progress > 0.9:
        alpha = ease_in((1 - progress) / 0.1)
    else:
        alpha = 1.0

    # Render text mask
    mask = render_text_mask(text, font_path, font_size)
    th, tw = mask.shape

    # Apply fill style
    if fill_style == 'nst_texture' and texture_path:
        rgba = apply_nst_texture(mask, texture_path, progress)
    elif fill_style == 'neon_glow':
        base_rgba = apply_gradient_fill(mask, 'rainbow', progress)
        neon_colors = [(255, 0, 128), (0, 255, 255), (255, 255, 0), (128, 0, 255)]
        rgba = apply_neon_glow(base_rgba, random.choice(neon_colors))
    elif fill_style.startswith('gradient_'):
        grad_type = fill_style.replace('gradient_', '')
        rgba = apply_gradient_fill(mask, grad_type, progress)
    elif fill_style == 'chrome':
        rgba = apply_gradient_fill(mask, 'chrome', progress)
    else:
        rgba = apply_gradient_fill(mask, 'rainbow', progress)

    # Apply distortion
    rgba = apply_distortion(rgba, distortion, progress)

    # Get animation position
    pos, rotation, scale = get_animation_position(
        animation, progress, (w, h), (tw, th)
    )

    # Apply rotation and scale
    rgba = rotate_and_scale_rgba(rgba, rotation, scale)

    # Apply overall alpha
    rgba[:, :, 3] = (rgba[:, :, 3] * alpha).astype(np.uint8)

    # Composite onto frame
    result = composite_rgba_on_frame(frame, rgba, pos)

    return result


def find_nst_textures(output_dir='/app/output'):
    """Find styled images to use as textures."""
    patterns = [
        f'{output_dir}/**/*tile*.jpg',
        f'{output_dir}/**/*styled*.jpg',
        f'{output_dir}/**/*candy*.jpg',
        f'{output_dir}/**/*mosaic*.jpg',
    ]

    textures = []
    for pattern in patterns:
        textures.extend(glob.glob(pattern, recursive=True))

    # Filter to reasonable sizes
    valid = []
    for t in textures[:50]:  # Limit search
        try:
            img = cv2.imread(t)
            if img is not None and img.shape[0] > 200 and img.shape[1] > 200:
                valid.append(t)
        except:
            pass

    return valid if valid else None


def process_video(input_path, output_path, phrases, seed=None,
                  animation=None, distortion=None, fill=None):
    """Process video with advanced text overlays."""
    print(f"\n{'='*60}")
    print(f"CRYPTIC TEXT OVERLAY v2")
    print(f"{'='*60}")

    if seed is not None:
        random.seed(seed)

    # Find NST textures
    textures = find_nst_textures()
    if textures:
        print(f"  Found {len(textures)} NST textures for fills")

    # Read video
    print(f"\n[1/3] Reading video: {input_path}")
    frames, fps, size = read_video(input_path)
    total_frames = len(frames)

    if total_frames == 0:
        print("[error] No frames")
        return None

    # Calculate timing
    frames_per_phrase = total_frames // len(phrases)

    print(f"\n[2/3] Applying text effects...")
    print(f"  {len(phrases)} phrases, {frames_per_phrase} frames each")

    for i, phrase in enumerate(phrases):
        # Select effects (or use specified)
        p_animation = animation if animation else random.choice(ANIMATIONS)
        p_distortion = distortion if distortion else random.choice(DISTORTIONS)
        p_fill = fill if fill else random.choice(FILLS)
        p_font = random.choice(FONTS)

        # Select texture for NST fill
        p_texture = random.choice(textures) if textures and p_fill == 'nst_texture' else None

        # Calculate font size
        font_size = min(size[0] // len(phrase), size[1] // 4)
        font_size = max(48, min(font_size, 200))

        print(f"  [{i+1}/{len(phrases)}] '{phrase}'")
        print(f"      animation={p_animation}, distortion={p_distortion}, fill={p_fill}")

        start_frame = i * frames_per_phrase
        end_frame = min(start_frame + frames_per_phrase, total_frames)

        for f in range(start_frame, end_frame):
            progress = (f - start_frame) / frames_per_phrase

            frames[f] = apply_text_to_frame(
                frames[f], phrase, p_font, font_size,
                p_animation, p_distortion, p_fill,
                progress, p_texture
            )

    # Write output
    print(f"\n[3/3] Writing video: {output_path}")
    write_video(frames, output_path, fps, size)

    duration = total_frames / fps
    print(f"\n{'='*60}")
    print(f"COMPLETE: {output_path}")
    print(f"  Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"{'='*60}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Advanced artistic text overlay for videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Random effects
  python cryptic_text.py --input video.mp4 --phrases "DREAM,CHAOS" --output out.mp4

  # Specific effects
  python cryptic_text.py --input video.mp4 --phrases "AWAKEN" \\
      --animation orbit --distortion ripple --fill nst_texture

Animations: {', '.join(ANIMATIONS)}
Distortions: {', '.join(DISTORTIONS)}
Fills: {', '.join(FILLS)}
        """
    )

    parser.add_argument('--input', required=False, help='Input video')
    parser.add_argument('--output', required=False, help='Output video')
    parser.add_argument('--phrases', required=False, help='Comma-separated phrases')
    parser.add_argument('--animation', choices=ANIMATIONS, help='Animation style')
    parser.add_argument('--distortion', choices=DISTORTIONS, help='Distortion effect')
    parser.add_argument('--fill', choices=FILLS, help='Fill style')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--list', action='store_true', help='List options')

    args = parser.parse_args()

    if args.list:
        print("Animations (how text moves):")
        for a in ANIMATIONS:
            print(f"  - {a}")
        print("\nDistortions (how text warps):")
        for d in DISTORTIONS:
            print(f"  - {d}")
        print("\nFills (text appearance):")
        for f in FILLS:
            print(f"  - {f}")
        return 0

    if not args.input or not args.output or not args.phrases:
        parser.error("--input, --output, and --phrases are required")

    phrases = [p.strip() for p in args.phrases.split(',') if p.strip()]

    if not phrases:
        print("[error] No phrases")
        return 1

    result = process_video(
        args.input, args.output, phrases,
        seed=args.seed,
        animation=args.animation,
        distortion=args.distortion,
        fill=args.fill
    )

    return 0 if result else 1


if __name__ == '__main__':
    exit(main())
