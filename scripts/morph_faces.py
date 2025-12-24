#!/usr/bin/env python3
"""
MorphFaces - Multi-face zoom-blend video pipeline

For each image:
1. Detect faces with > 3% coverage
2. Run Magenta style transfer with all 7 tile configs for each face
3. Create video: zoom 0% to 400% while blending tiles, then blend to next face

Usage:
    docker-compose run --rm style bash -lc "python /app/scripts/morph_faces.py --input_dir /app/input/self_style_samples"
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from PIL import Image
import subprocess
import tempfile
import shutil

# Import from morph_v2
from morph_v2 import (
    detect_faces, extract_face_region, run_magenta_style, run_pytorch_style,
    TILE_CONFIGS, PYTORCH_MODELS, _smootherstep
)


def save_run_log(output_path, args_dict, faces_info, output_video, duration_sec):
    """Save a JSON log of the run parameters and results."""
    log_path = output_path.parent / f"{output_path.stem}_run.json"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "script": "morph_faces.py",
        "input_image": str(args_dict.get('image_path', '')),
        "output_video": str(output_video) if output_video else None,
        "duration_seconds": round(duration_sec, 2),
        "parameters": {
            "min_coverage": args_dict.get('min_coverage'),
            "confidence_threshold": args_dict.get('confidence_threshold'),
            "iou_threshold": args_dict.get('iou_threshold'),
            "scale": args_dict.get('scale'),
            "blend": args_dict.get('blend'),
            "fps": args_dict.get('fps'),
            "zoom_in_duration": args_dict.get('zoom_in_duration'),
            "zoom_out_duration": args_dict.get('zoom_out_duration'),
            "transition_duration": args_dict.get('transition_duration'),
            "morph_time": args_dict.get('morph_time'),
            "min_zoom": args_dict.get('min_zoom'),
            "max_zoom": args_dict.get('max_zoom'),
            "vertical": args_dict.get('vertical'),
            "face_padding": args_dict.get('face_padding'),
            "pytorch_style": args_dict.get('pytorch_style'),
            "blob_mode": args_dict.get('blob_mode'),
            "num_blobs": args_dict.get('num_blobs'),
            "blob_frequency": args_dict.get('blob_frequency'),
            "blob_speed": args_dict.get('blob_speed'),
            "blob_feather": args_dict.get('blob_feather'),
            "selected_models": args_dict.get('selected_models'),
            "selected_tiles": args_dict.get('selected_tiles'),
        },
        "faces_detected": len(faces_info),
        "faces": faces_info,
    }

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  [log] Saved run parameters to {log_path.name}")
    return log_path


def create_soft_multi_blob_masks(H, W, frame_idx, num_blobs=4, frequency=2.5, speed=1.0, seed=42, feather=0.3):
    """Create soft multi-blob masks with gaussian-like smooth transitions between regions.

    Returns array of shape (num_blobs, H, W) with soft blend weights for each blob.
    """
    time_offset = frame_idx * speed * 0.02

    y_norm = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, W, dtype=np.float32)[None, :]

    blob_values = np.zeros((num_blobs, H, W), dtype=np.float32)

    for blob_idx in range(num_blobs):
        np.random.seed(seed + blob_idx * 1000)
        noise = np.zeros((H, W), dtype=np.float32)
        blob_phase = blob_idx * 2 * np.pi / num_blobs

        for octave in range(4):
            freq = frequency * (2 ** octave)
            amp = 1.0 / (1.5 ** octave)
            phase_x = np.random.random() * 2 * np.pi
            phase_y = np.random.random() * 2 * np.pi
            phase_t = np.random.random() * 2 * np.pi

            noise += amp * np.sin(y_norm * freq * np.pi + phase_y + time_offset * (1 + octave * 0.3) + blob_phase)
            noise += amp * np.sin(x_norm * freq * np.pi + phase_x + time_offset * (1.2 + octave * 0.2) + blob_phase)
            noise += amp * 0.5 * np.sin((x_norm + y_norm) * freq * np.pi + phase_t + time_offset * 1.5 + blob_phase)

        blob_values[blob_idx] = noise

    # Softmax-like weighting with temperature control for gaussian-like feathering
    temperature = max(0.1, feather * 5)  # Higher = softer boundaries
    blob_values = blob_values - blob_values.max(axis=0, keepdims=True)  # Numerical stability
    exp_values = np.exp(blob_values / temperature)
    weights = exp_values / (exp_values.sum(axis=0, keepdims=True) + 1e-6)

    return weights.astype(np.float32)


def get_blended_image(images, position):
    """Get smoothly interpolated image at fractional position."""
    num_images = len(images)
    pos = position % num_images

    idx1 = int(pos)
    idx2 = (idx1 + 1) % num_images
    blend = pos - idx1

    img1 = images[idx1]
    img2 = images[idx2]

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return cv2.addWeighted(img1, 1 - blend, img2, blend, 0)


def create_blob_face_morph(
    styled_images,
    original_image,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=3.0,
    fps=24,
    morph_time=0.5,
    zoom_center=None,
    num_blobs=4,
    blob_frequency=2.5,
    blob_speed=1.0,
    blob_feather=0.3,
    blend_to_original=True
):
    """Create face morph video with blob-based gaussian blending.

    Multiple blob regions show different styled images at different phases,
    creating organic blending between styles with smooth gaussian transitions.

    Args:
        styled_images: List of styled image paths
        original_image: Path to original unstyled image
        target_size: (width, height) of output
        min_zoom: Ending zoom level
        max_zoom: Starting zoom level
        fps: Frames per second
        morph_time: Seconds per image transition (0.5 = half second between images)
        zoom_center: (x, y) normalized center for zoom
        num_blobs: Number of blob regions (4-8 recommended)
        blob_frequency: Blob detail level (2-4 recommended)
        blob_speed: Blob animation speed
        blob_feather: Softness of blob boundaries (0.2-0.5 recommended)
        blend_to_original: If True, blend to original at end

    Returns:
        List of frames
    """
    # Load all styled images
    images = []
    for img_path in styled_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)

    if not images:
        return []

    # Load original image
    if isinstance(original_image, np.ndarray):
        orig_img = original_image.copy()
    else:
        orig_img = cv2.imread(str(original_image))

    num_images = len(images)
    safe_min_zoom = calculate_safe_zoom(zoom_center, min_zoom)

    # Calculate duration: each image pair transition takes morph_time seconds
    # Total morph phase is 80% of video, blend to original is 20%
    morph_duration = num_images * morph_time
    if blend_to_original:
        total_duration = morph_duration / 0.8  # Add 20% for original blend
    else:
        total_duration = morph_duration

    num_frames = int(total_duration * fps)
    morph_end_t = 0.8 if blend_to_original else 1.0

    frames = []

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Zoom OUT: from max_zoom to safe_min_zoom
        zoom = max_zoom - t_eased * (max_zoom - safe_min_zoom)

        if t < morph_end_t:
            # Phase 1: Blob-based morph through styled images
            morph_t = t / morph_end_t

            # Get blob masks for this frame
            H, W = images[0].shape[:2]
            blob_masks = create_soft_multi_blob_masks(
                H, W, frame_idx,
                num_blobs=num_blobs,
                frequency=blob_frequency,
                speed=blob_speed,
                feather=blob_feather
            )

            # Create blended frame using blobs
            blended = np.zeros((H, W, 3), dtype=np.float32)

            for blob_idx in range(num_blobs):
                # Each blob moves through images at different phase offset
                phase_offset = blob_idx / num_blobs
                # Position in image sequence
                pos = morph_t * (num_images - 1) + phase_offset * num_images

                # Get smooth blended image at this position
                img = get_blended_image(images, pos)

                # Resize if needed
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))

                # Apply blob mask with gaussian weighting
                mask_3ch = blob_masks[blob_idx][:, :, np.newaxis]
                blended += img.astype(np.float32) * mask_3ch

            styled = blended.astype(np.uint8)
            frame = apply_zoom_crop(styled, zoom, zoom_center, target_size)

        else:
            # Phase 2: Blend from styled to original
            blend_t = (t - morph_end_t) / (1.0 - morph_end_t)
            blend_t_eased = _smootherstep(blend_t)

            # Use final blended styled frame
            H, W = images[0].shape[:2]
            blob_masks = create_soft_multi_blob_masks(
                H, W, frame_idx,
                num_blobs=num_blobs,
                frequency=blob_frequency,
                speed=blob_speed,
                feather=blob_feather
            )

            blended = np.zeros((H, W, 3), dtype=np.float32)
            for blob_idx in range(num_blobs):
                phase_offset = blob_idx / num_blobs
                pos = (num_images - 1) + phase_offset * num_images
                img = get_blended_image(images, pos)
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))
                mask_3ch = blob_masks[blob_idx][:, :, np.newaxis]
                blended += img.astype(np.float32) * mask_3ch

            styled_frame = apply_zoom_crop(blended.astype(np.uint8), zoom, zoom_center, target_size)
            orig_frame = apply_zoom_crop(orig_img, zoom, zoom_center, target_size)

            frame = cv2.addWeighted(styled_frame, 1 - blend_t_eased, orig_frame, blend_t_eased, 0)

        frames.append(frame)

    return frames


def filter_overlapping_faces(faces, iou_threshold=0.3):
    """Filter out overlapping face detections using Non-Maximum Suppression.

    Keeps the higher confidence detection when faces overlap.

    Args:
        faces: List of face dictionaries with 'bbox' and 'confidence'
        iou_threshold: IoU threshold above which faces are considered duplicates

    Returns:
        Filtered list of faces
    """
    if len(faces) <= 1:
        return faces

    # Sort by confidence (highest first)
    faces = sorted(faces, key=lambda f: f.get('confidence', 0), reverse=True)

    def calculate_iou(box1, box2):
        """Calculate Intersection over Union between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to x1, y1, x2, y2 format
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2

        # Calculate intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    kept_faces = []
    for face in faces:
        # Check if this face overlaps with any already kept face
        dominated = False
        for kept in kept_faces:
            iou = calculate_iou(face['bbox'], kept['bbox'])
            if iou > iou_threshold:
                dominated = True
                break

        if not dominated:
            kept_faces.append(face)

    # Re-number IDs
    for i, face in enumerate(kept_faces):
        face['id'] = i + 1

    return kept_faces


def calculate_safe_zoom(center, min_zoom=1.0):
    """Calculate minimum zoom that keeps crop within image bounds for a given center."""
    if center is None:
        return min_zoom
    cx, cy = center
    min_dist = min(cx, 1 - cx, cy, 1 - cy)
    if min_dist <= 0:
        return 10.0
    return max(min_zoom, 0.5 / min_dist)


def apply_zoom_crop(img, zoom, center, target_size):
    """Apply zoom crop centered on a point."""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)

    if center is not None:
        cx = int(center[0] * w)
        cy = int(center[1] * h)
    else:
        cx, cy = w // 2, h // 2

    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def create_face_zoom_out(
    styled_images,
    original_image,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=4.0,
    fps=24,
    duration=5.0,
    zoom_center=None
):
    """Zoom out from max to min while morphing tiles, ending with blend to original.

    Flow:
    - First 80%: Zoom out while morphing from smallest tile (detailed) to largest (smooth)
    - Last 20%: Continue zoom, blend from styled to original

    Args:
        styled_images: List of styled image paths (index 0=largest tile, index -1=smallest tile)
        original_image: Path to original unstyled image
        target_size: (width, height) of output
        min_zoom: Ending zoom level (zoomed out)
        max_zoom: Starting zoom level (zoomed in)
        fps: Frames per second
        duration: Duration in seconds
        zoom_center: (x, y) normalized center for zoom

    Returns:
        List of frames
    """
    # Load all styled images
    images = []
    for img_path in styled_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)

    if not images:
        return []

    # Load original image
    if isinstance(original_image, np.ndarray):
        orig_img = original_image.copy()
    else:
        orig_img = cv2.imread(str(original_image))

    num_images = len(images)
    safe_min_zoom = calculate_safe_zoom(zoom_center, min_zoom)

    frames = []
    num_frames = int(duration * fps)

    # Timing: 80% for tile morph, 20% for blend to original
    morph_end = 0.8

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Zoom OUT: from max_zoom to safe_min_zoom (full duration)
        zoom = max_zoom - t_eased * (max_zoom - safe_min_zoom)

        if t < morph_end:
            # Phase 1: Morph through tiles
            morph_t = t / morph_end
            morph_t_eased = _smootherstep(morph_t)

            # Morph from smallest tile (last) to largest tile (first)
            img_pos = (1 - morph_t_eased) * (num_images - 1)
            img_idx1 = int(img_pos)
            img_idx2 = min(img_idx1 + 1, num_images - 1)
            blend_alpha = img_pos - img_idx1

            img1 = images[img_idx1]
            img2 = images[img_idx2]

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            if (h1, w1) != (h2, w2):
                img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)

            styled = cv2.addWeighted(img1, 1 - blend_alpha, img2, blend_alpha, 0)
            frame = apply_zoom_crop(styled, zoom, zoom_center, target_size)
        else:
            # Phase 2: Blend from largest tile to original
            blend_t = (t - morph_end) / (1.0 - morph_end)
            blend_t_eased = _smootherstep(blend_t)

            styled_frame = apply_zoom_crop(images[0], zoom, zoom_center, target_size)  # Largest tile
            orig_frame = apply_zoom_crop(orig_img, zoom, zoom_center, target_size)

            frame = cv2.addWeighted(styled_frame, 1 - blend_t_eased, orig_frame, blend_t_eased, 0)

        frames.append(frame)

    return frames


def create_face_crossfade(
    original_image,
    styled_images_to,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=4.0,
    fps=24,
    duration=1.0,
    center_from=None,
    center_to=None
):
    """Crossfade from original (zoomed out) to next face styled (zooming in).

    Face A ends at min zoom with original image, Face B starts at max zoom with smallest tile.
    This transition handles both the crossfade and the zoom change.

    Args:
        original_image: Path to original unstyled image
        styled_images_to: List of styled paths for dest face (use last/smallest tile)
        target_size: (width, height) of output
        min_zoom: Starting zoom level (zoomed out, where face A ended)
        max_zoom: Ending zoom level (zoomed in, where face B will start)
        fps: Frames per second
        duration: Duration in seconds
        center_from: Source face center (for original)
        center_to: Destination face center

    Returns:
        List of frames
    """
    # Load original image
    if isinstance(original_image, np.ndarray):
        orig_img = original_image.copy()
    else:
        orig_img = cv2.imread(str(original_image))

    if orig_img is None:
        return []

    # Load last (smallest tile) from dest face - where next zoom out will start
    styled_to = None
    for p in styled_images_to:
        img = cv2.imread(str(p))
        if img is not None:
            styled_to = img  # Keep going to get last one

    if styled_to is None:
        return []

    safe_min_from = calculate_safe_zoom(center_from, min_zoom)
    safe_min_to = calculate_safe_zoom(center_to, min_zoom)

    frames = []
    num_frames = int(duration * fps)

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Face A (original): stays at min zoom, fading out
        frame_from = apply_zoom_crop(orig_img, safe_min_from, center_from, target_size)

        # Face B (styled): zooms from min to max, fading in
        zoom_to = safe_min_to + t_eased * (max_zoom - safe_min_to)
        frame_to = apply_zoom_crop(styled_to, zoom_to, center_to, target_size)

        frame = cv2.addWeighted(frame_from, 1 - t_eased, frame_to, t_eased, 0)
        frames.append(frame)

    return frames


def create_morph_at_zoom(
    styled_images,
    target_size=(720, 1280),
    max_zoom=4.0,
    fps=24,
    duration=2.0,
    zoom_center=None,
    reverse=False
):
    """Morph through tile configurations at max zoom (no zoom change).

    Args:
        styled_images: List of styled image paths
        target_size: (width, height) of output
        max_zoom: Zoom level to maintain
        fps: Frames per second
        duration: Duration in seconds
        zoom_center: (x, y) normalized center
        reverse: If True, morph from last to first (for continuity after transitions)

    Returns:
        List of frames
    """
    # Load styled images
    images = []
    for img_path in styled_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)

    if not images:
        return []

    num_images = len(images)
    frames = []
    num_frames = int(duration * fps)

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Blend through all styled images
        if reverse:
            # Go from last to first (for continuity after transitions)
            img_pos = (1 - t_eased) * (num_images - 1)
        else:
            img_pos = t_eased * (num_images - 1)

        img_idx1 = int(img_pos)
        img_idx2 = min(img_idx1 + 1, num_images - 1)
        blend_alpha = img_pos - img_idx1

        img1 = images[img_idx1]
        img2 = images[img_idx2]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if (h1, w1) != (h2, w2):
            img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_LINEAR)

        styled = cv2.addWeighted(img1, 1 - blend_alpha, img2, blend_alpha, 0)
        frame = apply_zoom_crop(styled, max_zoom, zoom_center, target_size)
        frames.append(frame)

    return frames


def create_face_blend_transition(
    styled_images_from,
    styled_images_to,
    target_size=(720, 1280),
    max_zoom=4.0,
    fps=24,
    duration=2.0,
    center_from=None,
    center_to=None
):
    """Crossfade between two faces at max zoom, both morphing through tiles.

    Each face continues morphing through its tile configurations while crossfading.

    Args:
        styled_images_from: List of styled paths for source face
        styled_images_to: List of styled paths for dest face
        target_size: (width, height) of output
        max_zoom: Zoom level to maintain
        fps: Frames per second
        duration: Duration in seconds
        center_from: Source face center
        center_to: Destination face center

    Returns:
        List of frames
    """
    # Load all styled images from both faces
    images_from = []
    for p in styled_images_from:
        img = cv2.imread(str(p))
        if img is not None:
            images_from.append(img)

    images_to = []
    for p in styled_images_to:
        img = cv2.imread(str(p))
        if img is not None:
            images_to.append(img)

    if not images_from or not images_to:
        return []

    num_from = len(images_from)
    num_to = len(images_to)

    frames = []
    num_frames = int(duration * fps)

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Face A: morph in REVERSE (last to first) since previous phase ended at last tile
        # This ensures continuity from intro/previous morph
        img_pos_from = (1 - t_eased) * (num_from - 1)  # Reverse: last→first
        idx1_from = int(img_pos_from)
        idx2_from = min(idx1_from + 1, num_from - 1)
        alpha_from = img_pos_from - idx1_from

        img1_from = images_from[idx1_from]
        img2_from = images_from[idx2_from]
        h1, w1 = img1_from.shape[:2]
        h2, w2 = img2_from.shape[:2]
        if (h1, w1) != (h2, w2):
            img2_from = cv2.resize(img2_from, (w1, h1), interpolation=cv2.INTER_LINEAR)
        styled_from = cv2.addWeighted(img1_from, 1 - alpha_from, img2_from, alpha_from, 0)

        # Face B: morph in REVERSE (last to first) to end at first tile
        # Next morph phase will start from first tile (going first→last)
        img_pos_to = (1 - t_eased) * (num_to - 1)  # Reverse: last→first
        idx1_to = int(img_pos_to)
        idx2_to = min(idx1_to + 1, num_to - 1)
        alpha_to = img_pos_to - idx1_to

        img1_to = images_to[idx1_to]
        img2_to = images_to[idx2_to]
        h1, w1 = img1_to.shape[:2]
        h2, w2 = img2_to.shape[:2]
        if (h1, w1) != (h2, w2):
            img2_to = cv2.resize(img2_to, (w1, h1), interpolation=cv2.INTER_LINEAR)
        styled_to = cv2.addWeighted(img1_to, 1 - alpha_to, img2_to, alpha_to, 0)

        # Apply zoom at respective centers
        frame_from = apply_zoom_crop(styled_from, max_zoom, center_from, target_size)
        frame_to = apply_zoom_crop(styled_to, max_zoom, center_to, target_size)

        # Crossfade between the two morphing faces
        frame = cv2.addWeighted(frame_from, 1 - t_eased, frame_to, t_eased, 0)
        frames.append(frame)

    return frames


def create_outro_zoom(
    styled_images,
    original_image,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=4.0,
    fps=24,
    duration=2.0,
    zoom_center=None
):
    """Create outro: at max zoom, zoom out while blending styled to original.

    Args:
        styled_images: List of styled image paths (will use last one)
        original_image: Path or numpy array of original image
        target_size: (width, height) of output
        min_zoom: Ending zoom level
        max_zoom: Starting zoom level
        fps: Frames per second
        duration: Duration in seconds
        zoom_center: (x, y) normalized center

    Returns:
        List of frames
    """
    # Load original
    if isinstance(original_image, np.ndarray):
        orig_img = original_image.copy()
    else:
        orig_img = cv2.imread(str(original_image))

    if orig_img is None:
        return []

    # Load last styled image
    styled_img = None
    for p in styled_images:
        img = cv2.imread(str(p))
        if img is not None:
            styled_img = img

    if styled_img is None:
        return []

    safe_min_zoom = calculate_safe_zoom(zoom_center, min_zoom)

    frames = []
    num_frames = int(duration * fps)

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Zoom decreases from max to safe_min
        zoom = max_zoom - t_eased * (max_zoom - safe_min_zoom)

        # Apply zoom to both
        styled_frame = apply_zoom_crop(styled_img, zoom, zoom_center, target_size)
        orig_frame = apply_zoom_crop(orig_img, zoom, zoom_center, target_size)

        # Blend: start with styled (t=0), end with original (t=1)
        frame = cv2.addWeighted(styled_frame, 1 - t_eased, orig_frame, t_eased, 0)
        frames.append(frame)

    return frames


def process_image(
    image_path,
    output_dir,
    min_coverage=3.0,
    confidence_threshold=0.5,
    iou_threshold=0.3,
    scale=1440,
    blend=0.95,
    fps=24,
    zoom_in_duration=2.0,
    zoom_out_duration=2.0,
    transition_duration=1.0,
    min_zoom=1.0,
    max_zoom=4.0,
    vertical=True,
    force=False,
    face_padding=0.6,
    pytorch_style=True,
    morph_time=0.0,
    blob_mode=False,
    num_blobs=4,
    blob_frequency=2.5,
    blob_speed=1.0,
    blob_feather=0.3,
    selected_models=None,
    selected_tiles=None
):
    """Process a single image: detect faces, style each, create morph video.

    Video flow:
    1. Intro: Start on original, zoom into first face while blending to styled
    2. Per face: Morph through tile configurations at max zoom
    3. Transitions: Blend between faces at max zoom (pan between centers)
    4. Outro: Zoom out from last face while blending back to original

    Args:
        image_path: Path to input image
        output_dir: Base output directory
        min_coverage: Minimum face coverage percentage (default 3%)
        confidence_threshold: Minimum confidence for face detection
        iou_threshold: IoU threshold for filtering overlapping faces
        scale: Resolution for styled images
        blend: Style blend ratio
        fps: Video framerate
        zoom_in_duration: Seconds for intro zoom in / outro zoom out
        zoom_out_duration: Seconds to morph tiles per face
        transition_duration: Seconds to transition between faces
        min_zoom: Starting/ending zoom level (1.0 = full view)
        max_zoom: Zoom level while morphing between faces
        vertical: True for vertical video (720x1280)
        force: Force regenerate existing outputs
    """
    image_path = Path(image_path)
    name = image_path.stem

    # Output directories
    base_output = Path(output_dir) / name
    styled_dir = base_output / "styled"
    work_dir = base_output / "work"

    base_output.mkdir(parents=True, exist_ok=True)
    styled_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    target_size = (720, 1280) if vertical else (1280, 720)

    print(f"\n{'='*60}")
    print(f"MORPH FACES: {name}")
    print(f"{'='*60}")

    # Step 1: Detect faces
    print(f"\n[1/3] Detecting faces (min coverage: {min_coverage}%, confidence: {confidence_threshold})...")
    faces = detect_faces(str(image_path), confidence_threshold=confidence_threshold)

    # Filter by coverage
    valid_faces = [f for f in faces if f['coverage'] >= min_coverage]

    if not valid_faces:
        print(f"[skip] No faces with >= {min_coverage}% coverage found")
        return None

    # Filter overlapping faces (keep higher confidence when faces overlap)
    pre_filter_count = len(valid_faces)
    valid_faces = filter_overlapping_faces(valid_faces, iou_threshold=iou_threshold)
    if len(valid_faces) < pre_filter_count:
        print(f"  Filtered {pre_filter_count - len(valid_faces)} overlapping detection(s)")

    print(f"  Found {len(valid_faces)} face(s) with >= {min_coverage}% coverage:")
    for f in valid_faces:
        x, y, w, h = f['bbox']
        conf = f.get('confidence', 0)
        print(f"    Face #{f['id']}: {w}x{h} ({f['coverage']:.1f}% coverage, {conf:.0%} confidence)")

    # Get original image dimensions for normalizing face centers
    orig_img = cv2.imread(str(image_path))
    orig_h, orig_w = orig_img.shape[:2]

    # Step 2: For each face, extract and run all tile configs
    # Build list of style sources (original + PyTorch pre-styled variants)
    # Use selected_models if provided, otherwise fall back to PYTORCH_MODELS
    models_to_use = selected_models if selected_models else PYTORCH_MODELS
    tiles_to_use = selected_tiles if selected_tiles else TILE_CONFIGS

    style_sources = [('none', None)]  # (prefix, model_name)
    if pytorch_style and models_to_use:
        for model_name in models_to_use:
            style_sources.append((model_name, model_name))

    num_configs = len(tiles_to_use) * len(style_sources)
    print(f"\n[2/3] Styling each face with {num_configs} configurations...")
    if pytorch_style and models_to_use:
        model_names = [m for m in models_to_use]
        print(f"       ({len(tiles_to_use)} tiles × {len(style_sources)} style sources: none + {len(model_names)} models)")
        if len(model_names) <= 10:
            print(f"       Models: {', '.join(model_names)}")

    face_styled_images = {}  # face_id -> list of styled image paths
    face_last_images = {}    # face_id -> last styled image (for transitions)
    face_first_images = {}   # face_id -> first styled image (for transitions)
    face_centers = {}        # face_id -> normalized (x, y) center coordinates

    for face in valid_faces:
        face_id = face['id']
        face_dir = styled_dir / f"face{face_id}"
        face_dir.mkdir(exist_ok=True)

        # Store normalized face center for zoom targeting
        face_cx, face_cy = face['center']
        face_centers[face_id] = (face_cx / orig_w, face_cy / orig_h)

        # Extract face crop (original, no pre-styling)
        face_crop_path = work_dir / f"face{face_id}_crop.jpg"
        if not face_crop_path.exists() or force:
            extract_face_region(image_path, face['bbox'], face_crop_path, padding_pct=face_padding)

        # Create PyTorch pre-styled face crops if enabled
        pytorch_crops = {}  # model_name -> path
        if pytorch_style and models_to_use:
            for model_name in models_to_use:
                pytorch_crop_path = work_dir / f"face{face_id}_crop_{model_name}.jpg"
                if not pytorch_crop_path.exists() or force:
                    print(f"    [pytorch] Pre-styling face {face_id} with {model_name}...")
                    success, _ = run_pytorch_style(
                        face_crop_path, pytorch_crop_path,
                        model_name=model_name, scale=scale, blend=0.95
                    )
                    if success:
                        pytorch_crops[model_name] = pytorch_crop_path
                elif pytorch_crop_path.exists():
                    pytorch_crops[model_name] = pytorch_crop_path

        # Run all tile configs for each style source
        styled_paths = []
        for prefix, model_name in style_sources:
            # Determine which style source to use
            if prefix == 'none':
                style_source = face_crop_path
                name_prefix = ""
            else:
                if model_name not in pytorch_crops:
                    print(f"    [skip] {model_name} pre-style failed, skipping")
                    continue
                style_source = pytorch_crops[model_name]
                name_prefix = f"{model_name}_"

            for tile, overlap in tiles_to_use:
                output_name = f"face{face_id}_{name_prefix}tile{tile}_overlap{overlap}.jpg"
                output_path = face_dir / output_name

                if output_path.exists() and not force:
                    print(f"    [skip] {output_name}")
                else:
                    # Style the FACE CROP (not full image) using style_source
                    # For 'none' prefix: self-style the face crop
                    # For pytorch prefix: use pytorch-styled face as both content and style
                    success = run_magenta_style(
                        face_crop_path, style_source, output_path,
                        tile=tile, overlap=overlap, scale=scale, blend=blend
                    )
                    if success:
                        print(f"    -> {output_name}")

                if output_path.exists():
                    styled_paths.append(output_path)

        # Reverse order: largest tile (512, smoothest) first, smallest tile (128, most detailed) last
        # This way at max zoom we show the most detailed/stylized version
        styled_paths_reversed = list(reversed(styled_paths))
        face_styled_images[face_id] = styled_paths_reversed

        # Store first and last for transitions
        if styled_paths_reversed:
            face_first_images[face_id] = cv2.imread(str(styled_paths_reversed[0]))
            face_last_images[face_id] = cv2.imread(str(styled_paths_reversed[-1]))

    # Step 3: Create combined video
    # Simple flow:
    # 1. Each face: Start zoomed in at max zoom + smallest tile, zoom out to min zoom + largest tile
    # 2. Crossfade to next face and repeat

    # Calculate zoom duration based on morph_time if specified
    # morph_time = seconds per image transition
    first_face_id = list(face_styled_images.keys())[0] if face_styled_images else None
    num_styled = len(face_styled_images[first_face_id]) if first_face_id else 0

    if blob_mode:
        # Blob mode with gaussian blending - use morph_time per image
        actual_morph_time = morph_time if morph_time > 0 else 0.5  # Default 0.5s per image in blob mode
        print(f"\n[3/3] Creating BLOB face morph ({num_styled} images × {actual_morph_time}s, {num_blobs} blobs, feather={blob_feather})...")
    elif morph_time > 0:
        # Calculate zoom duration from morph_time
        zoom_duration = (num_styled * morph_time) / 0.8
        print(f"\n[3/3] Creating face zoom-out video ({num_styled} images × {morph_time}s = {zoom_duration:.1f}s per face)...")
    else:
        zoom_duration = zoom_in_duration + zoom_out_duration  # Combine both params for zoom out duration
        print(f"\n[3/3] Creating face zoom-out video...")

    all_frames = []
    face_ids = sorted(face_styled_images.keys())

    for i, face_id in enumerate(face_ids):
        styled_paths = face_styled_images[face_id]

        if not styled_paths:
            continue

        zoom_center = face_centers.get(face_id)
        is_last = (i == len(face_ids) - 1)

        if blob_mode:
            # Use blob-based gaussian blending
            actual_morph_time = morph_time if morph_time > 0 else 0.5
            total_duration = (len(styled_paths) * actual_morph_time) / 0.8
            print(f"  [face {face_id}] Blob morph {total_duration:.1f}s: {max_zoom}x -> {min_zoom}x with {num_blobs} blobs (center: {zoom_center[0]:.2f}, {zoom_center[1]:.2f})...")
            zoom_frames = create_blob_face_morph(
                styled_paths,
                image_path,
                target_size=target_size,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                fps=fps,
                morph_time=actual_morph_time,
                zoom_center=zoom_center,
                num_blobs=num_blobs,
                blob_frequency=blob_frequency,
                blob_speed=blob_speed,
                blob_feather=blob_feather,
                blend_to_original=True
            )
        else:
            # Original zoom-out mode
            if morph_time > 0:
                zoom_duration = (len(styled_paths) * morph_time) / 0.8
            print(f"  [face {face_id}] Zoom out {zoom_duration}s: {max_zoom}x -> {min_zoom}x + morph -> original (center: {zoom_center[0]:.2f}, {zoom_center[1]:.2f})...")
            zoom_frames = create_face_zoom_out(
                styled_paths,
                image_path,  # Original image for blend at end
                target_size=target_size,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                fps=fps,
                duration=zoom_duration,
                zoom_center=zoom_center
            )
        all_frames.extend(zoom_frames)

        # CROSSFADE to next face (if not last)
        if not is_last and transition_duration > 0:
            next_face_id = face_ids[i + 1]
            next_center = face_centers.get(next_face_id)
            next_styled = face_styled_images.get(next_face_id, [])

            if next_center and next_styled:
                print(f"  [fade] Original -> Face {next_face_id} ({transition_duration}s)...")
                fade_frames = create_face_crossfade(
                    image_path,  # Original image (where zoom out ended)
                    next_styled,
                    target_size=target_size,
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                    fps=fps,
                    duration=transition_duration,
                    center_from=zoom_center,
                    center_to=next_center
                )
                all_frames.extend(fade_frames)

    if not all_frames:
        print("[error] No frames generated")
        return None

    # Write video
    if blob_mode:
        output_video = base_output / f"{name}_faces_blob.mp4"
    else:
        output_video = base_output / f"{name}_faces_zoom.mp4"
    temp_video = work_dir / "temp_raw.mp4"

    # Write raw video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video), fourcc, fps, target_size)

    for frame in all_frames:
        out.write(frame)
    out.release()

    # Convert to H.264
    print(f"  [encode] Converting to H.264...")
    subprocess.run([
        'ffmpeg', '-y', '-i', str(temp_video),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(output_video)
    ], capture_output=True)

    total_duration = len(all_frames) / fps
    print(f"\n{'='*60}")
    print(f"COMPLETE: {output_video}")
    print(f"  Faces: {len(valid_faces)}")
    print(f"  Frames: {len(all_frames)}")
    print(f"  Duration: {total_duration:.1f}s")
    print(f"{'='*60}")

    # Save run log
    args_dict = {
        'image_path': str(image_path),
        'min_coverage': min_coverage,
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold,
        'scale': scale,
        'blend': blend,
        'fps': fps,
        'zoom_in_duration': zoom_in_duration,
        'zoom_out_duration': zoom_out_duration,
        'transition_duration': transition_duration,
        'morph_time': morph_time,
        'min_zoom': min_zoom,
        'max_zoom': max_zoom,
        'vertical': vertical,
        'face_padding': face_padding,
        'pytorch_style': pytorch_style,
        'blob_mode': blob_mode,
        'num_blobs': num_blobs,
        'blob_frequency': blob_frequency,
        'blob_speed': blob_speed,
        'blob_feather': blob_feather,
        'selected_models': selected_models,
        'selected_tiles': [(t, o) for t, o in selected_tiles] if selected_tiles else None,
    }
    faces_info = [
        {
            'id': f['id'],
            'bbox': f['bbox'],
            'coverage': round(f['coverage'], 2),
            'confidence': round(f.get('confidence', 0), 3),
            'center': list(f['center']),
        }
        for f in valid_faces
    ]
    save_run_log(output_video, args_dict, faces_info, output_video, total_duration)

    return output_video


def main():
    parser = argparse.ArgumentParser(description='Multi-face zoom-in/out video pipeline')
    parser.add_argument('--input_dir', default='/app/input/self_style_samples',
                        help='Input directory containing images')
    parser.add_argument('--output_dir', default='/app/output/morph_faces',
                        help='Output directory')
    parser.add_argument('--image', help='Process single image instead of directory')
    parser.add_argument('--min_coverage', type=float, default=3.0,
                        help='Minimum face coverage percentage (default: 3%%)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence for face detection (0.0-1.0, default: 0.5)')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for filtering overlapping faces (default: 0.3)')
    parser.add_argument('--scale', type=int, default=1440,
                        help='Resolution for styled images')
    parser.add_argument('--blend', type=float, default=0.95,
                        help='Style blend ratio')
    parser.add_argument('--fps', type=int, default=24,
                        help='Video framerate')
    parser.add_argument('--zoom_in', type=float, default=2.0,
                        help='Seconds for intro/outro zoom (default: 2.0)')
    parser.add_argument('--zoom_out', type=float, default=2.0,
                        help='Seconds to morph tiles per face (default: 2.0)')
    parser.add_argument('--transition', type=float, default=2.0,
                        help='Seconds to crossfade between faces (default: 2.0)')
    parser.add_argument('--morph_time', type=float, default=0.0,
                        help='Seconds per image transition (0=auto based on zoom_in+zoom_out, 0.5=half second per image)')
    parser.add_argument('--min_zoom', type=float, default=1.0,
                        help='Starting/ending zoom level (default: 1.0)')
    parser.add_argument('--max_zoom', type=float, default=4.0,
                        help='Zoom level while morphing faces (default: 4.0)')
    # Blob mode options
    parser.add_argument('--blob', action='store_true',
                        help='Enable blob mode with gaussian-weighted blending')
    parser.add_argument('--num_blobs', type=int, default=4,
                        help='Number of blob regions (default: 4)')
    parser.add_argument('--blob_frequency', type=float, default=2.5,
                        help='Blob detail level (default: 2.5)')
    parser.add_argument('--blob_speed', type=float, default=1.0,
                        help='Blob animation speed (default: 1.0)')
    parser.add_argument('--blob_feather', type=float, default=0.3,
                        help='Blob boundary softness (default: 0.3)')
    parser.add_argument('--vertical', action='store_true',
                        help='Vertical video (720x1280)')
    parser.add_argument('--face_padding', type=float, default=0.6,
                        help='Padding around face crop as percentage (default: 0.6 = 60%%)')
    parser.add_argument('--pytorch_style', action='store_true', default=True,
                        help='Pre-style face crops with PyTorch NST models (default: enabled)')
    parser.add_argument('--no_pytorch_style', action='store_true',
                        help='Disable PyTorch NST pre-styling')
    # Model and tile control
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of PyTorch models to use (e.g., "candy,mosaic,udnie"). Use "all" for all models.')
    parser.add_argument('--max_models', type=int, default=None,
                        help='Maximum number of PyTorch models to use (random sample if exceeds)')
    parser.add_argument('--tiles', type=str, default=None,
                        help='Comma-separated list of tile sizes to use (e.g., "128,256,512"). Default: all 7 tiles.')
    parser.add_argument('--list_models', action='store_true',
                        help='List available PyTorch models and exit')
    parser.add_argument('--list_tiles', action='store_true',
                        help='List available tile configurations and exit')
    parser.add_argument('--force', action='store_true',
                        help='Force regenerate existing outputs')

    args = parser.parse_args()

    # Handle --list_models and --list_tiles
    if args.list_models:
        print("Available PyTorch models:")
        print("  Base models: candy, mosaic, rain_princess, udnie")
        print("\n  All models:")
        for i, model in enumerate(PYTORCH_MODELS, 1):
            print(f"    {i:2d}. {model}")
        print(f"\n  Total: {len(PYTORCH_MODELS)} models")
        return 0

    if args.list_tiles:
        print("Available tile configurations:")
        for tile, overlap in TILE_CONFIGS:
            print(f"  tile={tile}, overlap={overlap}")
        print(f"\n  Total: {len(TILE_CONFIGS)} configurations")
        return 0

    # Parse and filter models
    selected_models = None
    if args.models:
        if args.models.lower() == 'all':
            selected_models = list(PYTORCH_MODELS)
        else:
            model_list = [m.strip() for m in args.models.split(',')]
            selected_models = []
            for m in model_list:
                if m in PYTORCH_MODELS:
                    selected_models.append(m)
                else:
                    print(f"[warning] Unknown model '{m}', skipping. Use --list_models to see available models.")
            if not selected_models:
                print("[error] No valid models specified")
                return 1
    elif args.pytorch_style and not args.no_pytorch_style:
        selected_models = list(PYTORCH_MODELS)

    # Apply max_models limit
    if selected_models and args.max_models and args.max_models < len(selected_models):
        print(f"[info] Randomly sampling {args.max_models} models from {len(selected_models)} available")
        selected_models = random.sample(selected_models, args.max_models)

    # Parse and filter tiles
    selected_tiles = None
    if args.tiles:
        tile_list = [int(t.strip()) for t in args.tiles.split(',')]
        selected_tiles = []
        for tile, overlap in TILE_CONFIGS:
            if tile in tile_list:
                selected_tiles.append((tile, overlap))
        if not selected_tiles:
            print(f"[error] No valid tiles specified. Available: {[t for t,o in TILE_CONFIGS]}")
            return 1
    else:
        selected_tiles = list(TILE_CONFIGS)

    # Get list of images to process
    if args.image:
        images = [Path(args.image)]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[error] Input directory not found: {input_dir}")
            return 1

        images = list(input_dir.glob('*.jpg')) + \
                 list(input_dir.glob('*.jpeg')) + \
                 list(input_dir.glob('*.png'))
        images = sorted(images)

    if not images:
        print("[error] No images found")
        return 1

    print(f"[info] Processing {len(images)} image(s)...")

    results = []
    for img_path in images:
        result = process_image(
            img_path,
            args.output_dir,
            min_coverage=args.min_coverage,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            scale=args.scale,
            blend=args.blend,
            fps=args.fps,
            zoom_in_duration=args.zoom_in,
            zoom_out_duration=args.zoom_out,
            transition_duration=args.transition,
            min_zoom=args.min_zoom,
            max_zoom=args.max_zoom,
            vertical=args.vertical,
            force=args.force,
            face_padding=args.face_padding,
            pytorch_style=args.pytorch_style and not args.no_pytorch_style,
            morph_time=args.morph_time,
            blob_mode=args.blob,
            num_blobs=args.num_blobs,
            blob_frequency=args.blob_frequency,
            blob_speed=args.blob_speed,
            blob_feather=args.blob_feather,
            selected_models=selected_models,
            selected_tiles=selected_tiles
        )
        if result:
            results.append(result)

    print(f"\n[summary] Generated {len(results)} video(s)")
    for r in results:
        print(f"  - {r}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
