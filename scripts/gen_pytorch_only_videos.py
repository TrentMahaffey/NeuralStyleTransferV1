#!/usr/bin/env python3
"""
Generate morph videos using only PyTorch+Magenta styled frames.
Excludes plain Magenta-only frames. Uses face zoom logic from morph_faces.
"""
import sys
from pathlib import Path
import subprocess
import argparse
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from morph_v2 import PYTORCH_MODELS, _smootherstep
from morph_faces import (
    create_face_crossfade,
    detect_faces,
    calculate_safe_zoom,
    apply_zoom_crop
)
import cv2
import numpy as np
import math
import torch


def create_spiral_mask(H, W, progress, tightness=3.0, rotation_offset=0.0):
    """Create a spiral wipe mask for transitions.

    Args:
        H, W: Height and width
        progress: Transition progress 0.0 to 1.0
        tightness: How tight the spiral is (higher = more rotations)
        rotation_offset: Starting rotation in radians

    Returns:
        Mask as numpy array (H, W) with values 0-1
    """
    cx, cy = W / 2, H / 2

    y_coords = np.arange(H, dtype=np.float32)[:, None] - cy
    x_coords = np.arange(W, dtype=np.float32)[None, :] - cx

    r = np.sqrt(x_coords**2 + y_coords**2)
    theta = np.arctan2(y_coords, x_coords) + np.pi + rotation_offset

    # Spiral formula
    spiral = (theta + r / max(H, W) * tightness * 2 * np.pi) % (2 * np.pi)
    spiral = spiral / (2 * np.pi)  # Normalize to [0, 1]

    # Create soft mask based on progress
    mask = np.clip((progress - spiral) * 10 + 0.5, 0, 1).astype(np.float32)

    return mask


def create_blob_mask(H, W, progress, frequency=3.0, seed=42, frame_idx=0, speed=1.0):
    """Create an organic blob wipe mask with animated boundaries.

    Args:
        H, W: Height and width
        progress: Transition progress 0.0 to 1.0
        frequency: Noise frequency (higher = more detail)
        seed: Random seed
        frame_idx: For animation
        speed: Animation speed

    Returns:
        Mask as numpy array (H, W) with values 0-1
    """
    time_offset = frame_idx * speed * 0.05

    # Create coordinate grids
    y_norm = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, W, dtype=np.float32)[None, :]

    # Generate simplex-like noise using multiple sine waves
    np.random.seed(seed)
    noise = np.zeros((H, W), dtype=np.float32)

    for octave in range(3):
        freq = frequency * (2 ** octave)
        amp = 1.0 / (2 ** octave)
        phase_x = np.random.random() * 2 * np.pi
        phase_y = np.random.random() * 2 * np.pi

        noise += amp * np.sin(y_norm * freq * np.pi + phase_y + time_offset)
        noise += amp * np.sin(x_norm * freq * np.pi + phase_x + time_offset * 1.3)

    # Normalize noise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    # Combine with linear wipe for directional progress
    linear_wipe = x_norm  # Left to right
    combined = linear_wipe * 0.6 + noise * 0.4

    # Create soft mask based on progress
    mask = np.clip((progress - combined) * 8 + 0.5, 0, 1).astype(np.float32)

    return mask


def create_radial_mask(H, W, progress, center=None):
    """Create a radial wipe mask from center.

    Args:
        H, W: Height and width
        progress: Transition progress 0.0 to 1.0
        center: (x, y) normalized center (default: 0.5, 0.5)

    Returns:
        Mask as numpy array (H, W) with values 0-1
    """
    if center is None:
        center = (0.5, 0.5)

    cx, cy = center[0] * W, center[1] * H

    y_coords = np.arange(H, dtype=np.float32)[:, None] - cy
    x_coords = np.arange(W, dtype=np.float32)[None, :] - cx

    r = np.sqrt(x_coords**2 + y_coords**2)
    r = r / r.max()  # Normalize to [0, 1]

    # Create soft mask based on progress (expand from center)
    mask = np.clip((progress - r) * 8 + 0.5, 0, 1).astype(np.float32)

    return mask


def create_animated_blob_mask(H, W, frame_idx, frequency=3.0, speed=1.0, seed=42):
    """Create an animated blob mask that morphs over time (not a wipe).

    Args:
        H, W: Height and width
        frame_idx: Current frame number for animation
        frequency: Noise frequency (higher = more detail)
        speed: Animation speed
        seed: Random seed for consistency

    Returns:
        Mask as numpy array (H, W) with values 0-1
    """
    time_offset = frame_idx * speed * 0.03

    # Create coordinate grids
    y_norm = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, W, dtype=np.float32)[None, :]

    # Generate animated noise using multiple sine waves
    np.random.seed(seed)
    noise = np.zeros((H, W), dtype=np.float32)

    for octave in range(4):
        freq = frequency * (2 ** octave)
        amp = 1.0 / (1.5 ** octave)
        phase_x = np.random.random() * 2 * np.pi
        phase_y = np.random.random() * 2 * np.pi
        phase_t = np.random.random() * 2 * np.pi

        # Animate the phases over time
        noise += amp * np.sin(y_norm * freq * np.pi + phase_y + time_offset * (1 + octave * 0.3))
        noise += amp * np.sin(x_norm * freq * np.pi + phase_x + time_offset * (1.2 + octave * 0.2))
        # Add diagonal movement
        noise += amp * 0.5 * np.sin((x_norm + y_norm) * freq * np.pi + phase_t + time_offset * 1.5)

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    # Apply sigmoid for sharper but still soft edges
    noise = 1 / (1 + np.exp(-8 * (noise - 0.5)))

    return noise.astype(np.float32)


def create_multi_region_blob_masks(H, W, num_regions, frame_idx, frequency=2.5, speed=1.0, seed=42):
    """Create animated blob masks for multiple regions (like voronoi but organic).

    Args:
        H, W: Height and width
        num_regions: Number of regions (e.g., 5 for 5 model families)
        frame_idx: Current frame number for animation
        frequency: Noise frequency (higher = more detail)
        speed: Animation speed
        seed: Random seed for consistency

    Returns:
        List of masks, one per region, each (H, W) with values 0-1
        Each pixel's masks sum to 1.0 (soft assignment to regions)
    """
    time_offset = frame_idx * speed * 0.02

    # Create coordinate grids
    y_norm = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, W, dtype=np.float32)[None, :]

    # Generate a noise field for each region with different seeds/phases
    region_values = []
    np.random.seed(seed)

    for region_idx in range(num_regions):
        noise = np.zeros((H, W), dtype=np.float32)

        # Use different random phases for each region
        region_seed = seed + region_idx * 1000
        np.random.seed(region_seed)

        # Create region center that slowly moves
        center_x = 0.2 + 0.6 * (region_idx % 3) / 2  # Spread horizontally
        center_y = 0.3 + 0.4 * (region_idx // 3)      # Stack vertically if > 3

        # Add time-varying offset to center
        center_x += 0.1 * np.sin(time_offset * 0.5 + region_idx * 1.2)
        center_y += 0.1 * np.cos(time_offset * 0.4 + region_idx * 0.9)

        # Distance from animated center (creates blob shape)
        dist = np.sqrt((x_norm - center_x)**2 + (y_norm - center_y)**2)

        # Add organic noise to the distance field
        for octave in range(3):
            freq = frequency * (2 ** octave)
            amp = 0.15 / (1.5 ** octave)
            phase_x = np.random.random() * 2 * np.pi
            phase_y = np.random.random() * 2 * np.pi

            noise += amp * np.sin(y_norm * freq * np.pi + phase_y + time_offset * (1 + octave * 0.2))
            noise += amp * np.sin(x_norm * freq * np.pi + phase_x + time_offset * (1.1 + octave * 0.15))

        # Combine distance with noise - lower value = more likely this region
        region_field = dist + noise
        region_values.append(region_field)

    # Convert to soft masks using softmax-like normalization
    # Each pixel goes to the region with lowest value (closest center + noise)
    region_values = np.stack(region_values, axis=0)  # (num_regions, H, W)

    # Use negative values for softmax (lower = higher weight)
    temperature = 0.08  # Lower = sharper boundaries, higher = softer
    exp_values = np.exp(-region_values / temperature)
    sum_exp = np.sum(exp_values, axis=0, keepdims=True) + 1e-6

    masks = exp_values / sum_exp  # (num_regions, H, W)

    return [masks[i] for i in range(num_regions)]


def create_multi_model_blob_video(
    model_frame_sets,
    model_names,
    original_image,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=4.0,
    fps=24,
    seconds_per_frame=2.0,
    zoom_center=None,
    blob_frequency=2.5,
    blob_speed=1.0
):
    """Create a video with multiple model families shown simultaneously in blob regions.

    Each model family morphs through its weight variations while occupying an
    animated blob region of the screen.

    Args:
        model_frame_sets: Dict of model_name -> list of (path, blended_image) tuples
        model_names: List of model names in order
        original_image: Path to original unstyled image
        target_size: (width, height) of output
        min_zoom: Ending zoom level
        max_zoom: Starting zoom level
        fps: Frames per second
        seconds_per_frame: Seconds per styled frame (determines duration)
        zoom_center: (x, y) normalized center for zoom
        blob_frequency: Blob detail level
        blob_speed: Blob animation speed

    Returns:
        List of frames
    """
    num_models = len(model_names)
    if num_models == 0:
        return []

    # Find the max number of frames across all models
    max_frames = max(len(model_frame_sets[name]) for name in model_names)
    if max_frames == 0:
        return []

    # Calculate duration based on max frames
    duration = max_frames * seconds_per_frame

    # Load original image
    if isinstance(original_image, np.ndarray):
        orig_img = original_image.copy()
    else:
        orig_img = cv2.imread(str(original_image))

    safe_min_zoom = calculate_safe_zoom(zoom_center, min_zoom)
    frames = []
    num_output_frames = int(duration * fps)

    # 85% morph through styles, 15% blend to original
    morph_end = 0.85

    H, W = target_size[1], target_size[0]

    print(f"      Creating {num_output_frames} frames ({duration:.1f}s) with {num_models} model regions")

    for frame_idx in range(num_output_frames):
        t = frame_idx / max(1, num_output_frames - 1)
        t_eased = _smootherstep(t)

        # Zoom from max to min
        zoom = max_zoom - t_eased * (max_zoom - safe_min_zoom)

        # Create animated blob masks for all regions
        blob_masks = create_multi_region_blob_masks(
            H, W, num_models, frame_idx, blob_frequency, blob_speed
        )

        # Initialize output frame
        output_frame = np.zeros((H, W, 3), dtype=np.float32)

        if t < morph_end:
            morph_t = t / morph_end

            for model_idx, model_name in enumerate(model_names):
                model_frames = model_frame_sets[model_name]
                num_model_frames = len(model_frames)

                if num_model_frames == 0:
                    continue

                # Calculate position in this model's frame sequence
                pos = morph_t * (num_model_frames - 1)
                idx1 = int(pos)
                idx2 = min(idx1 + 1, num_model_frames - 1)
                blend_alpha = pos - idx1

                # Get frames (could be paths or pre-loaded images)
                frame1_data = model_frames[idx1]
                frame2_data = model_frames[idx2]

                # Load or use preloaded images
                if isinstance(frame1_data, tuple):
                    img1 = frame1_data[1] if frame1_data[1] is not None else cv2.imread(str(frame1_data[0]))
                elif isinstance(frame1_data, np.ndarray):
                    img1 = frame1_data
                else:
                    img1 = cv2.imread(str(frame1_data))

                if isinstance(frame2_data, tuple):
                    img2 = frame2_data[1] if frame2_data[1] is not None else cv2.imread(str(frame2_data[0]))
                elif isinstance(frame2_data, np.ndarray):
                    img2 = frame2_data
                else:
                    img2 = cv2.imread(str(frame2_data))

                if img1 is None or img2 is None:
                    continue

                # Resize if needed
                if img1.shape != img2.shape:
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

                # Blend between consecutive frames
                styled = cv2.addWeighted(img1, 1 - blend_alpha, img2, blend_alpha, 0)

                # Apply zoom crop
                cropped = apply_zoom_crop(styled, zoom, zoom_center, target_size)

                # Add to output using this model's blob mask
                mask_3ch = blob_masks[model_idx][:, :, np.newaxis]
                output_frame += cropped.astype(np.float32) * mask_3ch

        else:
            # Blend to original
            blend_t = (t - morph_end) / (1.0 - morph_end)
            blend_t_eased = _smootherstep(blend_t)

            orig_cropped = apply_zoom_crop(orig_img, zoom, zoom_center, target_size)

            for model_idx, model_name in enumerate(model_names):
                model_frames = model_frame_sets[model_name]
                if not model_frames:
                    continue

                # Use last frame from each model
                last_frame_data = model_frames[-1]
                if isinstance(last_frame_data, tuple):
                    last_img = last_frame_data[1] if last_frame_data[1] is not None else cv2.imread(str(last_frame_data[0]))
                elif isinstance(last_frame_data, np.ndarray):
                    last_img = last_frame_data
                else:
                    last_img = cv2.imread(str(last_frame_data))

                if last_img is None:
                    continue

                styled_cropped = apply_zoom_crop(last_img, zoom, zoom_center, target_size)

                # Blend styled to original
                blended = cv2.addWeighted(
                    styled_cropped, 1 - blend_t_eased,
                    orig_cropped, blend_t_eased, 0
                )

                # Add to output using this model's blob mask
                mask_3ch = blob_masks[model_idx][:, :, np.newaxis]
                output_frame += blended.astype(np.float32) * mask_3ch

        frames.append(output_frame.astype(np.uint8))

    return frames


def blend_with_effect(img1, img2, progress, effect='linear', frame_idx=0, **kwargs):
    """Blend two images using various transition effects.

    Args:
        img1: Source image (numpy array)
        img2: Target image (numpy array)
        progress: Transition progress 0.0 to 1.0
        effect: 'linear', 'spiral', 'blob', 'radial'
        frame_idx: Current frame for animated effects
        **kwargs: Additional effect parameters

    Returns:
        Blended image
    """
    H, W = img1.shape[:2]

    if effect == 'linear':
        # Simple crossfade
        return cv2.addWeighted(img1, 1 - progress, img2, progress, 0)

    elif effect == 'spiral':
        tightness = kwargs.get('tightness', 3.0)
        rotation = kwargs.get('rotation_offset', frame_idx * 0.02)
        mask = create_spiral_mask(H, W, progress, tightness, rotation)

    elif effect == 'blob':
        frequency = kwargs.get('frequency', 3.0)
        seed = kwargs.get('seed', 42)
        speed = kwargs.get('speed', 1.0)
        mask = create_blob_mask(H, W, progress, frequency, seed, frame_idx, speed)

    elif effect == 'radial':
        center = kwargs.get('center', None)
        mask = create_radial_mask(H, W, progress, center)

    else:
        # Default to linear
        return cv2.addWeighted(img1, 1 - progress, img2, progress, 0)

    # Apply mask for non-linear effects
    mask_3ch = mask[:, :, np.newaxis]
    blended = img1.astype(np.float32) * (1 - mask_3ch) + img2.astype(np.float32) * mask_3ch

    return blended.astype(np.uint8)


def create_dual_morph_blob(
    styled_images,
    original_image,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=4.0,
    fps=24,
    duration=5.0,
    zoom_center=None,
    preloaded_images=None,
    blob_frequency=3.0,
    blob_speed=1.0
):
    """Create a video with two morph sequences separated by animated blob regions.

    Splits frames into two groups, each morphing independently while an animated
    blob mask divides the screen showing both simultaneously.

    Args:
        styled_images: List of styled image paths
        original_image: Path to original unstyled image
        target_size: (width, height) of output
        min_zoom: Ending zoom level
        max_zoom: Starting zoom level
        fps: Frames per second
        duration: Duration in seconds
        zoom_center: (x, y) normalized center for zoom
        preloaded_images: Optional list of pre-loaded numpy arrays
        blob_frequency: Blob detail level (higher = more intricate)
        blob_speed: Blob animation speed

    Returns:
        List of frames
    """
    # Use preloaded images if provided
    if preloaded_images and len(preloaded_images) > 0:
        images = preloaded_images
    else:
        images = []
        for img_path in styled_images:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)

    if len(images) < 2:
        return []

    # Load original image
    if isinstance(original_image, np.ndarray):
        orig_img = original_image.copy()
    else:
        orig_img = cv2.imread(str(original_image))

    # Split into two groups - odd and even indices for variety
    group_a = images[0::2]  # Even indices (0, 2, 4, ...)
    group_b = images[1::2]  # Odd indices (1, 3, 5, ...)

    # Ensure both groups have frames
    if not group_a:
        group_a = images[:len(images)//2]
    if not group_b:
        group_b = images[len(images)//2:]

    num_a = len(group_a)
    num_b = len(group_b)

    safe_min_zoom = calculate_safe_zoom(zoom_center, min_zoom)
    frames = []
    num_frames = int(duration * fps)

    # 80% morph, 20% blend to original
    morph_end = 0.8

    for frame_idx in range(num_frames):
        t = frame_idx / max(1, num_frames - 1)
        t_eased = _smootherstep(t)

        # Zoom from max to min
        zoom = max_zoom - t_eased * (max_zoom - safe_min_zoom)

        # Create animated blob mask for this frame
        H, W = target_size[1], target_size[0]  # Note: target_size is (width, height)
        blob_mask = create_animated_blob_mask(H, W, frame_idx, blob_frequency, blob_speed)
        blob_mask_3ch = blob_mask[:, :, np.newaxis]

        if t < morph_end:
            morph_t = t / morph_end

            # Group A morph position (linear, forward)
            pos_a = morph_t * (num_a - 1)
            idx_a1 = int(pos_a)
            idx_a2 = min(idx_a1 + 1, num_a - 1)
            blend_a = pos_a - idx_a1

            # Group B morph position (reverse direction for variety, no wrapping)
            pos_b = (1.0 - morph_t) * (num_b - 1)
            idx_b1 = int(pos_b)
            idx_b2 = min(idx_b1 + 1, num_b - 1)
            blend_b = pos_b - idx_b1

            # Get frames for group A
            img_a1 = group_a[idx_a1]
            img_a2 = group_a[idx_a2]
            if img_a1.shape != img_a2.shape:
                img_a2 = cv2.resize(img_a2, (img_a1.shape[1], img_a1.shape[0]))
            styled_a = cv2.addWeighted(img_a1, 1 - blend_a, img_a2, blend_a, 0)

            # Get frames for group B
            img_b1 = group_b[idx_b1]
            img_b2 = group_b[idx_b2]
            if img_b1.shape != img_b2.shape:
                img_b2 = cv2.resize(img_b2, (img_b1.shape[1], img_b1.shape[0]))
            styled_b = cv2.addWeighted(img_b1, 1 - blend_b, img_b2, blend_b, 0)

            # Apply zoom crop to both
            frame_a = apply_zoom_crop(styled_a, zoom, zoom_center, target_size)
            frame_b = apply_zoom_crop(styled_b, zoom, zoom_center, target_size)

            # Blend using blob mask
            frame = (frame_a.astype(np.float32) * (1 - blob_mask_3ch) +
                     frame_b.astype(np.float32) * blob_mask_3ch).astype(np.uint8)
        else:
            # Blend to original
            blend_t = (t - morph_end) / (1.0 - morph_end)
            blend_t_eased = _smootherstep(blend_t)

            # Use last frames from each group
            styled_a = apply_zoom_crop(group_a[-1], zoom, zoom_center, target_size)
            styled_b = apply_zoom_crop(group_b[-1], zoom, zoom_center, target_size)
            orig_frame = apply_zoom_crop(orig_img, zoom, zoom_center, target_size)

            # Blend both styled frames to original
            frame_a = cv2.addWeighted(styled_a, 1 - blend_t_eased, orig_frame, blend_t_eased, 0)
            frame_b = cv2.addWeighted(styled_b, 1 - blend_t_eased, orig_frame, blend_t_eased, 0)

            # Still apply blob mask during blend out
            frame = (frame_a.astype(np.float32) * (1 - blob_mask_3ch) +
                     frame_b.astype(np.float32) * blob_mask_3ch).astype(np.uint8)

        frames.append(frame)

    return frames


def create_face_zoom_out_linear(
    styled_images,
    original_image,
    target_size=(720, 1280),
    min_zoom=1.0,
    max_zoom=4.0,
    fps=24,
    duration=5.0,
    zoom_center=None,
    linear_frames=True,
    preloaded_images=None,
    transition_effect='linear'
):
    """Zoom out from max to min while morphing tiles, with linear frame progression.

    Args:
        styled_images: List of styled image paths (ignored if preloaded_images provided)
        original_image: Path to original unstyled image
        target_size: (width, height) of output
        min_zoom: Ending zoom level (zoomed out)
        max_zoom: Starting zoom level (zoomed in)
        fps: Frames per second
        duration: Duration in seconds
        zoom_center: (x, y) normalized center for zoom
        linear_frames: If True, frames progress linearly; if False, use eased progression
        preloaded_images: Optional list of pre-loaded numpy arrays (already blended)
        transition_effect: 'linear', 'spiral', 'blob', 'radial'

    Returns:
        List of frames
    """
    # Use preloaded images if provided, otherwise load from paths
    if preloaded_images and len(preloaded_images) > 0:
        images = preloaded_images
    else:
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

        # Zoom OUT: from max_zoom to safe_min_zoom (still eased for smooth camera)
        zoom = max_zoom - t_eased * (max_zoom - safe_min_zoom)

        if t < morph_end:
            # Phase 1: Morph through tiles
            morph_t = t / morph_end

            # LINEAR frame progression (not eased) - each frame gets equal time
            if linear_frames:
                img_pos = (1 - morph_t) * (num_images - 1)
            else:
                morph_t_eased = _smootherstep(morph_t)
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

            # Use transition effect for blending
            styled = blend_with_effect(img1, img2, blend_alpha, transition_effect, frame_idx,
                                       center=zoom_center)
            frame = apply_zoom_crop(styled, zoom, zoom_center, target_size)
        else:
            # Phase 2: Blend from largest tile to original
            blend_t = (t - morph_end) / (1.0 - morph_end)
            blend_t_eased = _smootherstep(blend_t)

            styled_frame = apply_zoom_crop(images[0], zoom, zoom_center, target_size)
            orig_frame = apply_zoom_crop(orig_img, zoom, zoom_center, target_size)

            # Use transition effect for final blend too
            frame = blend_with_effect(styled_frame, orig_frame, blend_t_eased, transition_effect,
                                      frame_idx, center=zoom_center)

        frames.append(frame)

    return frames

TILE_CONFIGS = [
    (128, 16), (160, 20), (192, 24), (224, 28), (256, 32), (384, 48), (512, 64)
]


def get_image_brightness(image_path):
    """Calculate average brightness of an image (0-255 scale)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 128  # Default to middle brightness if can't read
    # Convert to grayscale and get mean
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def get_image_color_intensity(image_path, channel='red'):
    """Calculate average intensity of a specific color channel.

    Args:
        image_path: Path to image
        channel: 'red', 'green', or 'blue'

    Returns:
        Average intensity of the channel (0-255)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 128

    # OpenCV uses BGR order
    channel_idx = {'blue': 0, 'green': 1, 'red': 2}.get(channel, 2)
    return np.mean(img[:, :, channel_idx])


def blend_with_magenta(pytorch_frame_path, face_dir, face_id, blend_ratio=0.5, original_img=None, blend_original=0.0):
    """Blend a PyTorch+Magenta frame with its corresponding plain Magenta frame and optionally original.

    Args:
        pytorch_frame_path: Path to PyTorch+Magenta styled frame
        face_dir: Directory containing face frames
        face_id: Face ID number
        blend_ratio: Blend ratio for NST vs Magenta (0.5 = 50/50, ignored if blend_original > 0)
        original_img: Original image (numpy array) for 3-way blend
        blend_original: If > 0, do 3-way blend (e.g., 0.333 = 1/3 each)

    Returns:
        Blended image as numpy array, or None if plain Magenta frame not found
    """
    # Extract tile info from filename
    # Format: face1_candy_tile128_overlap16.jpg
    name = pytorch_frame_path.name

    # Find tile and overlap from the name
    for tile, overlap in TILE_CONFIGS:
        tile_str = f"tile{tile}_overlap{overlap}"
        if tile_str in name:
            # Construct plain Magenta filename
            plain_filename = f"face{face_id}_{tile_str}.jpg"
            plain_path = face_dir / plain_filename

            if plain_path.exists():
                # Load both images
                pytorch_img = cv2.imread(str(pytorch_frame_path))
                plain_img = cv2.imread(str(plain_path))

                if pytorch_img is not None and plain_img is not None:
                    # Resize if needed
                    if pytorch_img.shape != plain_img.shape:
                        plain_img = cv2.resize(plain_img, (pytorch_img.shape[1], pytorch_img.shape[0]))

                    if blend_original > 0 and original_img is not None:
                        # 3-way blend: NST, Magenta, Original
                        orig_resized = cv2.resize(original_img, (pytorch_img.shape[1], pytorch_img.shape[0]))
                        # Each gets equal weight (1/3 each when blend_original=0.333)
                        w_orig = blend_original
                        w_nst = (1 - w_orig) / 2
                        w_mag = (1 - w_orig) / 2
                        blended = (pytorch_img.astype(np.float32) * w_nst +
                                   plain_img.astype(np.float32) * w_mag +
                                   orig_resized.astype(np.float32) * w_orig).astype(np.uint8)
                    else:
                        # 2-way blend: NST vs Magenta
                        blended = cv2.addWeighted(pytorch_img, blend_ratio, plain_img, 1 - blend_ratio, 0)
                    return blended
            break

    return None


def collect_pytorch_styled_frames(styled_dir, face_id, sort_by='tile', reverse=False, blend_magenta=0.0,
                                   original_img=None, blend_original=0.0, model_filter=None):
    """Collect only PyTorch+Magenta styled frames for a face.

    Args:
        styled_dir: Directory containing styled images
        face_id: Face ID number
        sort_by: Sorting method - 'tile', 'brightness', 'model', 'red', 'green', 'blue'
        reverse: Reverse sort order
        blend_magenta: If > 0, blend with plain Magenta frames at this ratio (0.5 = 50% each)
        original_img: Original image for 3-way blend
        blend_original: If > 0, do 3-way blend with original (0.333 = 1/3 each)
        model_filter: If set, only include models starting with this prefix (e.g., 'candy', 'mosaic')

    Returns:
        List of frame paths (or blended images if blend_magenta > 0) in sorted order
    """
    face_dir = styled_dir / f"face{face_id}"
    if not face_dir.exists():
        return [], []

    # Filter models if specified
    models_to_use = PYTORCH_MODELS
    if model_filter:
        models_to_use = [m for m in PYTORCH_MODELS if m.startswith(model_filter)]
        print(f"      Filtering to {len(models_to_use)} models matching '{model_filter}*'")

    styled_paths = []
    # Collect all PyTorch+Magenta frames
    for tile, overlap in TILE_CONFIGS:
        for model_name in models_to_use:
            filename = f"face{face_id}_{model_name}_tile{tile}_overlap{overlap}.jpg"
            path = face_dir / filename
            if path.exists():
                styled_paths.append(path)

    if not styled_paths:
        return [], []

    if sort_by == 'brightness':
        # Sort by average brightness (dark to light by default)
        print(f"      Calculating brightness for {len(styled_paths)} frames...")
        values = [(p, get_image_brightness(p)) for p in styled_paths]
        values.sort(key=lambda x: x[1], reverse=reverse)
        styled_paths = [p for p, _ in values]
        min_v = min(v for _, v in values)
        max_v = max(v for _, v in values)
        print(f"      Brightness range: {min_v:.1f} - {max_v:.1f}")
    elif sort_by in ('red', 'green', 'blue'):
        # Sort by color channel intensity
        print(f"      Calculating {sort_by} intensity for {len(styled_paths)} frames...")
        values = [(p, get_image_color_intensity(p, sort_by)) for p in styled_paths]
        values.sort(key=lambda x: x[1], reverse=reverse)
        styled_paths = [p for p, _ in values]
        min_v = min(v for _, v in values)
        max_v = max(v for _, v in values)
        print(f"      {sort_by.capitalize()} range: {min_v:.1f} - {max_v:.1f}")
    elif sort_by == 'tile':
        # Sort by tile size (large to small for zoom effect)
        def tile_sort_key(path):
            name = path.name
            for i, (tile, overlap) in enumerate(reversed(TILE_CONFIGS)):
                if f"tile{tile}_overlap{overlap}" in name:
                    return i
            return 999
        styled_paths.sort(key=tile_sort_key)
    elif sort_by == 'model':
        # Sort by model name
        def model_sort_key(path):
            name = path.name
            for i, model in enumerate(PYTORCH_MODELS):
                if f"_{model}_tile" in name:
                    return i
            return 999
        styled_paths.sort(key=model_sort_key, reverse=reverse)

    # Optionally blend with plain Magenta frames (and original)
    blended_images = []
    if blend_magenta > 0 or blend_original > 0:
        if blend_original > 0:
            print(f"      3-way blend: {(1-blend_original)/2*100:.0f}% NST / {(1-blend_original)/2*100:.0f}% Magenta / {blend_original*100:.0f}% Original...")
        else:
            print(f"      Blending with plain Magenta at {blend_magenta*100:.0f}% NST / {(1-blend_magenta)*100:.0f}% plain...")
        for path in styled_paths:
            blended = blend_with_magenta(path, face_dir, face_id, blend_magenta, original_img, blend_original)
            if blended is not None:
                blended_images.append(blended)
            else:
                # Fallback to original if no plain Magenta found
                blended_images.append(cv2.imread(str(path)))
        print(f"      Blended {len(blended_images)} frames")

    return styled_paths, blended_images


def get_face_centers_from_image(image_path, styled_dir):
    """Get face centers either from cached detection or by re-detecting."""
    # Try to find face directories to infer face IDs
    face_dirs = sorted([d for d in styled_dir.iterdir() if d.is_dir() and d.name.startswith('face')])

    if not face_dirs:
        return {}

    # Re-detect faces to get centers
    orig_img = cv2.imread(str(image_path))
    if orig_img is None:
        return {}

    orig_h, orig_w = orig_img.shape[:2]
    faces = detect_faces(str(image_path), confidence_threshold=0.2)

    # Build face center map
    face_centers = {}
    for face in faces:
        face_id = face['id']
        face_cx, face_cy = face['center']
        face_centers[face_id] = (face_cx / orig_w, face_cy / orig_h)

    return face_centers


def process_image(image_dir, fps=24, zoom_duration=4.0, transition_duration=1.0,
                  min_zoom=1.0, max_zoom=4.0, vertical=False, sort_by='tile', sort_reverse=False,
                  seconds_per_frame=None, blend_magenta=0.0, transition_effect='linear', blend_original=0.0,
                  model_filter=None):
    """Generate PyTorch-only video with face zoom for a single image directory.

    Args:
        image_dir: Directory containing styled/ subdirectory
        fps: Frames per second
        zoom_duration: Seconds for zoom-out per face (ignored if seconds_per_frame is set)
        transition_duration: Seconds for crossfade between faces
        min_zoom: Ending zoom level (zoomed out)
        max_zoom: Starting zoom level (zoomed in on face)
        vertical: True for vertical video (720x1280)
        sort_by: Frame sorting method - 'tile', 'brightness', 'model', 'red', 'green', 'blue'
        sort_reverse: Reverse sort order
        seconds_per_frame: If set, calculates zoom_duration = num_frames * seconds_per_frame
        blend_magenta: Blend ratio with plain Magenta frames (0.5 = 50% NST, 50% plain)
        transition_effect: Transition effect - 'linear', 'spiral', 'blob', 'radial'
        blend_original: Blend ratio for original image in 3-way blend (0.333 = 1/3 each)
        model_filter: Filter to only models starting with this prefix (e.g., 'candy', 'mosaic')
    """
    image_dir = Path(image_dir)
    styled_dir = image_dir / "styled"

    if not styled_dir.exists():
        print(f"No styled directory found: {styled_dir}")
        return None

    print(f"\nProcessing: {image_dir.name}")

    # Find original image - check multiple locations
    orig_candidates = []
    image_name = image_dir.name

    # 1. Check self_style_samples directory (most common for morph_faces)
    self_style_dir = Path("/app/input/self_style_samples")
    if self_style_dir.exists():
        orig_candidates.extend(list(self_style_dir.glob(f"{image_name}.*")))
        orig_candidates.extend(list(self_style_dir.glob(f"{image_name}*")))

    # 2. Check main input directory
    if not orig_candidates:
        parent_input = Path("/app/input")
        orig_candidates.extend(list(parent_input.glob(f"{image_name}.*")))

    # 3. Check the image_dir itself (filter out generated files)
    if not orig_candidates:
        dir_candidates = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
        orig_candidates = [p for p in dir_candidates if 'styled' not in str(p) and 'work' not in str(p)
                          and '_pytorch' not in p.name and '_faces_zoom' not in p.name]

    # Filter to only image files
    orig_candidates = [p for p in orig_candidates if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    if not orig_candidates:
        print(f"  Could not find original image")
        return None

    original_image = orig_candidates[0]
    print(f"  Original: {original_image.name}")

    # Find face directories
    face_dirs = sorted([d for d in styled_dir.iterdir() if d.is_dir() and d.name.startswith('face')])
    if not face_dirs:
        print(f"  No face directories found")
        return None

    # Video settings
    target_size = (720, 1280) if vertical else (1280, 720)

    # Get face centers by re-detecting
    face_centers = get_face_centers_from_image(original_image, styled_dir)
    print(f"  Detected {len(face_centers)} face center(s)")

    # Load original image for potential 3-way blend
    orig_img = cv2.imread(str(original_image))

    # Collect PyTorch-styled frames per face
    sort_desc = f"by {sort_by}" + (" (reversed)" if sort_reverse else "")
    if blend_original > 0:
        sort_desc += f" + 3-way blend (1/3 each)"
    elif blend_magenta > 0:
        sort_desc += f" + {blend_magenta*100:.0f}% NST / {(1-blend_magenta)*100:.0f}% plain Magenta blend"
    print(f"  Sorting frames {sort_desc}")

    face_styled_images = {}  # face_id -> (paths, blended_images)
    for face_dir in face_dirs:
        face_id = int(face_dir.name.replace('face', ''))
        styled_paths, blended_images = collect_pytorch_styled_frames(
            styled_dir, face_id, sort_by=sort_by, reverse=sort_reverse, blend_magenta=blend_magenta,
            original_img=orig_img, blend_original=blend_original, model_filter=model_filter
        )
        if styled_paths:
            print(f"  Face {face_id}: {len(styled_paths)} PyTorch+Magenta frames")
            face_styled_images[face_id] = (styled_paths, blended_images)

    if not face_styled_images:
        print(f"  No PyTorch-styled frames found")
        return None

    # Create zoom video
    print(f"  Creating face zoom video...")
    all_frames = []
    face_ids = sorted(face_styled_images.keys())

    for i, face_id in enumerate(face_ids):
        styled_paths, blended_images = face_styled_images[face_id]
        if not styled_paths:
            continue

        # Get zoom center for this face
        zoom_center = face_centers.get(face_id)
        if zoom_center is None:
            # Fallback to center if face not found in re-detection
            zoom_center = (0.5, 0.5)
            print(f"    Face {face_id}: using center fallback")

        is_last = (i == len(face_ids) - 1)

        # Calculate zoom duration for this face
        if seconds_per_frame is not None:
            face_zoom_duration = len(styled_paths) * seconds_per_frame
        else:
            face_zoom_duration = zoom_duration

        # ZOOM OUT: Start at max zoom with detailed tiles, zoom out while morphing
        effect_desc = f" [{transition_effect}]" if transition_effect != 'linear' else ""
        print(f"    Face {face_id}: zoom {max_zoom}x -> {min_zoom}x ({face_zoom_duration:.1f}s, {len(styled_paths)} frames){effect_desc}")

        if transition_effect == 'dual_blob':
            # Use dual morph with blob separation
            zoom_frames = create_dual_morph_blob(
                styled_paths,
                original_image,
                target_size=target_size,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                fps=fps,
                duration=face_zoom_duration,
                zoom_center=zoom_center,
                preloaded_images=blended_images if blended_images else None,
                blob_frequency=3.0,
                blob_speed=1.0
            )
        else:
            zoom_frames = create_face_zoom_out_linear(
                styled_paths,
                original_image,
                target_size=target_size,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
                fps=fps,
                duration=face_zoom_duration,
                zoom_center=zoom_center,
                linear_frames=True,  # Each styled frame gets equal time
                preloaded_images=blended_images if blended_images else None,
                transition_effect=transition_effect
            )
        all_frames.extend(zoom_frames)

        # Crossfade to next face (if not last)
        if not is_last:
            next_face_id = face_ids[i + 1]
            next_data = face_styled_images.get(next_face_id, ([], []))
            next_styled_paths = next_data[0] if next_data else []
            next_zoom_center = face_centers.get(next_face_id, (0.5, 0.5))

            if next_styled_paths:
                print(f"    Transition: face {face_id} -> face {next_face_id} ({transition_duration}s)")
                crossfade_frames = create_face_crossfade(
                    original_image,
                    next_styled_paths,
                    target_size=target_size,
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                    fps=fps,
                    duration=transition_duration,
                    center_from=zoom_center,
                    center_to=next_zoom_center
                )
                all_frames.extend(crossfade_frames)

    if not all_frames:
        print(f"  ERROR: No frames generated")
        return None

    # Write video - include sort method, blend, and transition effect in filename
    sort_suffix = f"_{sort_by}" if sort_by != 'tile' else ""
    if blend_original > 0:
        blend_suffix = "_3way"
    elif blend_magenta > 0:
        blend_suffix = f"_blend{int(blend_magenta*100)}"
    else:
        blend_suffix = ""
    effect_suffix = f"_{transition_effect}" if transition_effect != 'linear' else ""
    model_suffix = f"_{model_filter}" if model_filter else ""
    output_path = image_dir / f"{image_dir.name}_pytorch_zoom{sort_suffix}{blend_suffix}{effect_suffix}{model_suffix}.mp4"
    temp_path = image_dir / "temp_pytorch_zoom.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tw, th = target_size
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (tw, th))

    for frame in all_frames:
        writer.write(frame)
    writer.release()

    # Re-encode with ffmpeg for better compatibility
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', str(temp_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True)
    temp_path.unlink(missing_ok=True)

    if output_path.exists():
        # Get video info
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        print(f"  Output: {output_path}")
        print(f"  Duration: {duration:.1f}s ({len(all_frames)} frames)")
        return output_path
    else:
        print(f"  ERROR: Failed to create video")
        return None


def process_multi_model_blob(image_dir, fps=24, seconds_per_frame=2.0, min_zoom=1.0, max_zoom=4.0,
                              vertical=False, sort_by='tile', sort_reverse=False, face_id=1):
    """Generate a video with all 5 model families shown simultaneously in blob regions.

    Args:
        image_dir: Directory containing styled/ subdirectory
        fps: Frames per second
        seconds_per_frame: Seconds per styled frame
        min_zoom: Ending zoom level
        max_zoom: Starting zoom level
        vertical: Vertical video output
        sort_by: Frame sorting method
        sort_reverse: Reverse sort order
        face_id: Which face to process (default: 1)
    """
    image_dir = Path(image_dir)
    styled_dir = image_dir / "styled"

    if not styled_dir.exists():
        print(f"No styled directory found: {styled_dir}")
        return None

    print(f"\nProcessing multi-model blob: {image_dir.name}")

    # Find original image
    image_name = image_dir.name
    orig_candidates = []

    self_style_dir = Path("/app/input/self_style_samples")
    if self_style_dir.exists():
        orig_candidates.extend(list(self_style_dir.glob(f"{image_name}.*")))
        orig_candidates.extend(list(self_style_dir.glob(f"{image_name}*")))

    if not orig_candidates:
        parent_input = Path("/app/input")
        orig_candidates.extend(list(parent_input.glob(f"{image_name}.*")))

    orig_candidates = [p for p in orig_candidates if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    if not orig_candidates:
        print(f"  Could not find original image")
        return None

    original_image = orig_candidates[0]
    print(f"  Original: {original_image.name}")

    # Video settings
    target_size = (720, 1280) if vertical else (1280, 720)

    # Get face center
    face_centers = get_face_centers_from_image(original_image, styled_dir)
    zoom_center = face_centers.get(face_id, (0.5, 0.5))
    print(f"  Using face {face_id} center: ({zoom_center[0]:.2f}, {zoom_center[1]:.2f})")

    # Define the 5 model families
    model_families = ['candy', 'mosaic', 'rain_princess', 'tenharmsel', 'training_image_2']

    # Collect frames for each model family
    model_frame_sets = {}
    for model_family in model_families:
        styled_paths, blended_images = collect_pytorch_styled_frames(
            styled_dir, face_id, sort_by=sort_by, reverse=sort_reverse,
            model_filter=model_family
        )
        if styled_paths:
            # Store as list of (path, image) tuples or just images if blended
            if blended_images:
                model_frame_sets[model_family] = list(zip(styled_paths, blended_images))
            else:
                # Load images
                loaded = []
                for p in styled_paths:
                    img = cv2.imread(str(p))
                    if img is not None:
                        loaded.append(img)
                model_frame_sets[model_family] = loaded
            print(f"  {model_family}: {len(model_frame_sets[model_family])} frames")
        else:
            print(f"  {model_family}: no frames found")

    # Filter to only models with frames
    active_models = [m for m in model_families if m in model_frame_sets and len(model_frame_sets[m]) > 0]

    if not active_models:
        print(f"  No model frames found")
        return None

    print(f"  Active models: {len(active_models)} ({', '.join(active_models)})")

    # Create the multi-model blob video
    print(f"  Creating multi-model blob video...")
    all_frames = create_multi_model_blob_video(
        model_frame_sets,
        active_models,
        original_image,
        target_size=target_size,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        fps=fps,
        seconds_per_frame=seconds_per_frame,
        zoom_center=zoom_center,
        blob_frequency=2.5,
        blob_speed=1.0
    )

    if not all_frames:
        print(f"  ERROR: No frames generated")
        return None

    # Write video
    output_path = image_dir / f"{image_dir.name}_multi_model_blob_face{face_id}.mp4"
    temp_path = image_dir / "temp_multi_blob.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tw, th = target_size
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (tw, th))

    for frame in all_frames:
        writer.write(frame)
    writer.release()

    # Re-encode with ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', str(temp_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True)
    temp_path.unlink(missing_ok=True)

    if output_path.exists():
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        print(f"  Output: {output_path}")
        print(f"  Duration: {duration:.1f}s ({len(all_frames)} frames)")
        return output_path
    else:
        print(f"  ERROR: Failed to create video")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch+Magenta only face zoom videos")
    parser.add_argument("--input_dir", type=str, default="/app/output/morph_faces",
                        help="Directory containing morph_faces output subdirectories")
    parser.add_argument("--image", type=str, default=None,
                        help="Process single image directory (e.g., IMG_3227)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--zoom_duration", type=float, default=4.0,
                        help="Seconds for zoom-out per face (default: 4.0)")
    parser.add_argument("--transition_duration", type=float, default=1.0,
                        help="Seconds for crossfade between faces (default: 1.0)")
    parser.add_argument("--min_zoom", type=float, default=1.0,
                        help="Ending zoom level (default: 1.0 = full view)")
    parser.add_argument("--max_zoom", type=float, default=4.0,
                        help="Starting zoom level (default: 4.0 = zoomed in on face)")
    parser.add_argument("--vertical", action="store_true",
                        help="Output vertical video (720x1280)")
    parser.add_argument("--sort", type=str, default="tile",
                        choices=["tile", "brightness", "model", "red", "green", "blue"],
                        help="Frame sorting: 'tile', 'brightness', 'model', 'red', 'green', 'blue'")
    parser.add_argument("--sort_reverse", action="store_true",
                        help="Reverse sort order (e.g., light to dark for brightness)")
    parser.add_argument("--seconds_per_frame", type=float, default=None,
                        help="Seconds per frame transition (overrides --zoom_duration, calculates based on frame count)")
    parser.add_argument("--blend_magenta", type=float, default=0.0,
                        help="Blend NST frames with plain Magenta frames (0.5 = 50%% each)")
    parser.add_argument("--blend_original", type=float, default=0.0,
                        help="3-way blend: NST + Magenta + Original (0.333 = 1/3 each)")
    parser.add_argument("--transition_effect", type=str, default="linear",
                        choices=["linear", "spiral", "blob", "radial", "dual_blob"],
                        help="Transition effect: 'linear', 'spiral', 'blob', 'radial', 'dual_blob' (two morphs with animated blob separation)")
    parser.add_argument("--model_filter", type=str, default=None,
                        help="Filter to only models starting with this prefix (e.g., 'candy', 'mosaic', 'tenharmsel')")
    parser.add_argument("--multi_model_blob", action="store_true",
                        help="Create single video with all 5 model families in animated blob regions")
    parser.add_argument("--face_id", type=int, default=1,
                        help="Face ID to use for multi_model_blob mode (default: 1)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Handle multi-model blob mode
    if args.multi_model_blob:
        if args.image:
            image_dir = input_dir / args.image
            if not image_dir.exists():
                print(f"Directory not found: {image_dir}")
                sys.exit(1)
            process_multi_model_blob(
                image_dir,
                fps=args.fps,
                seconds_per_frame=args.seconds_per_frame or 2.0,
                min_zoom=args.min_zoom,
                max_zoom=args.max_zoom,
                vertical=args.vertical,
                sort_by=args.sort,
                sort_reverse=args.sort_reverse,
                face_id=args.face_id
            )
        else:
            # Process all image directories
            for image_dir in sorted(input_dir.iterdir()):
                if image_dir.is_dir() and (image_dir / "styled").exists():
                    process_multi_model_blob(
                        image_dir,
                        fps=args.fps,
                        seconds_per_frame=args.seconds_per_frame or 2.0,
                        min_zoom=args.min_zoom,
                        max_zoom=args.max_zoom,
                        vertical=args.vertical,
                        sort_by=args.sort,
                        sort_reverse=args.sort_reverse,
                        face_id=args.face_id
                    )
        return

    if args.image:
        # Process single image
        image_dir = input_dir / args.image
        if not image_dir.exists():
            print(f"Directory not found: {image_dir}")
            sys.exit(1)
        process_image(
            image_dir,
            fps=args.fps,
            zoom_duration=args.zoom_duration,
            transition_duration=args.transition_duration,
            min_zoom=args.min_zoom,
            max_zoom=args.max_zoom,
            vertical=args.vertical,
            sort_by=args.sort,
            sort_reverse=args.sort_reverse,
            seconds_per_frame=args.seconds_per_frame,
            blend_magenta=args.blend_magenta,
            transition_effect=args.transition_effect,
            blend_original=args.blend_original,
            model_filter=args.model_filter
        )
    else:
        # Process all image directories
        for image_dir in sorted(input_dir.iterdir()):
            if image_dir.is_dir() and (image_dir / "styled").exists():
                process_image(
                    image_dir,
                    fps=args.fps,
                    zoom_duration=args.zoom_duration,
                    transition_duration=args.transition_duration,
                    min_zoom=args.min_zoom,
                    max_zoom=args.max_zoom,
                    vertical=args.vertical,
                    sort_by=args.sort,
                    sort_reverse=args.sort_reverse,
                    seconds_per_frame=args.seconds_per_frame,
                    blend_magenta=args.blend_magenta,
                    transition_effect=args.transition_effect,
                    blend_original=args.blend_original,
                    model_filter=args.model_filter
                )


if __name__ == "__main__":
    main()
