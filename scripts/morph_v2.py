#!/usr/bin/env python3
"""
MorphV2 - Fully Automated Self-Style Morph Video Pipeline

Given a single input image:
1. Run DeepLab semantic segmentation to detect ALL regions
2. Automatically select the best region using scoring algorithm
3. Run Magenta arbitrary style transfer with multiple tile/overlap configurations
4. Generate optical flow morph video from the styled outputs

AUTOMATIC MODE (recommended):
    docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input.jpg --auto"

MANUAL MODE (specify target):
    docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input.jpg --target_label person"

ANALYZE MODE (see what regions are detected):
    docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input.jpg --analyze"

SKIP MASK MODE (use whole image as style):
    docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input.jpg --skip_mask"

Scoring algorithm considers:
    - Coverage percentage (sweet spot 5-40%)
    - Aspect ratio (prefers square regions)
    - Position (slight preference for centered regions)
    - Semantic preference (person, animals > vehicles > furniture)

The script generates:
    - output/morphv2/<name>/styled/   - styled images at different tile sizes
    - output/morphv2/<name>/morph.mp4 - optical flow morph video
"""

import argparse
import os
import sys
import subprocess
import json
import random
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image

# Available PyTorch neural style models
PYTORCH_MODELS = [
    'candy',
    'mosaic',
    'rain_princess',
    'udnie',
]

# Tile configurations (tile, overlap) with 12.5% overlap ratio
TILE_CONFIGS = [
    (128, 16),
    (160, 20),
    (192, 24),
    (224, 28),
    (256, 32),
    (384, 48),
    (512, 64),
]

# VOC21 class labels for semantic segmentation
VOC21_LABELS = {
    "background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
    "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
    "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
    "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
    "sofa": 18, "train": 19, "tvmonitor": 20,
}


def detect_faces(image_path, confidence_threshold=0.5):
    """Detect faces in an image using OpenCV's DNN-based face detector.

    This uses a pre-trained SSD (Single Shot Detector) model that is much more
    accurate than Haar cascades, especially for varied poses and lighting.

    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for detection (0.0-1.0, default 0.5)

    Returns:
        List of face dictionaries with keys: 'bbox' (x, y, w, h), 'center', 'area', 'confidence', 'coverage'
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[faces] Failed to load image: {image_path}")
        return []

    h, w = img.shape[:2]

    # Load DNN face detector model
    model_dir = Path(__file__).parent.parent / "models" / "face_detector"
    prototxt_path = model_dir / "deploy.prototxt"
    caffemodel_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

    if not prototxt_path.exists() or not caffemodel_path.exists():
        print(f"[faces] Error: DNN face detector model not found at {model_dir}")
        print("[faces] Please download the model files:")
        print("  - deploy.prototxt")
        print("  - res10_300x300_ssd_iter_140000.caffemodel")
        return []

    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))

    # Create blob from image (resize to 300x300 as expected by the model)
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Run inference
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < confidence_threshold:
            continue

        # Get bounding box coordinates (model outputs normalized coordinates)
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        fw = x2 - x1
        fh = y2 - y1

        if fw <= 0 or fh <= 0:
            continue

        area = fw * fh
        coverage = area / (w * h) * 100
        center_x = x1 + fw / 2
        center_y = y1 + fh / 2

        results.append({
            'id': i + 1,
            'bbox': (x1, y1, fw, fh),
            'center': (center_x, center_y),
            'area': area,
            'coverage': coverage,
            'confidence': float(confidence),
            'aspect_ratio': fw / fh if fh > 0 else 1.0,
        })

    # Sort by area (largest first)
    results.sort(key=lambda f: f['area'], reverse=True)

    # Re-number IDs after sorting
    for i, face in enumerate(results):
        face['id'] = i + 1

    return results


def extract_face_region(image_path, face_bbox, output_path, padding_pct=0.3):
    """Extract a face region with padding as a crop.

    Args:
        image_path: Path to the input image
        face_bbox: Tuple of (x, y, w, h) for the face bounding box
        output_path: Path to save the cropped face
        padding_pct: Padding as percentage of face size (0.3 = 30% padding)

    Returns:
        True if successful, False otherwise
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    x, y, w, h = face_bbox
    img_h, img_w = img.shape[:2]

    # Add padding around face
    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(str(output_path), cropped)

    crop_w, crop_h = x2 - x1, y2 - y1
    print(f"[faces] Extracted face crop: {crop_w}x{crop_h} pixels (with {int(padding_pct*100)}% padding)")

    return True


def _ease_in_out_cubic(t):
    """Smooth easing: slow start, fast middle, slow end."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def _smoothstep(t):
    """Hermite interpolation for smooth blending (S-curve)."""
    return t * t * (3 - 2 * t)


def _smootherstep(t):
    """Ken Perlin's improved smoothstep - even smoother S-curve."""
    return t * t * t * (t * (6 * t - 15) + 10)


def temporal_smooth_frames(frames, kernel_size=3, sigma=1.0):
    """Apply temporal smoothing to reduce frame-to-frame jitter.

    Blends each frame with its neighbors using a Gaussian-weighted average.
    This reduces micro-jitter from optical flow while preserving motion.

    Args:
        frames: List of video frames (numpy arrays)
        kernel_size: Number of frames to blend (must be odd, default 3)
        sigma: Gaussian sigma for weighting (higher = more blur)

    Returns:
        List of temporally smoothed frames
    """
    if len(frames) < kernel_size:
        return frames

    # Create Gaussian weights
    half = kernel_size // 2
    weights = np.array([np.exp(-((i - half) ** 2) / (2 * sigma ** 2))
                        for i in range(kernel_size)])
    weights = weights / weights.sum()

    smoothed = []
    for i in range(len(frames)):
        # Gather neighboring frames
        blended = np.zeros_like(frames[i], dtype=np.float32)
        total_weight = 0

        for j, w in enumerate(weights):
            idx = i + j - half
            if 0 <= idx < len(frames):
                blended += frames[idx].astype(np.float32) * w
                total_weight += w

        # Normalize and convert back
        blended = (blended / total_weight).astype(np.uint8)
        smoothed.append(blended)

    return smoothed


def apply_hue_shift(frame, shift_degrees):
    """Shift the hue of a frame by the specified degrees (0-360).

    Args:
        frame: BGR image (numpy array)
        shift_degrees: Hue shift in degrees (0-360)

    Returns:
        Hue-shifted BGR image
    """
    if abs(shift_degrees) < 0.1:
        return frame

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Shift hue (OpenCV hue is 0-180, so divide by 2)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_degrees / 2) % 180

    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def calculate_zoom_pulse(progress, pulse_amplitude=0.05, pulse_frequency=2.0):
    """Calculate zoom multiplier for breathing/pulsing effect.

    Args:
        progress: Overall video progress (0.0 to 1.0)
        pulse_amplitude: How much to zoom in/out (0.05 = 5% variation)
        pulse_frequency: Number of complete pulse cycles in the video

    Returns:
        Zoom multiplier (e.g., 1.05 for 5% zoom in)
    """
    # Use absolute sine wave for smooth breathing effect (always zoom in, never out)
    # This avoids issues with zooming below 1.0 which causes crop failures
    pulse = abs(np.sin(progress * pulse_frequency * 2 * np.pi))
    return 1.0 + pulse * pulse_amplitude


def optical_flow_morph(img1, img2, num_interp_frames=72, easing='smooth'):
    """Generate interpolated frames between two images using optical flow.

    Uses improved flow estimation with smoothing to reduce artifacts like
    vertical lines that can occur with large style differences.

    Args:
        img1, img2: Source and destination images
        num_interp_frames: Number of frames to generate
        easing: Easing mode for transitions:
            - 'linear': No easing (abrupt)
            - 'smooth': Cubic ease-in-out (recommended)
            - 'smoother': Ken Perlin's smootherstep (very smooth)
    """
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply slight Gaussian blur to reduce high-frequency noise that causes artifacts
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 1.0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 1.0)

    # Use more robust flow parameters to reduce artifacts
    flow_forward = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=6, winsize=21,  # Larger window, more levels
        iterations=5, poly_n=7, poly_sigma=1.5,  # More iterations
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    flow_backward = cv2.calcOpticalFlowFarneback(
        gray2, gray1, None,
        pyr_scale=0.5, levels=6, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # Smooth the flow fields to reduce artifacts (vertical lines, etc.)
    flow_forward[:, :, 0] = cv2.GaussianBlur(flow_forward[:, :, 0], (15, 15), 3.0)
    flow_forward[:, :, 1] = cv2.GaussianBlur(flow_forward[:, :, 1], (15, 15), 3.0)
    flow_backward[:, :, 0] = cv2.GaussianBlur(flow_backward[:, :, 0], (15, 15), 3.0)
    flow_backward[:, :, 1] = cv2.GaussianBlur(flow_backward[:, :, 1], (15, 15), 3.0)

    # Add minimum flow to ensure all pixels morph (not just areas with detected motion)
    # This prevents static-looking regions where optical flow is near-zero
    flow_magnitude_fwd = np.sqrt(flow_forward[:, :, 0]**2 + flow_forward[:, :, 1]**2)
    flow_magnitude_bwd = np.sqrt(flow_backward[:, :, 0]**2 + flow_backward[:, :, 1]**2)

    # Minimum displacement of 2 pixels ensures visible morphing everywhere
    min_flow = 2.0

    # For areas with low flow, add radial flow toward/from center for organic look
    cy, cx = h / 2, w / 2
    radial_y = (np.arange(h)[:, None] - cy) / h
    radial_x = (np.arange(w)[None, :] - cx) / w

    # Boost low-flow areas with radial displacement
    low_flow_mask_fwd = (flow_magnitude_fwd < min_flow).astype(np.float32)
    low_flow_mask_bwd = (flow_magnitude_bwd < min_flow).astype(np.float32)

    # Add subtle radial flow where optical flow is weak
    flow_forward[:, :, 0] += low_flow_mask_fwd * radial_x * min_flow * 2
    flow_forward[:, :, 1] += low_flow_mask_fwd * radial_y * min_flow * 2
    flow_backward[:, :, 0] -= low_flow_mask_bwd * radial_x * min_flow * 2
    flow_backward[:, :, 1] -= low_flow_mask_bwd * radial_y * min_flow * 2

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Select easing function
    if easing == 'smoother':
        ease_func = _smootherstep
    elif easing == 'smooth':
        ease_func = _ease_in_out_cubic
    else:  # 'linear'
        ease_func = lambda x: x

    frames = []
    for i in range(num_interp_frames):
        # Linear progress through frames
        t_linear = i / (num_interp_frames - 1) if num_interp_frames > 1 else 0

        # Apply easing to both warp timing and blend alpha
        t = ease_func(t_linear)

        # Warp both images toward each other
        map1_x = x_coords + t * flow_forward[:, :, 0]
        map1_y = y_coords + t * flow_forward[:, :, 1]
        warped1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        map2_x = x_coords + (1 - t) * flow_backward[:, :, 0]
        map2_y = y_coords + (1 - t) * flow_backward[:, :, 1]
        warped2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # Use smoothstep for alpha blending (extra smooth crossfade)
        alpha = _smoothstep(t_linear)
        blended = cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)
        frames.append(blended)

    return frames


def run_deeplab_segmentation(image_path, output_mask_path, target_ids,
                              weights_path, resolution=1024):
    """Run DeepLab segmentation to generate mask for target classes."""
    sky_swap_path = Path(__file__).parent.parent / "sky_swap.py"

    cmd = [
        "python3", str(sky_swap_path),
        "--image", str(image_path),
        "--weights", str(weights_path),
        "--backbone", "resnet",
        "--resolution", str(resolution),
        "--target_ids", target_ids,
        "--mask_feather_pct", "0.5",
        "--out_mask", str(output_mask_path)
    ]

    print(f"[mask] Running DeepLab segmentation...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[mask] Error: {result.stderr}")
        return False

    return output_mask_path.exists()


def extract_masked_region(image_path, mask_path, output_path, padding=0):
    """Extract the masked region as a tight bounding box crop.

    This crops the original image to just the bounding box of the mask,
    keeping all original pixels without any black masked areas.
    No padding is added by default to ensure a tight crop.
    """
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"[extract] Failed to load image or mask")
        return False

    # Resize mask to match image dimensions if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Find bounding box of mask
    coords = cv2.findNonZero(mask)
    if coords is None:
        print("[extract] No masked region found")
        return False

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding if specified (default is 0 for tight crop)
    if padding > 0:
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)

    # Crop the original image to bounding box (no mask applied)
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite(str(output_path), cropped)

    print(f"[extract] Extracted tight crop: {w}x{h} pixels (no black areas)")

    return True


def run_magenta_style(content_path, style_path, output_path, tile, overlap, scale=1440, blend=0.95):
    """Run Magenta arbitrary style transfer."""
    pipeline_path = Path(__file__).parent.parent / "pipeline.py"

    cmd = [
        "python3", str(pipeline_path),
        "--input_image", str(content_path),
        "--output_image", str(output_path),
        "--model_type", "magenta",
        "--magenta_style", str(style_path),
        "--magenta_tile", str(tile),
        "--magenta_overlap", str(overlap),
        "--scale", str(scale),
        "--blend", str(blend),
    ]

    print(f"    tile={tile}, overlap={overlap}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:200]}")
        return False

    return Path(output_path).exists()


def run_pytorch_style(content_path, output_path, model_name=None, scale=720, blend=0.95, inference_res=1440):
    """
    Run PyTorch neural style transfer on an image.

    Args:
        content_path: Path to input image
        output_path: Path for styled output
        model_name: Name of PyTorch model (candy, mosaic, rain_princess, udnie)
                   If None, picks a random one
        scale: Output resolution (default 720)
        blend: Style blend ratio
        inference_res: Max inference resolution (pipeline.py handles downscale/upscale)

    Returns:
        Tuple of (success, model_used)
    """
    pipeline_path = Path(__file__).parent.parent / "pipeline.py"

    # Pick random model if not specified
    if model_name is None:
        model_name = random.choice(PYTORCH_MODELS)

    model_path = f"/app/models/pytorch/{model_name}.pth"

    cmd = [
        "python3", str(pipeline_path),
        "--input_image", str(content_path),
        "--output_image", str(output_path),
        "--model_type", "transformer",
        "--model", model_path,
        "--io_preset", "raw_255",
        "--scale", str(scale),
        "--blend", str(blend),
        "--inference_res", str(inference_res),
    ]

    print(f"[pytorch] Applying {model_name} style (scale={scale}, inference_res={inference_res})...")
    print(f"[pytorch] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "No error message"
        stdout_msg = result.stdout.strip() if result.stdout else "No stdout"
        print(f"[pytorch] ERROR (code {result.returncode}):")
        print(f"[pytorch] stderr: {error_msg[:500]}")
        print(f"[pytorch] stdout: {stdout_msg[:500]}")
        return False, model_name

    if not Path(output_path).exists():
        print(f"[pytorch] ERROR: Output file not created at {output_path}")
        return False, model_name

    return True, model_name


def create_morph_video(styled_dir, output_path, fps=24, morph_seconds=2.0,
                       target_size=(720, 1280), zoom=1.0, hold_frames=24,
                       pan_zoom=None, pan_direction='horizontal', easing='smooth',
                       temporal_smooth=0, zoom_pulse=0.0, zoom_pulse_freq=2.0,
                       hue_rotate=0.0, zoom_in_pct=0.25):
    """Create optical flow morph video from styled images.

    Args:
        styled_dir: Directory containing styled images (including blend variants)
        output_path: Output video path
        fps: Frames per second
        morph_seconds: Duration of each morph transition in seconds
        target_size: Video dimensions (width, height)
        zoom: Static zoom factor for images (default 1.0)
        hold_frames: Number of frames to hold on final image
        pan_zoom: Ken Burns zoom level (e.g., 2.0 = show 50% of image, pan across rest)
                  If None, no pan effect is applied
        easing: Easing mode for transitions ('linear', 'smooth', 'smoother')
        pan_direction: Direction to pan - 'horizontal', 'vertical', 'diagonal', 'diagonal_reverse'
        temporal_smooth: Kernel size for temporal smoothing (0=disabled, 3-5 recommended)
        zoom_pulse: Amplitude of zoom pulsing effect (0.0=disabled, 0.03-0.08 subtle)
        zoom_pulse_freq: Frequency of zoom pulse cycles per video
        hue_rotate: Total hue rotation in degrees over video duration (0=disabled)
        zoom_in_pct: Percentage of video for zoom-in phase (0.0-1.0, default 0.25)
    """
    interp_frames = int(fps * morph_seconds)

    # Collect images in order
    # First: blend variants (blend_0, blend_25, blend_50, blend_75, blend_100)
    # Then: pure Magenta tile variants (tile128 through tile512)
    images = []
    image_labels = []

    # Collect all blend variants (processed through full pipeline) with all tile configs
    # Order: blend0_tile128 -> blend0_tile160 -> ... -> blend0_tile512 -> blend25_tile128 -> ...
    blend_ratios = [0, 25, 50, 75, 100]
    for ratio in blend_ratios:
        for tile, overlap in TILE_CONFIGS:
            pattern = f"*_blend{ratio}_tile{tile}_overlap{overlap}.jpg"
            matches = list(styled_dir.glob(pattern))
            if matches:
                images.append(matches[0])
                image_labels.append(f"blend{ratio}%_tile{tile}")

    # Add pure Magenta styled images (using original crop as style source)
    for tile, overlap in TILE_CONFIGS:
        pattern = f"*_tile{tile}_overlap{overlap}.jpg"
        matches = list(styled_dir.glob(pattern))
        # Filter out blend variants
        pure_matches = [m for m in matches if '_blend' not in m.name]
        if pure_matches:
            images.append(pure_matches[0])
            image_labels.append(f"magenta_tile{tile}")

    if len(images) < 2:
        print(f"[video] Need at least 2 images, found {len(images)}")
        return False

    print(f"[video] Creating morph video from {len(images)} images ({morph_seconds}s per morph)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)

    def load_and_resize(path):
        """Load a single image and resize to target size with zoom."""
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Apply zoom
        if zoom > 1.0:
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
            h, w = img.shape[:2]

        # Scale to fill target (cover mode)
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]
        return img

    def load_for_pan(path):
        """Load image scaled up for pan/zoom effect without cropping."""
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Apply static zoom first
        if zoom > 1.0:
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
            h, w = img.shape[:2]

        # Scale up by pan_zoom factor so we can pan across
        # We need the image to be pan_zoom times larger than target
        scale = max(target_w / w, target_h / h) * pan_zoom
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return img

    def ease_in_out_cubic(t):
        """Smooth easing function: slow start, fast middle, slow end."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def extract_pan_frame(full_img, progress):
        """Extract a frame with zoom-in then pan effect.

        Phase 1 (0 to zoom_in_pct): Zoom in from full view to pan_zoom level
        Phase 2 (zoom_in_pct to 1.0): Pan across at the zoomed level

        Uses cubic easing for smooth motion and sub-pixel precision.
        Uses zoom_in_pct from outer scope.
        """
        target_w, target_h = target_size
        h, w = full_img.shape[:2]

        # Determine which phase we're in
        if progress < zoom_in_pct and zoom_in_pct > 0:
            # PHASE 1: Zoom in
            # zoom_progress goes from 0 to 1 during zoom phase
            zoom_progress = progress / zoom_in_pct
            eased_zoom = ease_in_out_cubic(zoom_progress)

            # Calculate crop size: start with full image, shrink to target_size
            # At zoom_progress=0: crop = full image size
            # At zoom_progress=1: crop = target_size
            start_crop_w = min(w, int(target_w * pan_zoom))  # Full pan area
            start_crop_h = min(h, int(target_h * pan_zoom))

            crop_w = int(start_crop_w + (target_w - start_crop_w) * eased_zoom)
            crop_h = int(start_crop_h + (target_h - start_crop_h) * eased_zoom)

            # Keep aspect ratio
            crop_w = max(target_w, crop_w)
            crop_h = max(target_h, crop_h)

            # Center the crop during zoom
            center_x = w / 2.0
            center_y = h / 2.0

            # Extract larger crop and resize to target
            crop = cv2.getRectSubPix(full_img, (crop_w, crop_h), (center_x, center_y))
            return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        else:
            # PHASE 2: Pan across (starting from center where zoom ended)
            # Remap progress to 0-1 for pan phase
            if zoom_in_pct < 1.0:
                pan_progress = (progress - zoom_in_pct) / (1.0 - zoom_in_pct)
            else:
                pan_progress = 0.0

            # Calculate available pan distance
            max_pan_x = w - target_w
            max_pan_y = h - target_h

            # Apply easing for smooth motion
            eased = ease_in_out_cubic(pan_progress)

            # Pan starts from CENTER (where zoom ended) and moves to edge
            # This ensures seamless transition from zoom to pan
            if pan_direction == 'horizontal':
                # Start at center, pan to right edge
                x = (max_pan_x / 2.0) + eased * (max_pan_x / 2.0)
                y = max_pan_y / 2.0
            elif pan_direction == 'vertical':
                # Start at center, pan to bottom
                x = max_pan_x / 2.0
                y = (max_pan_y / 2.0) + eased * (max_pan_y / 2.0)
            elif pan_direction == 'diagonal':
                # Start at center, pan to bottom-right
                x = (max_pan_x / 2.0) + eased * (max_pan_x / 2.0)
                y = (max_pan_y / 2.0) + eased * (max_pan_y / 2.0)
            elif pan_direction == 'diagonal_reverse':
                # Start at center, pan to bottom-left
                x = (max_pan_x / 2.0) - eased * (max_pan_x / 2.0)
                y = (max_pan_y / 2.0) + eased * (max_pan_y / 2.0)
            else:
                x = (max_pan_x / 2.0) + eased * (max_pan_x / 2.0)
                y = max_pan_y / 2.0

            # Ensure bounds
            x = max(0.0, min(x, float(max_pan_x)))
            y = max(0.0, min(y, float(max_pan_y)))

            # Calculate center point for extraction
            center_x = x + target_w / 2.0
            center_y = y + target_h / 2.0

            return cv2.getRectSubPix(full_img, (target_w, target_h), (center_x, center_y))

    def get_label(idx):
        """Get label for image at index."""
        if idx < len(image_labels):
            return image_labels[idx]
        return f"image_{idx}"

    # If pan_zoom is enabled, we need to pan across each morph frame
    # Load images at pan_zoom size, morph between them, then extract panned crops
    if pan_zoom is not None and pan_zoom > 1.0:
        print(f"[video] Ken Burns effect enabled: {pan_zoom}x zoom, {pan_direction} pan")

        # Print enabled effects
        effects_enabled = []
        if temporal_smooth > 0:
            effects_enabled.append(f"temporal_smooth={temporal_smooth}")
        if zoom_pulse > 0:
            effects_enabled.append(f"zoom_pulse={zoom_pulse:.2f}")
        if hue_rotate != 0:
            effects_enabled.append(f"hue_rotate={hue_rotate:.0f}°")
        if effects_enabled:
            print(f"[video] Effects: {', '.join(effects_enabled)}")

        # Calculate total frames for progress tracking
        num_transitions = len(images) - 1
        frames_per_transition = interp_frames
        expected_total = num_transitions * frames_per_transition + hold_frames

        # Collect all frames first (for temporal smoothing and effects)
        all_frames = []
        global_frame = 0

        for idx in range(len(images)):
            print(f"  [{idx+1}/{len(images)}] {get_label(idx)}")

            if idx < len(images) - 1:
                # Load both images at pan_zoom size
                curr_pan = load_for_pan(images[idx])
                next_pan = load_for_pan(images[idx + 1])

                if curr_pan is not None and next_pan is not None:
                    try:
                        # Morph at the larger pan_zoom size
                        morph_frames = optical_flow_morph(curr_pan, next_pan, interp_frames, easing=easing)

                        # Extract panned crops from each morphed frame
                        for frame in morph_frames:
                            progress = global_frame / max(1, expected_total - 1)

                            # Apply zoom pulse if enabled
                            if zoom_pulse > 0:
                                pulse_zoom = calculate_zoom_pulse(progress, zoom_pulse, zoom_pulse_freq)
                                # Scale the frame slightly based on pulse
                                h, w = frame.shape[:2]
                                new_w, new_h = int(w * pulse_zoom), int(h * pulse_zoom)
                                scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                                # Center crop back to original size
                                start_x = (new_w - w) // 2
                                start_y = (new_h - h) // 2
                                frame = scaled[start_y:start_y+h, start_x:start_x+w]

                            panned = extract_pan_frame(frame, progress)

                            # Apply hue rotation if enabled
                            if hue_rotate != 0:
                                hue_shift = progress * hue_rotate
                                panned = apply_hue_shift(panned, hue_shift)

                            all_frames.append(panned)
                            global_frame += 1
                    except Exception as e:
                        print(f"    Morph failed: {e}")
                        # Fallback: crossfade with pan
                        for i in range(interp_frames):
                            t = i / (interp_frames - 1)
                            blended = cv2.addWeighted(curr_pan, 1 - t, next_pan, t, 0)
                            progress = global_frame / max(1, expected_total - 1)
                            panned = extract_pan_frame(blended, progress)
                            if hue_rotate != 0:
                                panned = apply_hue_shift(panned, progress * hue_rotate)
                            all_frames.append(panned)
                            global_frame += 1

        # Add hold frames at the end (continue panning)
        if hold_frames > 0:
            print(f"  [hold] Adding {hold_frames} hold frames at end")
            last_pan = load_for_pan(images[-1])
            if last_pan is not None:
                for i in range(hold_frames):
                    progress = global_frame / max(1, expected_total - 1)

                    # Apply zoom pulse to hold frames too (for consistency)
                    frame = last_pan
                    if zoom_pulse > 0:
                        pulse_zoom = calculate_zoom_pulse(progress, zoom_pulse, zoom_pulse_freq)
                        h, w = frame.shape[:2]
                        new_w, new_h = int(w * pulse_zoom), int(h * pulse_zoom)
                        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        start_x = (new_w - w) // 2
                        start_y = (new_h - h) // 2
                        frame = scaled[start_y:start_y+h, start_x:start_x+w]

                    panned = extract_pan_frame(frame, progress)
                    if hue_rotate != 0:
                        panned = apply_hue_shift(panned, progress * hue_rotate)
                    all_frames.append(panned)
                    global_frame += 1

        # Apply temporal smoothing if enabled
        if temporal_smooth > 0 and len(all_frames) > temporal_smooth:
            print(f"  [smooth] Applying temporal smoothing (kernel={temporal_smooth})...")
            all_frames = temporal_smooth_frames(all_frames, kernel_size=temporal_smooth, sigma=1.0)

        # Write all frames to video
        total_frames = len(all_frames)
        for frame in all_frames:
            out.write(frame)

    else:
        # No pan effect - standard processing
        total_frames = 0
        for idx in range(len(images)):
            curr_img = load_and_resize(images[idx])
            if curr_img is None:
                continue

            print(f"  [{idx+1}/{len(images)}] {get_label(idx)}")

            if idx < len(images) - 1:
                next_img = load_and_resize(images[idx + 1])
                if next_img is not None:
                    try:
                        morph_frames = optical_flow_morph(curr_img, next_img, interp_frames, easing=easing)
                        for frame in morph_frames:
                            out.write(frame)
                            total_frames += 1
                    except Exception as e:
                        print(f"    Morph failed: {e}")
                        for i in range(interp_frames):
                            t = i / (interp_frames - 1)
                            blended = cv2.addWeighted(curr_img, 1 - t, next_img, t, 0)
                            out.write(blended)
                            total_frames += 1

        # Add hold frames at the end
        if hold_frames > 0 and total_frames > 0:
            print(f"  [hold] Adding {hold_frames} hold frames at end")
            # Get the last image
            last_img = load_and_resize(images[-1])
            if last_img is not None:
                for _ in range(hold_frames):
                    out.write(last_img)
                    total_frames += 1

    out.release()

    # Convert to H.264
    h264_path = str(output_path).replace('.mp4', '_h264.mp4')
    print(f"[video] Converting to H.264...")
    os.system(f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset medium -crf 23 "{h264_path}" 2>/dev/null')

    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
        os.remove(output_path)
        os.rename(h264_path, output_path)

    duration = total_frames / fps
    print(f"[video] Complete: {total_frames} frames, {duration:.1f}s")
    return True


def analyze_all_masks(image_path, weights_path, resolution=512):
    """
    Run DeepLab segmentation and analyze ALL detected semantic regions.
    Returns a list of detected regions with their statistics, sorted by score.
    """
    from modeling.deeplab import DeepLab
    import torch
    from torchvision import transforms

    print("[auto-detect] Running DeepLab to detect all semantic regions...")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLab(num_classes=21, backbone='resnet')

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"[auto-detect] Warning: weights not found at {weights_path}")
        return []

    model = model.to(device)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    # Resize for inference
    scale = resolution / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()

    # Analyze each class present in the prediction
    total_pixels = pred.size
    regions = []
    id_to_label = {v: k for k, v in VOC21_LABELS.items()}

    for class_id in range(21):
        if class_id == 0:  # Skip background
            continue

        mask = (pred == class_id)
        pixel_count = mask.sum()

        if pixel_count < 100:  # Skip tiny regions (noise)
            continue

        coverage_pct = (pixel_count / total_pixels) * 100
        label = id_to_label.get(class_id, f"class_{class_id}")

        # Find bounding box
        coords = np.where(mask)
        if len(coords[0]) == 0:
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Calculate centroid position (normalized 0-1)
        center_y = (y_min + y_max) / 2 / new_h
        center_x = (x_min + x_max) / 2 / new_w

        # Calculate "interestingness" score
        # Higher score = better candidate for style extraction
        score = calculate_region_score(
            coverage_pct,
            bbox_w, bbox_h,
            center_x, center_y,
            new_w, new_h,
            label
        )

        regions.append({
            'class_id': class_id,
            'label': label,
            'pixel_count': int(pixel_count),
            'coverage_pct': coverage_pct,
            'bbox': (x_min, y_min, bbox_w, bbox_h),
            'center': (center_x, center_y),
            'score': score
        })

    # Sort by score (highest first)
    regions.sort(key=lambda x: x['score'], reverse=True)

    return regions


def calculate_region_score(coverage_pct, bbox_w, bbox_h, center_x, center_y,
                           img_w, img_h, label):
    """
    Calculate a score for how good a region is for style extraction.
    Higher score = better candidate.

    Factors considered:
    - Coverage: Sweet spot is 5-40% of image (not too small, not too large)
    - Aspect ratio: Prefer more square regions
    - Position: Slight preference for centered regions
    - Semantic preference: Some labels are better style sources (person, animals)
    """
    score = 0.0

    # Coverage score (bell curve: optimal around 10-25%)
    if coverage_pct < 2:
        score += coverage_pct * 5  # Very small - low score
    elif coverage_pct < 5:
        score += 10 + (coverage_pct - 2) * 10  # Small but usable
    elif coverage_pct < 25:
        score += 40 + (coverage_pct - 5) * 2  # Sweet spot
    elif coverage_pct < 50:
        score += 80 - (coverage_pct - 25)  # Getting too large
    else:
        score += 55 - (coverage_pct - 50) * 0.5  # Very large, diminishing

    # Aspect ratio bonus (prefer square-ish regions)
    if bbox_w > 0 and bbox_h > 0:
        aspect = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)
        score += aspect * 15  # Up to 15 points for square

    # Center bias (slight preference for centered regions)
    dist_from_center = ((center_x - 0.5)**2 + (center_y - 0.5)**2) ** 0.5
    score += (1 - dist_from_center) * 10  # Up to 10 points for centered

    # Semantic preference (some objects make better style sources)
    preferred_labels = ['person', 'cat', 'dog', 'bird', 'horse', 'cow', 'sheep']
    good_labels = ['car', 'motorbike', 'bicycle', 'bus', 'train', 'aeroplane', 'boat']

    if label in preferred_labels:
        score += 25  # Strong preference for living things
    elif label in good_labels:
        score += 15  # Vehicles/transport are decent
    else:
        score += 5   # Furniture, bottles, etc. - lower preference

    return score


def select_best_region(regions, min_coverage=1.0, max_coverage=60.0):
    """
    Select the best region from detected regions based on scoring.

    Args:
        regions: List of region dicts from analyze_all_masks()
        min_coverage: Minimum coverage percentage to consider
        max_coverage: Maximum coverage percentage to consider

    Returns:
        Best region dict, or None if no suitable region found
    """
    candidates = [
        r for r in regions
        if min_coverage <= r['coverage_pct'] <= max_coverage
    ]

    if not candidates:
        # Fall back to any region
        candidates = regions

    if not candidates:
        return None

    return candidates[0]  # Already sorted by score


def main():
    parser = argparse.ArgumentParser(description='MorphV2 - Automated Self-Style Morph Pipeline')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--auto', action='store_true',
                        help='Automatically detect and select best region (no manual label needed)')
    parser.add_argument('--target_label', help='Target semantic label (e.g., person, car, dog)')
    parser.add_argument('--target_id', type=int, help='Target class ID (alternative to --target_label)')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: output/morphv2/<name>)')
    parser.add_argument('--name', default=None, help='Output name (default: derived from image)')
    parser.add_argument('--output_suffix', default='', help='Suffix to append to output directory name (e.g., "_v2")')
    parser.add_argument('--scale', type=int, default=1440, help='Output resolution for styled images')
    parser.add_argument('--blend', type=float, default=0.95, help='Style blend ratio')
    parser.add_argument('--fps', type=int, default=24, help='Video framerate')
    parser.add_argument('--morph_seconds', type=float, default=2.0, help='Seconds per morph transition')
    parser.add_argument('--easing', default='smooth', choices=['linear', 'smooth', 'smoother'],
                        help='Easing mode for morph transitions (default: smooth)')
    parser.add_argument('--hold_frames', type=int, default=24, help='Frames to hold on final image')
    parser.add_argument('--num_blend_frames', type=int, default=5,
                        help='Number of blend levels between original and pytorch (default 5)')
    parser.add_argument('--zoom', type=float, default=1.0, help='Static video zoom factor (default: 1.0)')
    parser.add_argument('--pan_zoom', type=float, default=None,
                        help='Ken Burns pan/zoom level (e.g., 2.0 = zoom in 2x and pan across image)')
    parser.add_argument('--pan_direction', default='horizontal',
                        choices=['horizontal', 'vertical', 'diagonal', 'diagonal_reverse'],
                        help='Pan direction for Ken Burns effect (default: horizontal)')
    parser.add_argument('--zoom_in_pct', type=float, default=0.25,
                        help='Percentage of video for zoom-in phase (0.0-1.0, default: 0.25 = 25%%)')
    parser.add_argument('--temporal_smooth', type=int, default=0, metavar='N',
                        help='Temporal smoothing kernel size (odd number, 0=disabled, 3-5 recommended)')
    parser.add_argument('--zoom_pulse', type=float, default=0.0, metavar='AMP',
                        help='Zoom pulse amplitude (0.0=disabled, 0.03-0.08 for subtle breathing effect)')
    parser.add_argument('--zoom_pulse_freq', type=float, default=2.0,
                        help='Zoom pulse frequency - cycles per video (default: 2.0)')
    parser.add_argument('--hue_rotate', type=float, default=0.0, metavar='DEG',
                        help='Hue rotation per video cycle in degrees (0=disabled, 30-60 for subtle shift)')
    parser.add_argument('--vertical', action='store_true', help='Vertical video (720x1280)')
    parser.add_argument('--skip_mask', action='store_true', help='Skip mask step, use whole image as style')
    parser.add_argument('--skip_video', action='store_true', help='Skip video generation')
    parser.add_argument('--force', action='store_true', help='Force regenerate existing outputs')
    parser.add_argument('--list_labels', action='store_true', help='List available semantic labels')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze image and show all detected regions (no processing)')
    parser.add_argument('--detect_faces', action='store_true',
                        help='Use face detection instead of semantic segmentation')
    parser.add_argument('--face_index', type=int, default=1,
                        help='Which face to use as style source (1=largest, 2=second largest, etc.)')
    parser.add_argument('--face_padding', type=float, default=0.3,
                        help='Padding around face as percentage (0.3 = 30%% padding)')
    parser.add_argument('--pytorch_style', action='store_true',
                        help='Pre-style the cropped region with a random PyTorch neural style model')
    parser.add_argument('--pytorch_model', default=None,
                        help=f'Specific PyTorch model to use (default: random). Options: {", ".join(PYTORCH_MODELS)}')
    parser.add_argument('--min_coverage', type=float, default=1.0,
                        help='Minimum coverage %% for auto-selection (default: 1.0)')
    parser.add_argument('--max_coverage', type=float, default=60.0,
                        help='Maximum coverage %% for auto-selection (default: 60.0)')
    parser.add_argument('--weights', default='/app/models/deeplab/deeplab-resnet.pth.tar',
                        help='Path to DeepLab weights')
    args = parser.parse_args()

    if args.list_labels:
        print("\nAvailable VOC21 semantic labels:")
        for label, id in sorted(VOC21_LABELS.items(), key=lambda x: x[1]):
            print(f"  {id:2d}: {label}")
        return

    # Validate inputs
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[error] Image not found: {image_path}")
        return 1

    # Analyze mode - just show detected regions
    if args.analyze:
        print(f"\n[analyze] Analyzing: {image_path}")

        # Always run face detection
        print("\n[analyze] Face Detection:")
        faces = detect_faces(str(image_path))
        if faces:
            print(f"  Detected {len(faces)} face(s):\n")
            print(f"  {'#':<3} {'Size':<12} {'Coverage':<10} {'Position':<20}")
            print("  " + "-" * 50)
            for f in faces:
                x, y, w, h = f['bbox']
                print(f"  {f['id']:<3} {w}x{h:<8} {f['coverage']:<10.1f}% ({x}, {y})")
        else:
            print("  No faces detected")

        # Also run semantic segmentation
        print("\n[analyze] Semantic Segmentation:")
        regions = analyze_all_masks(str(image_path), args.weights, resolution=512)
        if not regions:
            print("  No semantic regions detected")
        else:
            print(f"  Detected {len(regions)} region(s):\n")
            print(f"  {'Rank':<5} {'Label':<15} {'Coverage':<10} {'Score':<8} {'Center':<12}")
            print("  " + "-" * 55)
            for i, r in enumerate(regions, 1):
                cx, cy = r['center']
                print(f"  {i:<5} {r['label']:<15} {r['coverage_pct']:<10.1f} {r['score']:<8.1f} ({cx:.2f}, {cy:.2f})")

            best = select_best_region(regions, args.min_coverage, args.max_coverage)
            if best:
                print(f"\n[analyze] Best region for styling: {best['label']} (score={best['score']:.1f})")

        return 0

    # Determine target class ID or face
    target_id = None
    selected_label = None
    detected_face = None  # Will hold face bbox if using face detection

    # Face detection mode
    if args.detect_faces:
        print(f"\n[faces] Detecting faces in image...")
        faces = detect_faces(str(image_path))

        if not faces:
            print("[faces] No faces detected - falling back to whole image")
            args.skip_mask = True
        else:
            print(f"[faces] Found {len(faces)} face(s):")
            for f in faces:
                x, y, w, h = f['bbox']
                print(f"  #{f['id']}: {w}x{h} pixels, {f['coverage']:.1f}% of image")

            # Select face by index (1-based)
            face_idx = args.face_index - 1
            if face_idx >= len(faces):
                print(f"[faces] Requested face #{args.face_index} but only {len(faces)} found, using largest")
                face_idx = 0

            detected_face = faces[face_idx]
            x, y, w, h = detected_face['bbox']
            print(f"\n[faces] Selected face #{detected_face['id']}: {w}x{h} at ({x}, {y})")

    elif args.auto:
        # Automatic detection mode
        print(f"\n[auto] Analyzing image for best region...")
        regions = analyze_all_masks(str(image_path), args.weights, resolution=512)

        if not regions:
            print("[auto] No semantic regions detected - falling back to whole image")
            args.skip_mask = True
        else:
            # Show all detected regions
            print(f"[auto] Found {len(regions)} regions:")
            for i, r in enumerate(regions[:5], 1):  # Show top 5
                print(f"  {i}. {r['label']}: {r['coverage_pct']:.1f}% coverage, score={r['score']:.1f}")

            # Select best region
            best = select_best_region(regions, args.min_coverage, args.max_coverage)
            if best:
                target_id = best['class_id']
                selected_label = best['label']
                print(f"\n[auto] Selected: {selected_label} (id={target_id}, score={best['score']:.1f})")
            else:
                print("[auto] No suitable region found - falling back to whole image")
                args.skip_mask = True

    elif args.target_id is not None:
        target_id = args.target_id
    elif args.target_label:
        label_lower = args.target_label.lower().strip()
        if label_lower in VOC21_LABELS:
            target_id = VOC21_LABELS[label_lower]
            selected_label = label_lower
        else:
            print(f"[error] Unknown label: {args.target_label}")
            print(f"[error] Available: {', '.join(VOC21_LABELS.keys())}")
            return 1
    elif not args.skip_mask:
        print("[error] Must specify --auto, --target_label, --target_id, or --skip_mask")
        return 1

    # Setup output directory
    name = args.name or image_path.stem
    name_with_suffix = name + args.output_suffix
    base_output = Path(args.output_dir) if args.output_dir else Path("/app/output/morphv2") / name_with_suffix
    base_output.mkdir(parents=True, exist_ok=True)

    styled_dir = base_output / "styled"
    styled_dir.mkdir(exist_ok=True)

    work_dir = base_output / "work"
    work_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MORPHV2 PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {image_path}")
    print(f"Output: {base_output}")
    if detected_face is not None:
        x, y, w, h = detected_face['bbox']
        print(f"Target: Face #{detected_face['id']} ({w}x{h} pixels)")
    elif target_id is not None:
        label_name = selected_label or [k for k, v in VOC21_LABELS.items() if v == target_id][0]
        print(f"Target: {label_name} (id={target_id})")
    elif args.skip_mask:
        print(f"Mode: Whole image (no mask)")
    print()

    # Step 1: Extract style source (face, semantic mask, or whole image)
    style_image = image_path

    if detected_face is not None:
        # Face detection mode - extract face region
        style_image = work_dir / "style_crop.jpg"
        if not style_image.exists() or args.force:
            print(f"[1/3] Extracting face region as style image...")
            success = extract_face_region(
                image_path, detected_face['bbox'], style_image,
                padding_pct=args.face_padding
            )
            if not success:
                print("[error] Failed to extract face region")
                return 1
        else:
            print("  [skip] Face crop already exists")

        # Optional: Apply PyTorch neural style to the cropped face
        if args.pytorch_style:
            pytorch_styled_path = work_dir / "style_crop_pytorch.jpg"
            if not pytorch_styled_path.exists() or args.force:
                success, model_used = run_pytorch_style(
                    style_image, pytorch_styled_path,
                    model_name=args.pytorch_model,
                    blend=0.95
                )
                if success:
                    print(f"[1/3] PyTorch styled with '{model_used}' model")
                else:
                    print("[1/3] PyTorch styling failed, will skip blend variants")
            else:
                print(f"  [skip] PyTorch styled crop already exists")

    elif not args.skip_mask:
        # Semantic segmentation mode
        mask_path = work_dir / "mask.png"
        print(f"[1/3] Generating semantic mask for class {target_id}...")

        if not mask_path.exists() or args.force:
            success = run_deeplab_segmentation(
                image_path, mask_path, str(target_id),
                args.weights, resolution=1024
            )
            if not success:
                print("[error] Failed to generate mask")
                return 1
        else:
            print("  [skip] Mask already exists")

        # Extract masked region as style image
        style_image = work_dir / "style_crop.jpg"
        if not style_image.exists() or args.force:
            print("[1/3] Extracting masked region as style image...")
            extract_masked_region(image_path, mask_path, style_image)
        else:
            print("  [skip] Style crop already exists")

        # Optional: Apply PyTorch neural style to the cropped region
        if args.pytorch_style:
            pytorch_styled_path = work_dir / "style_crop_pytorch.jpg"
            if not pytorch_styled_path.exists() or args.force:
                success, model_used = run_pytorch_style(
                    style_image, pytorch_styled_path,
                    model_name=args.pytorch_model,
                    blend=0.95
                )
                if success:
                    print(f"[1/3] PyTorch styled with '{model_used}' model")
                else:
                    print("[1/3] PyTorch styling failed, will skip blend variants")
            else:
                print(f"  [skip] PyTorch styled crop already exists")

    # Copy the original style source to output for reference
    style_source_output = base_output / "style_source.jpg"
    if not style_source_output.exists() or args.force:
        shutil.copy(str(style_image), str(style_source_output))
        print(f"[info] Style source saved to: {style_source_output}")

    # Step 2: Run Magenta style transfer
    # If pytorch_style is enabled, create blend variants and run each through Magenta
    # Otherwise, just run the standard tile configs with original crop as style

    original_crop = work_dir / "style_crop.jpg"
    pytorch_crop = work_dir / "style_crop_pytorch.jpg"

    if args.pytorch_style and pytorch_crop.exists() and original_crop.exists():
        # Create blend style source images and run each through style transfer with ALL tile configs
        blend_ratios = [0, 25, 50, 75, 100]  # 0% = pure original, 100% = pure pytorch
        total_blend_outputs = len(blend_ratios) * len(TILE_CONFIGS)

        print(f"\n[2/3] Running Magenta style transfer with {len(blend_ratios)} blend variants × {len(TILE_CONFIGS)} tiles = {total_blend_outputs} outputs...")

        # Load original and pytorch crops for blending
        orig_img = cv2.imread(str(original_crop))
        pytorch_img = cv2.imread(str(pytorch_crop))

        if orig_img is not None and pytorch_img is not None:
            # Resize pytorch to match original if needed
            if pytorch_img.shape[:2] != orig_img.shape[:2]:
                pytorch_img = cv2.resize(pytorch_img, (orig_img.shape[1], orig_img.shape[0]))

            # Create blend variants and run each through ALL tile configs
            for ratio in blend_ratios:
                blend_factor = ratio / 100.0
                blend_style_path = work_dir / f"style_crop_blend{ratio}.jpg"

                # Create blended style source
                if not blend_style_path.exists() or args.force:
                    blended = cv2.addWeighted(orig_img, 1.0 - blend_factor, pytorch_img, blend_factor, 0)
                    cv2.imwrite(str(blend_style_path), blended)
                    print(f"  Created blend style source: {ratio}% pytorch")

                # Run style transfer with ALL tile configs for this blend
                print(f"  Processing blend {ratio}% ({len(TILE_CONFIGS)} tiles)...")
                for tile, overlap in TILE_CONFIGS:
                    output_name = f"{name}_blend{ratio}_tile{tile}_overlap{overlap}.jpg"
                    output_path = styled_dir / output_name

                    if output_path.exists() and not args.force:
                        print(f"    [skip] {output_name}")
                        continue

                    success = run_magenta_style(
                        image_path, blend_style_path, output_path,
                        tile, overlap,
                        scale=args.scale,
                        blend=args.blend
                    )

                    if success:
                        print(f"    -> {output_name}")
                    else:
                        print(f"    FAILED: {output_name}")

    # Run Magenta style transfer
    print(f"\n[2/3] Running standard Magenta style transfer ({len(TILE_CONFIGS)} tile configs)...")

    for tile, overlap in TILE_CONFIGS:
        output_name = f"{name}_tile{tile}_overlap{overlap}.jpg"
        output_path = styled_dir / output_name

        if output_path.exists() and not args.force:
            print(f"  [skip] {output_name}")
            continue

        success = run_magenta_style(
            image_path, style_image, output_path,
            tile, overlap,
            scale=args.scale,
            blend=args.blend
        )

        if success:
            print(f"    -> {output_name}")
        else:
            print(f"    FAILED: {output_name}")

    # Step 3: Generate optical flow morph video
    # Build video filename with pan/zoom info if enabled
    if args.pan_zoom and args.pan_zoom > 1.0:
        zoom_str = f"{args.pan_zoom:.1f}".replace('.', 'p')
        video_path = base_output / f"{name}_morph_{args.pan_direction}_{zoom_str}x.mp4"
    else:
        video_path = base_output / f"{name}_morph.mp4"

    if not args.skip_video:
        print(f"\n[3/3] Creating optical flow morph video...")

        target_size = (720, 1280) if args.vertical else (1280, 720)

        if video_path.exists() and not args.force:
            print(f"  [skip] Video already exists")
        else:
            create_morph_video(
                styled_dir, video_path,
                fps=args.fps,
                morph_seconds=args.morph_seconds,
                target_size=target_size,
                zoom=args.zoom,
                hold_frames=args.hold_frames,
                pan_zoom=args.pan_zoom,
                pan_direction=args.pan_direction,
                easing=args.easing,
                temporal_smooth=args.temporal_smooth,
                zoom_pulse=args.zoom_pulse,
                zoom_pulse_freq=args.zoom_pulse_freq,
                hue_rotate=args.hue_rotate,
                zoom_in_pct=args.zoom_in_pct
            )

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Styled images: {styled_dir}")
    if not args.skip_video:
        print(f"Video: {video_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
