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


def optical_flow_morph(img1, img2, num_interp_frames=72):
    """Generate interpolated frames between two images using optical flow."""
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow_forward = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    flow_backward = cv2.calcOpticalFlowFarneback(
        gray2, gray1, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    frames = []
    for i in range(num_interp_frames):
        t = i / (num_interp_frames - 1) if num_interp_frames > 1 else 0

        map1_x = x_coords + t * flow_forward[:, :, 0]
        map1_y = y_coords + t * flow_forward[:, :, 1]
        warped1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        map2_x = x_coords + (1 - t) * flow_backward[:, :, 0]
        map2_y = y_coords + (1 - t) * flow_backward[:, :, 1]
        warped2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        alpha = t
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
                       target_size=(720, 1280), zoom=1.5, hold_frames=24):
    """Create optical flow morph video from styled images.

    Args:
        styled_dir: Directory containing styled images (including blend variants)
        output_path: Output video path
        fps: Frames per second
        morph_seconds: Duration of each morph transition in seconds
        target_size: Video dimensions (width, height)
        zoom: Zoom factor for images
        hold_frames: Number of frames to hold on final image
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

    def get_label(idx):
        """Get label for image at index."""
        if idx < len(image_labels):
            return image_labels[idx]
        return f"image_{idx}"

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
                    morph_frames = optical_flow_morph(curr_img, next_img, interp_frames)
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
    parser.add_argument('--scale', type=int, default=1440, help='Output resolution for styled images')
    parser.add_argument('--blend', type=float, default=0.95, help='Style blend ratio')
    parser.add_argument('--fps', type=int, default=24, help='Video framerate')
    parser.add_argument('--morph_seconds', type=float, default=2.0, help='Seconds per morph transition')
    parser.add_argument('--hold_frames', type=int, default=24, help='Frames to hold on final image')
    parser.add_argument('--num_blend_frames', type=int, default=5,
                        help='Number of blend levels between original and pytorch (default 5)')
    parser.add_argument('--zoom', type=float, default=1.5, help='Video zoom factor')
    parser.add_argument('--vertical', action='store_true', help='Vertical video (720x1280)')
    parser.add_argument('--skip_mask', action='store_true', help='Skip mask step, use whole image as style')
    parser.add_argument('--skip_video', action='store_true', help='Skip video generation')
    parser.add_argument('--force', action='store_true', help='Force regenerate existing outputs')
    parser.add_argument('--list_labels', action='store_true', help='List available semantic labels')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze image and show all detected regions (no processing)')
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
        regions = analyze_all_masks(str(image_path), args.weights, resolution=512)
        if not regions:
            print("[analyze] No semantic regions detected in image")
            return 0

        print(f"\n[analyze] Detected {len(regions)} regions:\n")
        print(f"{'Rank':<5} {'Label':<15} {'Coverage':<10} {'Score':<8} {'Center':<12}")
        print("-" * 55)
        for i, r in enumerate(regions, 1):
            cx, cy = r['center']
            print(f"{i:<5} {r['label']:<15} {r['coverage_pct']:<10.1f} {r['score']:<8.1f} ({cx:.2f}, {cy:.2f})")

        best = select_best_region(regions, args.min_coverage, args.max_coverage)
        if best:
            print(f"\n[analyze] Best region for styling: {best['label']} (score={best['score']:.1f})")
        return 0

    # Determine target class ID
    target_id = None
    selected_label = None

    if args.auto:
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
    base_output = Path(args.output_dir) if args.output_dir else Path("/app/output/morphv2") / name
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
    if target_id is not None:
        label_name = selected_label or [k for k, v in VOC21_LABELS.items() if v == target_id][0]
        print(f"Target: {label_name} (id={target_id})")
    elif args.skip_mask:
        print(f"Mode: Whole image (no mask)")
    print()

    # Step 1: Generate mask (if not skipped)
    style_image = image_path
    if not args.skip_mask:
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
        # Create blend style source images and run each through Magenta with ALL tile configs
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

                # Run Magenta with ALL tile configs for this blend
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

    # Always run standard Magenta tile configs with original crop as style
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
    if not args.skip_video:
        print(f"\n[3/3] Creating optical flow morph video...")

        target_size = (720, 1280) if args.vertical else (1280, 720)
        video_path = base_output / f"{name}_morph.mp4"

        if video_path.exists() and not args.force:
            print(f"  [skip] Video already exists")
        else:
            create_morph_video(
                styled_dir, video_path,
                fps=args.fps,
                morph_seconds=args.morph_seconds,
                target_size=target_size,
                zoom=args.zoom,
                hold_frames=args.hold_frames
            )

    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Styled images: {styled_dir}")
    if not args.skip_video:
        print(f"Video: {base_output / f'{name}_morph.mp4'}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
