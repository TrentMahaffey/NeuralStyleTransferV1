#!/usr/bin/env python3
"""
Generate selfstyle dual-blob morph videos from a single input image.

Pipeline:
1. Run Magenta self-style transfer at multiple tile/overlap configurations
2. Optionally apply PyTorch style transfer on top
3. Create dual-blob morph video with smooth crossfading between styles

Usage:
    python selfstyle_blob.py /path/to/image.jpg
    python selfstyle_blob.py /path/to/image.jpg --blend_original 0.5 --duration 30
    python selfstyle_blob.py /path/to/image.jpg --pytorch_model udnie --num_styles 50
"""
import sys
from pathlib import Path
import argparse
import subprocess
import cv2
import numpy as np


# PyTorch models available
PYTORCH_MODELS = {
    'candy': '/app/models/pytorch/candy.pth',
    'mosaic': '/app/models/pytorch/mosaic.pth',
    'rain_princess': '/app/models/pytorch/rain_princess.pth',
    'udnie': '/app/models/pytorch/udnie.pth',
}


def generate_tile_configs(min_tile=256, max_tile=512, num_styles=24):
    """Generate evenly distributed tile/overlap configurations.

    Args:
        min_tile: Minimum tile size
        max_tile: Maximum tile size
        num_styles: Approximate number of configurations to generate

    Returns:
        List of (tile, overlap) tuples
    """
    configs = []

    # Calculate step size based on desired number
    tile_range = max_tile - min_tile

    # Generate tile sizes with varying overlaps
    # More tiles = finer granularity
    num_tile_sizes = max(4, num_styles // 3)
    tile_step = max(16, tile_range // num_tile_sizes)

    current_tile = min_tile
    while current_tile <= max_tile and len(configs) < num_styles:
        # Generate 2-3 overlaps per tile size
        # Overlap typically ranges from tile/8 to tile/4
        min_overlap = max(16, current_tile // 8)
        max_overlap = current_tile // 3

        overlaps = [min_overlap]
        if max_overlap > min_overlap + 8:
            mid_overlap = (min_overlap + max_overlap) // 2
            # Round to nearest 8
            mid_overlap = (mid_overlap // 8) * 8
            if mid_overlap != min_overlap:
                overlaps.append(mid_overlap)
        if max_overlap > min_overlap:
            overlaps.append(max_overlap)

        for overlap in overlaps:
            if len(configs) < num_styles:
                configs.append((current_tile, overlap))

        current_tile += tile_step

    return configs


def run_magenta_selfstyle(input_image, output_dir, tile_configs, scale=1440, blend=0.95):
    """Run Magenta self-style transfer at multiple tile/overlap configs.

    Args:
        input_image: Path to input image (used as both content and style)
        output_dir: Directory to save styled images
        tile_configs: List of (tile, overlap) tuples
        scale: Output resolution (longest edge)
        blend: Style blend ratio

    Returns:
        List of generated image paths
    """
    input_image = Path(input_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = input_image.stem
    generated = []

    print(f"\nGenerating {len(tile_configs)} Magenta self-style variations...")
    print(f"  Input: {input_image}")
    print(f"  Output dir: {output_dir}")
    print(f"  Tile range: {tile_configs[0][0]} - {tile_configs[-1][0]}")

    for i, (tile, overlap) in enumerate(tile_configs):
        output_file = output_dir / f"{image_name}_tile{tile}_overlap{overlap}.jpg"

        if output_file.exists():
            print(f"  [{i+1}/{len(tile_configs)}] Skipping tile{tile}_overlap{overlap} (exists)")
            generated.append(output_file)
            continue

        print(f"  [{i+1}/{len(tile_configs)}] Processing tile{tile} overlap{overlap}...")

        cmd = [
            'python3', '/app/pipeline.py',
            '--input_image', str(input_image),
            '--output_image', str(output_file),
            '--model', 'magenta',
            '--model_type', 'magenta',
            '--magenta_style', str(input_image),  # Self-style
            '--magenta_tile', str(tile),
            '--magenta_overlap', str(overlap),
            '--scale', str(scale),
            '--blend', str(blend)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if output_file.exists():
            generated.append(output_file)
        else:
            print(f"    WARNING: Failed to generate {output_file.name}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")

    print(f"\nGenerated {len(generated)} Magenta styled images")
    return generated


def apply_pytorch_style(styled_images, output_dir, pytorch_model, scale=1440, blend=0.95):
    """Apply PyTorch style transfer to Magenta-styled images.

    Args:
        styled_images: List of paths to Magenta-styled images
        output_dir: Directory to save PyTorch-styled images
        pytorch_model: Name of PyTorch model (candy, mosaic, rain_princess, udnie)
        scale: Output resolution
        blend: Style blend ratio

    Returns:
        List of generated image paths
    """
    if pytorch_model not in PYTORCH_MODELS:
        print(f"Unknown PyTorch model: {pytorch_model}")
        print(f"Available: {list(PYTORCH_MODELS.keys())}")
        return styled_images

    model_path = PYTORCH_MODELS[pytorch_model]
    output_dir = Path(output_dir)
    pytorch_dir = output_dir / f"pytorch_{pytorch_model}"
    pytorch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nApplying PyTorch {pytorch_model} style to {len(styled_images)} images...")

    generated = []
    for i, img_path in enumerate(styled_images):
        img_path = Path(img_path)
        output_file = pytorch_dir / f"{img_path.stem}_{pytorch_model}.jpg"

        if output_file.exists():
            print(f"  [{i+1}/{len(styled_images)}] Skipping {img_path.stem} (exists)")
            generated.append(output_file)
            continue

        print(f"  [{i+1}/{len(styled_images)}] Styling {img_path.stem}...")

        cmd = [
            'python3', '/app/pipeline.py',
            '--input_image', str(img_path),
            '--output_image', str(output_file),
            '--model', model_path,
            '--model_type', 'transformer',
            '--io_preset', 'raw_255',
            '--scale', str(scale),
            '--blend', str(blend)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if output_file.exists():
            generated.append(output_file)
        else:
            print(f"    WARNING: Failed to generate {output_file.name}")

    print(f"\nGenerated {len(generated)} PyTorch-styled images")
    return generated


def create_multi_blob_mask(H, W, frame_idx, num_blobs=2, frequency=3.0, speed=1.0, seed=42):
    """Create animated multi-blob mask with organic pockets.

    Returns a mask with values 0 to num_blobs-1 indicating which blob each pixel belongs to.
    """
    time_offset = frame_idx * speed * 0.02  # Slower base animation

    y_norm = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    x_norm = np.linspace(0, 1, W, dtype=np.float32)[None, :]

    # Generate noise field for each blob
    blob_values = np.zeros((num_blobs, H, W), dtype=np.float32)

    for blob_idx in range(num_blobs):
        np.random.seed(seed + blob_idx * 1000)
        noise = np.zeros((H, W), dtype=np.float32)

        # Each blob has different phase offsets
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

    # Assign each pixel to the blob with highest value (creates distinct regions)
    blob_mask = np.argmax(blob_values, axis=0).astype(np.int32)

    return blob_mask


def create_soft_multi_blob_masks(H, W, frame_idx, num_blobs=2, frequency=3.0, speed=1.0, seed=42, feather=0.15):
    """Create soft multi-blob masks with smooth transitions between regions.

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

    # Softmax-like weighting with temperature control for feathering
    temperature = max(0.1, feather * 5)  # Higher = softer boundaries
    blob_values = blob_values - blob_values.max(axis=0, keepdims=True)  # Numerical stability
    exp_values = np.exp(blob_values / temperature)
    weights = exp_values / (exp_values.sum(axis=0, keepdims=True) + 1e-6)

    return weights.astype(np.float32)


def get_blended_image(loaded, position):
    """Get smoothly interpolated image at fractional position."""
    num_images = len(loaded)
    pos = position % num_images

    idx1 = int(pos)
    idx2 = (idx1 + 1) % num_images
    blend = pos - idx1

    img1 = loaded[idx1]
    img2 = loaded[idx2]

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return cv2.addWeighted(img1, 1 - blend, img2, blend, 0)


def generate_blob_video(
    styled_images,
    original_image,
    output_path,
    fps=24,
    duration=30,
    blend_original=0.5,
    blob_frequency=2.5,
    blob_speed=1.0,
    max_resolution=1280,
    seed=42,
    num_blobs=2,
    magenta_images=None,
    transition_speed=1.0
):
    """Generate multi-blob morph video from styled images.

    Args:
        styled_images: List of paths to styled images (e.g., PyTorch-styled)
        original_image: Path to original image for blending
        output_path: Output video path
        fps: Frames per second
        duration: Video duration in seconds
        blend_original: Blend ratio with original (0.0=full style, 1.0=full original)
        blob_frequency: Blob detail level (2-4 recommended)
        blob_speed: Animation speed (1.0=normal)
        max_resolution: Maximum output dimension
        seed: Random seed for blob pattern
        num_blobs: Number of blob regions (2-8)
        magenta_images: Optional list of Magenta-only images (no PyTorch) for mixing
        transition_speed: How fast to transition through images (0.5=slower, 1.0=normal, 2.0=faster)

    Returns:
        Path to output video
    """
    # Sort by tile size, then overlap
    def sort_key(p):
        name = Path(p).stem
        tile = overlap = 0
        for part in name.split('_'):
            if part.startswith('tile'):
                tile = int(part.replace('tile', ''))
            elif part.startswith('overlap'):
                overlap = int(part.replace('overlap', ''))
        return (tile, overlap)

    styled_images = sorted(styled_images, key=sort_key)

    # Load original for blending
    original = cv2.imread(str(original_image))
    if original is None:
        print(f"Failed to load original: {original_image}")
        return None

    def load_and_blend(image_paths):
        """Load images and blend with original."""
        loaded = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                if blend_original > 0:
                    if img.shape != original.shape:
                        orig_resized = cv2.resize(original, (img.shape[1], img.shape[0]))
                    else:
                        orig_resized = original
                    img = cv2.addWeighted(img, 1 - blend_original, orig_resized, blend_original, 0)
                loaded.append(img)
        return loaded

    # Load styled images
    loaded_styled = load_and_blend(styled_images)

    if not loaded_styled:
        print("Failed to load any styled images")
        return None

    # Load magenta-only images if provided (for mixing)
    loaded_magenta = None
    if magenta_images:
        magenta_images = sorted(magenta_images, key=sort_key)
        loaded_magenta = load_and_blend(magenta_images)

    print(f"\nGenerating multi-blob video...")
    print(f"  {len(loaded_styled)} styled images (blended {int(blend_original*100)}% original)")
    if loaded_magenta:
        print(f"  {len(loaded_magenta)} magenta-only images (for mixed blobs)")
    print(f"  {num_blobs} blobs, transition speed {transition_speed}x")
    print(f"  Duration: {duration}s at {fps}fps")

    # Video settings
    total_frames = fps * duration
    H, W = loaded_styled[0].shape[:2]
    num_styled = len(loaded_styled)
    num_magenta = len(loaded_magenta) if loaded_magenta else 0

    # Build blob streams - each blob has its own image stream
    # If we have magenta images, alternate between styled and magenta
    blob_streams = []
    for blob_idx in range(num_blobs):
        if loaded_magenta and blob_idx % 2 == 1:
            # Odd blobs use magenta-only
            blob_streams.append(('magenta', loaded_magenta))
        else:
            # Even blobs use styled (udnie etc)
            blob_streams.append(('styled', loaded_styled))

    all_frames = []

    for frame_idx in range(total_frames):
        t = frame_idx / total_frames

        # Get soft masks for all blobs
        masks = create_soft_multi_blob_masks(
            H, W, frame_idx, num_blobs, blob_frequency, blob_speed, seed
        )

        # Start with black frame
        frame = np.zeros((H, W, 3), dtype=np.float32)

        # Blend each blob stream
        for blob_idx in range(num_blobs):
            stream_type, stream_images = blob_streams[blob_idx]
            num_images = len(stream_images)

            # Each blob moves through images at different phase offset
            # Slower transitions with transition_speed
            phase_offset = blob_idx / num_blobs
            pos = (t * transition_speed * (num_images - 1) + phase_offset * num_images) % num_images

            img = get_blended_image(stream_images, pos)

            # Resize if needed
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))

            # Apply mask
            mask_3ch = masks[blob_idx][:, :, np.newaxis]
            frame += img.astype(np.float32) * mask_3ch

        all_frames.append(frame.astype(np.uint8))

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    # Output resolution
    if max(H, W) > max_resolution:
        scale = max_resolution / max(H, W)
        new_w = int(W * scale) - (int(W * scale) % 2)
        new_h = int(H * scale) - (int(H * scale) % 2)
    else:
        new_w, new_h = W - (W % 2), H - (H % 2)

    output_path = Path(output_path)
    temp_path = output_path.parent / "temp_blob.mp4"

    print(f"  Writing video {new_w}x{new_h}...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (new_w, new_h))

    for frame in all_frames:
        resized = cv2.resize(frame, (new_w, new_h))
        writer.write(resized)
    writer.release()

    subprocess.run([
        'ffmpeg', '-y', '-i', str(temp_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ], capture_output=True)
    temp_path.unlink(missing_ok=True)

    if output_path.exists():
        print(f"\nOutput: {output_path}")
        print(f"Duration: {duration}s ({len(all_frames)} frames)")
        return output_path
    else:
        print("Failed to create video")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate selfstyle multi-blob morph video from a single image"
    )
    parser.add_argument("input_image", type=str,
                        help="Input image (used as both content and style)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: /app/output/<image_name>_selfstyle)")
    parser.add_argument("--scale", type=int, default=1440,
                        help="Styled image resolution (default: 1440)")
    parser.add_argument("--style_blend", type=float, default=0.95,
                        help="Style transfer blend (default: 0.95)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Video FPS (default: 24)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Video duration in seconds (default: 30)")
    parser.add_argument("--blend_original", type=float, default=0.5,
                        help="Blend styled with original in video: 0=full style, 1=full original (default: 0.5)")
    parser.add_argument("--blob_frequency", type=float, default=2.5,
                        help="Blob detail level (default: 2.5)")
    parser.add_argument("--blob_speed", type=float, default=1.0,
                        help="Blob animation speed (default: 1.0)")
    parser.add_argument("--max_resolution", type=int, default=1280,
                        help="Maximum video dimension (default: 1280)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for blob pattern (default: 42)")
    parser.add_argument("--skip_styling", action="store_true",
                        help="Skip Magenta styling, use existing images")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate existing styled images")
    # New options for tile range and PyTorch
    parser.add_argument("--num_styles", type=int, default=24,
                        help="Number of Magenta tile variations to generate (default: 24)")
    parser.add_argument("--min_tile", type=int, default=256,
                        help="Minimum tile size (default: 256)")
    parser.add_argument("--max_tile", type=int, default=512,
                        help="Maximum tile size (default: 512)")
    parser.add_argument("--pytorch_model", type=str, default=None,
                        choices=['candy', 'mosaic', 'rain_princess', 'udnie'],
                        help="Apply PyTorch style on top of Magenta (optional)")
    parser.add_argument("--pytorch_blend", type=float, default=0.9,
                        help="PyTorch style blend ratio (default: 0.9)")
    # Multi-blob options
    parser.add_argument("--num_blobs", type=int, default=2,
                        help="Number of blob regions (default: 2, max 8)")
    parser.add_argument("--transition_speed", type=float, default=1.0,
                        help="Transition speed through images (0.5=slower, 1.0=normal, 2.0=faster)")
    parser.add_argument("--mix_styles", action="store_true",
                        help="Mix PyTorch-styled and Magenta-only in different blobs")

    args = parser.parse_args()

    input_image = Path(args.input_image)
    if not input_image.exists():
        print(f"Input image not found: {input_image}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "_selfstyle"
        if args.pytorch_model:
            suffix += f"_{args.pytorch_model}"
        output_dir = Path("/app/output") / f"{input_image.stem}{suffix}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tile configurations
    tile_configs = generate_tile_configs(
        min_tile=args.min_tile,
        max_tile=args.max_tile,
        num_styles=args.num_styles
    )
    print(f"Tile configurations: {len(tile_configs)} styles")
    print(f"  Range: tile {args.min_tile} -> {args.max_tile}")

    # Step 1: Generate Magenta self-styled images
    if args.skip_styling:
        # Look for existing images
        if args.pytorch_model:
            pytorch_dir = output_dir / f"pytorch_{args.pytorch_model}"
            styled_images = sorted(pytorch_dir.glob("*_tile*_overlap*.jpg"))
        else:
            styled_images = sorted(output_dir.glob("*_tile*_overlap*.jpg"))
            # Exclude pytorch subdirs
            styled_images = [p for p in styled_images if 'pytorch_' not in str(p)]

        if not styled_images:
            print(f"No styled images found in {output_dir}")
            sys.exit(1)
        print(f"Using {len(styled_images)} existing styled images")
    else:
        if args.force:
            # Remove existing styled images
            for f in output_dir.glob("*_tile*_overlap*.jpg"):
                f.unlink()

        styled_images = run_magenta_selfstyle(
            input_image, output_dir, tile_configs,
            scale=args.scale, blend=args.style_blend
        )

        # Step 2: Optionally apply PyTorch style
        if args.pytorch_model and styled_images:
            styled_images = apply_pytorch_style(
                styled_images, output_dir, args.pytorch_model,
                scale=args.scale, blend=args.pytorch_blend
            )

    if not styled_images:
        print("No styled images generated")
        sys.exit(1)

    # Keep track of magenta-only images for mixing
    magenta_only_images = None
    if args.mix_styles and args.pytorch_model:
        # Get original Magenta images (before PyTorch styling)
        magenta_only_images = sorted(output_dir.glob("*_tile*_overlap*.jpg"))
        # Exclude pytorch subdirs
        magenta_only_images = [p for p in magenta_only_images if 'pytorch_' not in str(p)]
        if magenta_only_images:
            print(f"Found {len(magenta_only_images)} Magenta-only images for mixing")

    # Step 3: Generate blob morph video
    video_suffix = ""
    if args.pytorch_model:
        video_suffix = f"_{args.pytorch_model}"
    if args.num_blobs > 2:
        video_suffix += f"_{args.num_blobs}blob"
    if args.mix_styles:
        video_suffix += "_mixed"
    video_path = output_dir / f"{input_image.stem}_selfstyle{video_suffix}_blob.mp4"

    generate_blob_video(
        styled_images=styled_images,
        original_image=input_image,
        output_path=video_path,
        fps=args.fps,
        duration=args.duration,
        blend_original=args.blend_original,
        blob_frequency=args.blob_frequency,
        blob_speed=args.blob_speed,
        max_resolution=args.max_resolution,
        seed=args.seed,
        num_blobs=args.num_blobs,
        magenta_images=magenta_only_images,
        transition_speed=args.transition_speed
    )


if __name__ == "__main__":
    main()
