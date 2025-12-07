#!/usr/bin/env python3
"""
Optical Flow Slideshow - All Batch Self-Style Results
Creates a video from all 49 styled images, ordered sequentially:
  Image 1: tile128 -> tile160 -> ... -> tile512
  Image 2: tile128 -> tile160 -> ... -> tile512
  ... etc
"""

import cv2
import numpy as np
import os
import glob

def optical_flow_morph(img1, img2, num_interp_frames=12):
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


def create_batch_slideshow(image_dir, output_path,
                           hold_frames=0, interp_frames=72,
                           fps=24, target_size=(1280, 720), zoom=1.0):
    """Create slideshow from all batch-styled images, ordered sequentially by image number."""

    # Build ordered list: img2_tile128, img2_tile160, ..., img2_tile512, img3_tile128, ...
    # Note: img1 and img4 were removed, so only process 2, 3, 5, 6, 7
    tile_sizes = [128, 160, 192, 224, 256, 384, 512]
    overlaps = [16, 20, 24, 28, 32, 48, 64]
    image_nums = [2, 3, 5, 6, 7]  # Remaining images (1 and 4 removed)

    all_images = []
    for img_num in image_nums:
        for tile, overlap in zip(tile_sizes, overlaps):
            filename = f"img{img_num}_tile{tile}_overlap{overlap}.jpg"
            filepath = os.path.join(image_dir, filename)
            if os.path.exists(filepath):
                all_images.append(filepath)

    if len(all_images) < 2:
        print(f"Need at least 2 images, found {len(all_images)}")
        return

    print(f"Processing {len(all_images)} styled images sequentially ({target_size[0]}x{target_size[1]})...")
    print("Order: Image 2 (all tiles) -> Image 3 (all tiles) -> Image 5 (all tiles) -> Image 6 (all tiles) -> Image 7 (all tiles)")
    print()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    def load_and_resize(path, zoom=1.0):
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Apply zoom - crop center portion before scaling
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

    total_frames = 0

    for idx in range(len(all_images)):
        img_path = all_images[idx]
        curr_img = load_and_resize(img_path, zoom=zoom)

        if curr_img is None:
            print(f"Skipping: {img_path}")
            continue

        print(f"[{idx+1}/{len(all_images)}] {os.path.basename(img_path)}")

        # Hold frames
        for _ in range(hold_frames):
            out.write(curr_img)
            total_frames += 1

        # Transition to next
        if idx < len(all_images) - 1:
            next_img = load_and_resize(all_images[idx + 1], zoom=zoom)
            if next_img is not None:
                try:
                    morph_frames = optical_flow_morph(curr_img, next_img, interp_frames)
                    for frame in morph_frames:
                        out.write(frame)
                        total_frames += 1
                except Exception as e:
                    print(f"  Morph failed: {e}")
                    for i in range(interp_frames):
                        t = i / (interp_frames - 1)
                        blended = cv2.addWeighted(curr_img, 1 - t, next_img, t, 0)
                        out.write(blended)
                        total_frames += 1

    out.release()

    duration = total_frames / fps
    print(f"\nComplete! Output: {output_path}")
    print(f"Total frames: {total_frames}, Duration: {duration:.1f}s ({duration/60:.1f} min)")

    # Convert to H.264
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    print(f"\nConverting to H.264...")
    os.system(f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset medium -crf 23 "{h264_path}" 2>/dev/null')

    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
        os.remove(output_path)
        os.rename(h264_path, output_path)
        print(f"Final output: {output_path}")


def create_single_image_slideshow(image_dir, output_path, img_num,
                                   hold_frames=0, interp_frames=72,
                                   fps=24, target_size=(720, 1280), zoom=1.5):
    """Create slideshow for a single image's tile variations."""

    tile_sizes = [128, 160, 192, 224, 256, 384, 512]
    overlaps = [16, 20, 24, 28, 32, 48, 64]

    all_images = []
    for tile, overlap in zip(tile_sizes, overlaps):
        filename = f"img{img_num}_tile{tile}_overlap{overlap}.jpg"
        filepath = os.path.join(image_dir, filename)
        if os.path.exists(filepath):
            all_images.append(filepath)

    if len(all_images) < 2:
        print(f"Need at least 2 images for img{img_num}, found {len(all_images)}")
        return

    print(f"Processing {len(all_images)} styled images for img{img_num} ({target_size[0]}x{target_size[1]}, zoom={zoom})...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    def load_and_resize(path, zoom=1.0):
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        target_w, target_h = target_size

        if zoom > 1.0:
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
            h, w = img.shape[:2]

        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]
        return img

    total_frames = 0

    for idx in range(len(all_images)):
        img_path = all_images[idx]
        curr_img = load_and_resize(img_path, zoom=zoom)

        if curr_img is None:
            print(f"Skipping: {img_path}")
            continue

        print(f"[{idx+1}/{len(all_images)}] {os.path.basename(img_path)}")

        for _ in range(hold_frames):
            out.write(curr_img)
            total_frames += 1

        if idx < len(all_images) - 1:
            next_img = load_and_resize(all_images[idx + 1], zoom=zoom)
            if next_img is not None:
                try:
                    morph_frames = optical_flow_morph(curr_img, next_img, interp_frames)
                    for frame in morph_frames:
                        out.write(frame)
                        total_frames += 1
                except Exception as e:
                    print(f"  Morph failed: {e}")
                    for i in range(interp_frames):
                        t = i / (interp_frames - 1)
                        blended = cv2.addWeighted(curr_img, 1 - t, next_img, t, 0)
                        out.write(blended)
                        total_frames += 1

    out.release()

    duration = total_frames / fps
    print(f"Complete! Output: {output_path}")
    print(f"Total frames: {total_frames}, Duration: {duration:.1f}s")

    # Convert to H.264
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    print(f"Converting to H.264...")
    os.system(f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset medium -crf 23 "{h264_path}" 2>/dev/null')

    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
        os.remove(output_path)
        os.rename(h264_path, output_path)
        print(f"Final output: {output_path}")


if __name__ == '__main__':
    import sys

    image_dir = '/Users/trentmahaffey/Dev/NeuralStyleTransferV1/output/batch_selfstyle'
    output_dir = '/Users/trentmahaffey/Dev/NeuralStyleTransferV1/output'

    # Check command line args for single image mode
    if len(sys.argv) >= 3:
        img_num = int(sys.argv[1])
        zoom = float(sys.argv[2])
        zoom_label = "200" if zoom >= 2.0 else "150"
        output_path = f"{output_dir}/img{img_num}_optflow_vertical_{zoom_label}zoom.mp4"

        create_single_image_slideshow(
            image_dir=image_dir,
            output_path=output_path,
            img_num=img_num,
            hold_frames=0,
            interp_frames=72,
            fps=24,
            target_size=(720, 1280),
            zoom=zoom
        )
    else:
        # Default: all images combined
        output_path = f"{output_dir}/batch_selfstyle_optflow_vertical.mp4"
        create_batch_slideshow(
            image_dir=image_dir,
            output_path=output_path,
            hold_frames=0,
            interp_frames=72,
            fps=24,
            target_size=(720, 1280),
            zoom=1.5
        )
