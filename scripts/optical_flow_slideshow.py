#!/usr/bin/env python3
"""
Optical Flow Slideshow Generator

Creates smooth morphing transitions between images using optical flow
to vector-interpolate pixel movements between frames.
"""

import cv2
import numpy as np
from PIL import Image
import os
import glob
import sys
from pathlib import Path

def optical_flow_morph(img1, img2, num_interp_frames=12):
    """
    Generate interpolated frames between two images using optical flow.
    Uses bidirectional flow for smoother warping.
    """
    # Ensure same size
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))

    # Convert to grayscale for flow calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate bidirectional optical flow (Farneback method)
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

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    frames = []
    for i in range(num_interp_frames):
        t = i / (num_interp_frames - 1) if num_interp_frames > 1 else 0

        # Warp img1 forward by t
        map1_x = x_coords + t * flow_forward[:, :, 0]
        map1_y = y_coords + t * flow_forward[:, :, 1]
        warped1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # Warp img2 backward by (1-t)
        map2_x = x_coords + (1 - t) * flow_backward[:, :, 0]
        map2_y = y_coords + (1 - t) * flow_backward[:, :, 1]
        warped2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # Blend warped images with crossfade
        alpha = t
        blended = cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)
        frames.append(blended)

    return frames


def create_optical_flow_slideshow(image_dir, output_path,
                                  hold_frames=18,      # frames to show each image
                                  interp_frames=12,    # transition frames
                                  fps=24,
                                  target_size=(1280, 720),
                                  max_images=None):
    """
    Create a video slideshow with optical flow morphing transitions.
    """
    # Collect all images
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))

    image_files.sort()

    if max_images:
        image_files = image_files[:max_images]

    if len(image_files) < 2:
        print("Need at least 2 images")
        return

    print(f"Processing {len(image_files)} images...")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    def load_and_resize(path):
        img = cv2.imread(path)
        if img is None:
            return None
        # Resize to target maintaining aspect ratio, then pad/crop
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Calculate scale to fill target
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop to target size
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return img

    # Load first image
    prev_img = load_and_resize(image_files[0])
    if prev_img is None:
        print(f"Failed to load: {image_files[0]}")
        return

    total_frames = 0

    for idx in range(len(image_files)):
        img_path = image_files[idx]
        curr_img = load_and_resize(img_path)

        if curr_img is None:
            print(f"Skipping: {img_path}")
            continue

        print(f"[{idx+1}/{len(image_files)}] {os.path.basename(img_path)}")

        # Hold frames for current image
        for _ in range(hold_frames):
            out.write(curr_img)
            total_frames += 1

        # Generate transition to next image (if not last)
        if idx < len(image_files) - 1:
            next_img = load_and_resize(image_files[idx + 1])
            if next_img is not None:
                try:
                    morph_frames = optical_flow_morph(curr_img, next_img, interp_frames)
                    for frame in morph_frames:
                        out.write(frame)
                        total_frames += 1
                except Exception as e:
                    print(f"  Morph failed, using crossfade: {e}")
                    # Fallback to simple crossfade
                    for i in range(interp_frames):
                        t = i / (interp_frames - 1)
                        blended = cv2.addWeighted(curr_img, 1 - t, next_img, t, 0)
                        out.write(blended)
                        total_frames += 1

    out.release()

    duration = total_frames / fps
    print(f"\nComplete! Output: {output_path}")
    print(f"Total frames: {total_frames}, Duration: {duration:.1f}s ({duration/60:.1f} min)")

    # Convert to H.264 for better compatibility
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    print(f"\nConverting to H.264...")
    os.system(f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset medium -crf 23 "{h264_path}" 2>/dev/null')

    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
        os.remove(output_path)
        os.rename(h264_path, output_path)
        print(f"Final output: {output_path}")


if __name__ == '__main__':
    image_dir = '/Users/trentmahaffey/Dev/NeuralStyleWeb/static/preset_samples'
    output_path = '/Users/trentmahaffey/Dev/NeuralStyleTransferV1/output/preset_samples_optflow_continuous.mp4'

    # Parameters
    hold_frames = 0     # No hold - continuous morphing
    interp_frames = 120 # 5s transition at 24fps
    fps = 24

    create_optical_flow_slideshow(
        image_dir=image_dir,
        output_path=output_path,
        hold_frames=hold_frames,
        interp_frames=interp_frames,
        fps=fps,
        target_size=(1280, 720)
    )
