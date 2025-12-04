#!/usr/bin/env python3
"""
Optical Flow Slideshow - Magenta Self-Style
Creates a video from self-styled images with optical flow morphing.
"""

import cv2
import numpy as np
import os
import glob
import random

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


def create_selfstyle_slideshow(image_dir, output_path,
                               hold_frames=0, interp_frames=48,
                               fps=24, target_size=(1280, 720),
                               seed=42):
    """Create slideshow from self-styled images."""

    # Collect selfstyle images
    all_images = glob.glob(os.path.join(image_dir, 'selfstyle_*.jpg'))
    all_images.extend(glob.glob(os.path.join(image_dir, 'selfstyle_*.png')))

    if len(all_images) < 2:
        print(f"Need at least 2 selfstyle images, found {len(all_images)}")
        return

    # Shuffle for variety
    random.seed(seed)
    random.shuffle(all_images)

    print(f"Processing {len(all_images)} self-styled images ({target_size[0]}x{target_size[1]})...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    def load_and_resize(path):
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        target_w, target_h = target_size
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
        curr_img = load_and_resize(img_path)

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
            next_img = load_and_resize(all_images[idx + 1])
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


if __name__ == '__main__':
    image_dir = '/Users/trentmahaffey/Dev/NeuralStyleWeb/static/magenta_self_style'
    output_path = '/Users/trentmahaffey/Dev/NeuralStyleTransferV1/output/selfstyle_optflow.mp4'

    create_selfstyle_slideshow(
        image_dir=image_dir,
        output_path=output_path,
        hold_frames=0,      # No hold - continuous morphing
        interp_frames=48,   # 2s transition at 24fps
        fps=24,
        target_size=(1280, 720),  # Horizontal 16:9
        seed=42
    )
