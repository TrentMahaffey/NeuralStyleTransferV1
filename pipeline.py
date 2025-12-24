#!/usr/bin/env python3
import argparse, subprocess, sys
import os, glob, shutil
from pathlib import Path
from typing import Optional, List
import re
import uuid

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import faulthandler, gc
import time
import resource

# Region-based spatial blending
try:
    from region_blend import (
        blend_by_regions, blend_by_regions_advanced, get_required_scales,
        clear_mask_cache, AVAILABLE_MODES as REGION_MODES,
        AVAILABLE_MORPH_MODES,
        # Optimized crop-based styling
        generate_region_masks, parse_region_configs, prepare_region_crops,
        extract_crop, composite_from_crops, get_models_needed_for_regions,
        get_regions_for_model, compute_crop_coverage, rotate_all_masks, feather_mask,
        RegionCrop,
        # Animated blend weights
        BlendAnimation, compute_animated_weights, parse_blend_animation,
        parse_region_blend_animations,
        # Animated scale (resolution oscillation)
        ScaleAnimation, compute_animated_scale, parse_scale_animation,
        parse_region_scale_animations,
        # Variable region sizes
        parse_region_sizes,
        # Organic morph animation
        MorphAnimation, parse_morph_animation, warp_all_masks_organic
    )
    _HAS_REGION_BLEND = True
except ImportError:
    _HAS_REGION_BLEND = False
    REGION_MODES = []
    AVAILABLE_MORPH_MODES = []
    def clear_mask_cache(): pass

# --- Optional backends: import once, use everywhere ---
try:
    from lib import ReCoNet as _ReCoNet
    _HAS_RECONET = True
except Exception:
    _ReCoNet = None
    _HAS_RECONET = False

def _get_transformer_net():
    """Resolve Johnson-style TransformerNet from any of the known locations."""
    try:
        from models.fast_neural_style.neural_style.transformer_net import TransformerNet
    except Exception:
        try:
            from fast_neural_style.neural_style.transformer_net import TransformerNet
        except Exception:
            from transformer_net import TransformerNet
    return TransformerNet

def _get_transformer_net_nst():
    """Get NST_Train style TransformerNet (different architecture)."""
    from transformer_net_nst import TransformerNet
    return TransformerNet

def _detect_transformer_type(checkpoint_path):
    """Detect which TransformerNet architecture a checkpoint uses based on keys."""
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    keys = set(state_dict.keys())
    # NST_Train uses 'down1.conv.weight', original uses 'conv1.conv2d.weight'
    if any(k.startswith('down1.') for k in keys):
        return 'nst'  # NST_Train architecture
    return 'original'  # Johnson/original architecture

def _load_transformer_model(checkpoint_path, device):
    """Load transformer model with auto-detection of architecture."""
    arch_type = _detect_transformer_type(checkpoint_path)
    if arch_type == 'nst':
        TransformerNet = _get_transformer_net_nst()
        print(f"[model] Detected NST_Train architecture for {checkpoint_path}")
    else:
        TransformerNet = _get_transformer_net()
        print(f"[model] Detected original architecture for {checkpoint_path}")
    model = TransformerNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, arch_type

try:
    import psutil
except Exception:
    psutil = None
# Print a Python traceback if the interpreter dies due to a native crash
try:
    faulthandler.enable()
except Exception:
    pass
# Keep OpenCV DNN single-threaded and quiet to avoid rare deadlocks/noise
try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    # Some builds still default to OpenCL; force it off for DNN stability
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# --- Optional TensorFlow/TF-Hub backend (Magenta tiled NST) ---
_MAGENTA_MODEL = None


_TF_GPU_MEMORY_LIMIT_MB = 32000  # Default 32GB, can be set via set_gpu_memory_limit()

def set_gpu_memory_limit(limit_mb: int):
    """Set GPU memory limit for both TensorFlow and PyTorch."""
    global _TF_GPU_MEMORY_LIMIT_MB
    _TF_GPU_MEMORY_LIMIT_MB = limit_mb

def _try_import_tf():
    global tf, hub
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_hub as hub  # type: ignore
        # NGC TensorFlow container has pre-compiled Blackwell (sm_120) kernels
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Set memory limit for each GPU
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=_TF_GPU_MEMORY_LIMIT_MB)]
                    )
                print(f"[magenta] TensorFlow GPU enabled: {len(gpus)} GPU(s), memory limit: {_TF_GPU_MEMORY_LIMIT_MB}MB")
            else:
                print("[magenta] No GPU detected, using CPU")
        except Exception as e:
            print(f"[magenta] GPU setup warning: {e}")
        return tf, hub
    except Exception as e:
        print(f"[magenta][WARN] TensorFlow not available: {e}")
        return None, None


def _magenta_load_model(model_root: str = "/app/models/magenta"):
    """Load Magenta TF-Hub model from a subdirectory under model_root (once)."""
    global _MAGENTA_MODEL
    if _MAGENTA_MODEL is not None:
        return _MAGENTA_MODEL
    tf, hub = _try_import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow is required for --model_type magenta")
    import os, glob
    subdirs = [d for d in glob.glob(os.path.join(model_root, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No model subdirectory found in {model_root}")
    print(f"[magenta] Loading TF-Hub model from {subdirs[0]}…")
    _MAGENTA_MODEL = hub.load(subdirs[0])
    print("[magenta] Model loaded.")
    return _MAGENTA_MODEL


def _get_image_with_exif_pil(image_path: str):
    from PIL import Image, ExifTags
    img = Image.open(image_path)
    exif = getattr(img, "_getexif", lambda: None)()
    orientation = None
    if exif:
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == "Orientation":
                orientation = value
                break
    if orientation == 3:
        img = img.rotate(180, expand=True)
    elif orientation == 6:
        img = img.rotate(270, expand=True)
    elif orientation == 8:
        img = img.rotate(90, expand=True)
    return img.convert("RGB")


def _magenta_style_pil(content_path: str, style_path: str,
                       tile_size: int = 256, overlap: int = 32,
                       target_resolution: int | None = None,
                       model_root: str = "/app/models/magenta"):
    """Return a PIL RGB image styled by Magenta (tiled), keeping EXIF orientation."""
    tf, hub = _try_import_tf()
    if tf is None:
        raise RuntimeError("TensorFlow not available for magenta mode")

    model = _magenta_load_model(model_root)

    # Load & (optionally) downscale content for target_resolution on long side .
    content_pil = _get_image_with_exif_pil(content_path)
    orig_size = content_pil.size  # true original size after EXIF orientation (before any downscale)
    if target_resolution:
        w, h = content_pil.size
        m = max(w, h)
        if m > target_resolution:
            r = target_resolution / float(m)
            content_pil = content_pil.resize((int(w * r), int(h * r)), Image.Resampling.LANCZOS)

    content = np.array(content_pil).astype(np.float32) / 255.0
    content = content[None, ...]

    # Style image to tile size
    style_pil = _get_image_with_exif_pil(style_path).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    style = np.array(style_pil).astype(np.float32) / 255.0
    style = style[None, ...]

    # Tiling
    H, W = content.shape[1:3]
    stride = tile_size - overlap
    tiles = []
    coords = []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            tile = content[:, y:y + tile_size, x:x + tile_size, :]
            ph = max(0, tile_size - tile.shape[1])
            pw = max(0, tile_size - tile.shape[2])
            if ph or pw:
                tile = np.pad(tile, ((0, 0), (0, ph), (0, pw), (0, 0)), mode="edge")
            tiles.append(tile)
            coords.append((y, x))

    # Process tiles
    styled_tiles = []
    for tile in tiles:
        out = model(tf.constant(tile), tf.constant(style))[0]  # 0..1 float32 NHWC
        styled_tiles.append(out.numpy())

    # Blend & stitch
    out = np.zeros((1, H, W, 3), dtype=np.float32)
    weight = np.zeros((1, H, W, 1), dtype=np.float32)
    # linear feather mask
    mask = np.ones((tile_size, tile_size, 1), dtype=np.float32)
    for i in range(overlap):
        wgt = i / float(overlap)
        mask[i, :, 0] *= wgt
        mask[-1 - i, :, 0] *= wgt
        mask[:, i, 0] *= wgt
        mask[:, -1 - i, 0] *= wgt

    for tile, (y, x) in zip(styled_tiles, coords):
        h = min(tile_size, H - y)
        w = min(tile_size, W - x)
        out[:, y:y + h, x:x + w, :] += tile[:, :h, :w, :] * mask[:h, :w, :]
        weight[:, y:y + h, x:x + w, :] += mask[:h, :w, :]

    out /= np.maximum(weight, 1e-6)
    img = np.clip(out[0] * 255.0, 0, 255).astype(np.uint8)
    pil_out = Image.fromarray(img)

    # Ensure Magenta output exactly matches the ORIGINAL content image size (pre-downscale)
    if pil_out.size != orig_size:
        pil_out = pil_out.resize(orig_size, Image.Resampling.LANCZOS)
    return pil_out


# ---------------------------
# Normalization constants
# ---------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ---------------------------
# Mask helpers
# ---------------------------
def _pct_to_px(pct: float, H: int) -> int:
    try:
        return int(round(max(0.0, float(pct)) * 0.01 * H))
    except Exception:
        return 0

def _load_mask_fit(mask_path: str, target_hw: tuple[int, int], invert: bool, feather_px: int, autofix: bool = True, force_transpose: bool = False) -> np.ndarray:
    """
    Returns float32 HxWx1 alpha in [0,1] matching (H,W) of target.
    If `force_transpose` is True, the source mask is transposed 90° before any resize.
    If `autofix` is True and the source mask appears 90°-transposed relative to the
    target shape, it will be transposed to match before resizing. This helps when the
    mask sequence was generated with swapped width/height.

    Heuristic:
      - If (mw, mh) == (H_tgt, W_tgt) and target is not square -> transpose.
      - Else if aspect ratio of mask is closer to the *swapped* target aspect ratio
        than to the target aspect ratio (within a tolerance), transpose.
    """
    # target dims
    W_tgt, H_tgt = target_hw[1], target_hw[0]

    # Load source as 8-bit L (do not auto-rotate via EXIF; masks shouldn't carry it)
    m_img = Image.open(mask_path).convert('L')
    mw, mh = m_img.size  # (W_src, H_src)

    # Optional hard override
    if force_transpose:
        try:
            print(f"[mask][force] {Path(mask_path).name}: applying Image.TRANSPOSE (pre-resize)")
        except Exception:
            pass
        m_img = m_img.transpose(Image.TRANSPOSE)
        mw, mh = m_img.size

    # Heuristic transpose
    if autofix and (W_tgt != H_tgt):
        transpose_reason = None
        # Case 1: exact swap
        if (mw, mh) == (H_tgt, W_tgt):
            transpose_reason = "exact-dimension swap"
        else:
            # Case 2: aspect ratio check
            try:
                ar_tgt = float(W_tgt) / float(H_tgt)
                ar_mask = float(mw) / float(mh)
                ar_swapped = float(H_tgt) / float(W_tgt)
                # Compare closeness in log space to handle very tall/wide shapes
                def _dist(a, b): return abs(np.log(max(a, 1e-6)) - np.log(max(b, 1e-6)))
                d_norm = _dist(ar_mask, ar_tgt)
                d_swap = _dist(ar_mask, ar_swapped)
                if d_swap + 1e-6 < d_norm:  # clearly closer to swapped orientation
                    transpose_reason = f"aspect-ratio closer to swapped ({ar_mask:.4f} vs tgt {ar_tgt:.4f}, swapped {ar_swapped:.4f})"
            except Exception:
                pass
        if transpose_reason:
            try:
                print(f"[mask][autofix] {Path(mask_path).name}: {transpose_reason}; applying Image.TRANSPOSE")
            except Exception:
                print("[mask][autofix] transposing mask to match target aspect")
            m_img = m_img.transpose(Image.TRANSPOSE)
            mw, mh = m_img.size

    # Resize to target exactly (nearest keeps hard edges; we feather afterwards)
    m_img = m_img.resize((W_tgt, H_tgt), Image.Resampling.NEAREST)

    # To numpy, optional invert, optional feather
    m = np.array(m_img, dtype=np.uint8)
    if invert:
        m = 255 - m

    if feather_px and feather_px > 0:
        # Gaussian blur sigma chosen so that radius≈feather_px; OpenCV uses sigma
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_px * 0.5, sigmaY=feather_px * 0.5)

    return (m.astype(np.float32) / 255.0)[..., None]


# ---------------------------
# Shell helper
# ---------------------------
def sh(cmd: str, check=True):
    """
    Shell helper that optionally quiets ffmpeg output.
    If env FFMPEG_QUIET=1 (default), injects -hide_banner -loglevel warning -nostats
    into ffmpeg commands to reduce console spam.
    """
    cmd_stripped = cmd.lstrip()
    if os.environ.get("FFMPEG_QUIET", "1") == "1":
        if cmd_stripped.startswith("ffmpeg "):
            # insert flags right after the executable token only once
            cmd = cmd_stripped.replace(
                "ffmpeg ",
                "ffmpeg -hide_banner -loglevel warning -nostats ",
                1
            )
    print(f"\n$ {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)


# ---------------------------
# Frame extraction
# ---------------------------
# In pipeline.py, lines 172–189
def extract_frames(input_video: Path, frames_dir: Path,
                   fps: Optional[int], scale: Optional[int],
                   img_ext: str, jpeg_quality: int,
                   canvas_wh: Optional[tuple[int,int]] = None):
    frames_dir.mkdir(parents=True, exist_ok=True)
    vf_parts: List[str] = []
    if canvas_wh:
        cw, ch = canvas_wh
        # Fit inside the canvas then pad to exact WxH
        vf_parts.append(f"scale={cw}:{ch}:flags=lanczos:force_original_aspect_ratio=decrease")
        vf_parts.append(f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black")
    else:
        if scale:
            vf_parts.append(f"scale='if(gte(iw,ih),{scale},-2)':'if(gte(ih,iw),{scale},-2)':flags=lanczos")
    if fps:
        vf_parts.append(f"fps={fps}")

    ext = "png" if img_ext.lower() == "png" else "jpg"
    pattern = frames_dir / f"frame_%04d.{ext}"
    vf = ",".join(vf_parts)

    if vf:
        cmd = f'ffmpeg -y -i "{input_video}" -vf "{vf}" -c:v mjpeg -q:v {jpeg_quality} -pix_fmt yuvj420p "{pattern}"'
    else:
        cmd = f'ffmpeg -y -i "{input_video}" -c:v mjpeg -q:v {jpeg_quality} -pix_fmt yuvj420p "{pattern}"'
    sh(cmd)

    # Verify extracted frames
    frame_files = sorted(frames_dir.glob(f"frame_*.{ext}"))
    for frame in frame_files:
        try:
            with Image.open(frame) as img:
                img.verify()  # Check if the image is valid
        except Exception as e:
            print(f"[error] Invalid frame file {frame}: {e}")
            sys.exit(1)


# ---------------------------
# Flow warp (prev -> curr) .
# ---------------------------
def _warp_with_flow(prev_img01: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    prev_img01: HxWxC float32 in [0,1]
    flow:       HxWx2 float32, flow[y,x] = (dx, dy)
    returns:    HxWxC float32 in [0,1]
    """
    H, W = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_img01, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped


# ---------------------------
# Torch7 OpenCV DNN backend
# ---------------------------
def _torch7_forward_openv(nn, pil_img_rgb: Image.Image) -> Image.Image:
    """Run a Torch7 (.t7) fast style network via OpenCV DNN. Returns a PIL RGB image."""
    import numpy as _np
    import cv2 as _cv2
    # Ensure contiguous uint8 RGB -> BGR
    arr = _np.asarray(pil_img_rgb, dtype=_np.uint8)
    if not arr.flags['C_CONTIGUOUS']:
        arr = _np.ascontiguousarray(arr)
    bgr = _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # Build Caffe-style blob manually to avoid blobFromImage segfaults on some builds
    # 1) to float32
    bgr_f32 = bgr.astype(_np.float32)
    # 2) subtract Caffe BGR mean in-place
    bgr_f32[..., 0] -= 103.939
    bgr_f32[..., 1] -= 116.779
    bgr_f32[..., 2] -= 123.68
    # 3) to NCHW and add batch dim
    blob = _np.transpose(bgr_f32, (2, 0, 1))[None, ...].copy()
    nn.setInput(blob)
    # Forward pass (may crash in native code if OpenCV DNN is unstable)
    out = nn.forward()  # NCHW, BGR with mean subtracted
    out = out.squeeze(0)  # CxHxW
    out = _np.transpose(out, (1, 2, 0))  # HxWxC
    # Add back caffe mean (per-channel)
    out[..., 0] += 103.939
    out[..., 1] += 116.779
    out[..., 2] += 123.68
    out = _np.clip(out, 0, 255).astype(_np.uint8)
    rgb = _cv2.cvtColor(out, _cv2.COLOR_BGR2RGB)
    # Help the allocator free native buffers every iteration
    del blob, out, bgr, arr
    gc.collect()
    return Image.fromarray(rgb)


def _rss_mb() -> float:
    """Return resident set size (MB) using psutil if available, otherwise resource."""
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    try:
        # On Linux ru_maxrss is in kilobytes; on macOS it's bytes.
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return val / (1024 * 1024)
        else:
            return val / 1024.0
    except Exception:
        return -1.0


# ---------------------------
# Blend weight parsing
# ---------------------------
def parse_blend_weights(weights_str: str, num_models: int) -> List[float]:
    """Parse comma-separated weights and validate they sum to 1.0."""
    if not weights_str:
        return [1.0 / num_models] * num_models  # Equal weights
    weights = [float(w) for w in weights_str.split(",")]
    if len(weights) != num_models:
        raise ValueError(f"Expected {num_models} weights, got {len(weights)}")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.6f}")
    return weights


def parse_lab_weights(weights_str: str) -> tuple[float, float]:
    """Parse LAB weights for L and a/b channels."""
    if not weights_str:
        return 0.5, 0.5  # Default to equal contribution
    wL, wab = [float(w) for w in weights_str.split(",")]
    if abs(wL + wab - 1.0) > 1e-6:
        raise ValueError(f"LAB weights must sum to 1.0, got {wL + wab:.6f}")
    return wL, wab


# ---------------------------
# Core styling
# ---------------------------
def style_frames(
        args,
        frames_dir: Path, model_path: Path, output_prefix: str,
        image_ext_out: str, device_str: str, threads: int,
        stride: int, max_frames: Optional[int],
        smooth_lightness: bool, smooth_alpha: float,
        jpeg_quality: int, io_preset: str,
        smooth_chroma: bool = False, chroma_alpha: float = 0.85,
        blend: float = 1.0,
        image_mode: bool = False,
        save_map: dict = None,
):
    # Threads
    if threads is not None and threads > 0:
        torch.set_num_threads(threads)

    # Device + model
    device = torch.device(device_str)
    # Set PyTorch GPU memory limit if using CUDA
    if device_str == "cuda" and torch.cuda.is_available():
        # Convert MB limit to fraction of total GPU memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # in MB
        fraction = min(1.0, _TF_GPU_MEMORY_LIMIT_MB / total_mem)
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"[pytorch] GPU memory limit: {_TF_GPU_MEMORY_LIMIT_MB}MB ({fraction*100:.1f}% of {total_mem:.0f}MB)")
    print(f"[cfg] io_preset={io_preset}")

    def _load_checkpoint_compat(model, ckpt_path: str):
        """Load checkpoints robustly across PyTorch versions and drop legacy InstanceNorm buffers."""
        try:
            state = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[load][compat] weights_only load failed ({e}); retrying with weights_only=False")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        DROP_TOKENS = ("running_mean", "running_var", "num_batches_tracked")
        state = {k: v for k, v in state.items() if not any(tok in k for tok in DROP_TOKENS)}

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f">> load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    # --- choose the net, then call the loader once ---
    use_magenta = (getattr(args, "model_type", "transformer") == "magenta")
    use_torch7 = (getattr(args, "model_type", "transformer") == "torch7")

    # Initialize model A variables (both may be None depending on type)
    model = None
    net_t7 = None

    if use_magenta:
        print("[magenta] Using TensorFlow-Hub backend; style image:", args.magenta_style)
        if not args.magenta_style:
            raise ValueError("--magenta_style is required when --model_type magenta")
    elif use_torch7:
        print(f"[torch7] Using OpenCV DNN (.t7) for {'video' if not image_mode else 'image'}: {model_path}")
        try:
            net_t7 = cv2.dnn.readNetFromTorch(str(model_path))
            try:
                net_t7.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net_t7.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[torch7] backend=OPENCV target=CPU (OpenCL enabled? {} )".format(
                    getattr(cv2.ocl, 'useOpenCL', lambda: False)()))
            except Exception as e:
                print(f"[torch7][WARN] Could not set backend/target: {e}")
        except Exception as e:
            print(f"[torch7][ERROR] Could not load .t7 network from {model_path}: {e}")
            sys.exit(2)
    else:
        if args.model_type == "reconet":
            if not _HAS_RECONET:
                print("[error] ReCoNet backend requested but lib.ReCoNet is not available. Ensure lib.py is on PYTHONPATH.")
                sys.exit(2)
            model = _ReCoNet().to(device)
            model_a_arch = 'reconet'
        else:
            # Auto-detect NST_Train vs original architecture
            model_a_arch = _detect_transformer_type(str(model_path))
            if model_a_arch == 'nst':
                TransformerNet = _get_transformer_net_nst()
                print(f"[model] Detected NST_Train architecture for {model_path}")
                # Auto-switch to raw_01 for NST models if io_preset is auto/raw_255
                if io_preset in ('auto', 'raw_255', 'imagenet_255'):
                    print(f"[model] Auto-switching io_preset from '{io_preset}' to 'raw_01' for NST_Train model")
                    io_preset = 'raw_01'
                    args.io_preset = 'raw_01'
            else:
                TransformerNet = _get_transformer_net()
            model = TransformerNet().to(device)
        _load_checkpoint_compat(model, str(model_path))
        model.eval()

    print(f"[backend] A: type={args.model_type} path={model_path if model_path else '(n/a)'}  device={device}  arch={model_a_arch if 'model_a_arch' in dir() else 'n/a'}")

    # --- Model B ---
    model_b = None
    net_t7_b = None
    use_b = bool(getattr(args, "model_b", None))
    b_type_resolved = None
    ipb_resolved = None
    if use_b:
        model_b_path = Path(args.model_b).resolve()
        auto_b_type = "torch7" if model_b_path.suffix.lower() == ".t7" else args.model_type
        model_b_type = getattr(args, "model_b_type", None) or auto_b_type
        io_preset_b = getattr(args, "io_preset_b", None) or io_preset
        b_type_resolved = model_b_type
        ipb_resolved = io_preset_b
        print(f"[backend] B: type={b_type_resolved} path={model_b_path}  device={device}  io_preset_b={ipb_resolved}")

        if model_b_type == "magenta":
            pass
        elif model_b_type == "torch7":
            try:
                net_t7_b = cv2.dnn.readNetFromTorch(str(model_b_path))
                try:
                    net_t7_b.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_b.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][B][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][B][ERROR] Could not load .t7 network from {model_b_path}: {e}")
                sys.exit(2)
        else:
            if model_b_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_b_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_b = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_b = TransformerNet().to(device)
            _load_checkpoint_compat(model_b, str(model_b_path))
            model_b.eval()

    # --- Model C ---
    model_c = None
    net_t7_c = None
    # Treat C as "in use" if a model path, type, or magenta style is provided
    use_c = bool(
        getattr(args, "model_c", None)
        or getattr(args, "model_c_type", None)
        or getattr(args, "magenta_style_c", None)
    )
    c_type_resolved = None
    ipc_resolved = None
    if use_c:
        model_c_path_raw = getattr(args, "model_c", None)
        model_c_type = getattr(args, "model_c_type", None)

        # Infer type if not explicit
        if model_c_type is None:
            if model_c_path_raw and Path(model_c_path_raw).suffix.lower() == ".t7":
                model_c_type = "torch7"
            elif model_c_path_raw:
                model_c_type = args.model_type
            else:
                # No path provided, assume magenta if magenta_style_c exists
                model_c_type = "magenta"

        model_c_path = Path(model_c_path_raw).resolve() if model_c_path_raw else None
        io_preset_c = getattr(args, "io_preset_c", None) or io_preset
        c_type_resolved = model_c_type
        ipc_resolved = io_preset_c
        print(f"[backend] C: type={c_type_resolved} path={model_c_path or '(n/a)'} device={device} io_preset_c={ipc_resolved}")

        if model_c_type == "magenta":
            pass  # Magenta uses magenta_style_c per-frame; no model to load
        elif model_c_type == "torch7":
            if not model_c_path:
                print("[error] model_c path is required when model_c_type=torch7")
                sys.exit(2)
            try:
                net_t7_c = cv2.dnn.readNetFromTorch(str(model_c_path))
                try:
                    net_t7_c.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_c.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][C][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][C][ERROR] Could not load .t7 network from {model_c_path}: {e}")
                sys.exit(2)
        else:
            if not model_c_path:
                print("[error] model_c path is required for transformer/reconet C")
                sys.exit(2)
            if model_c_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_c_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_c = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_c = TransformerNet().to(device)
            _load_checkpoint_compat(model_c, str(model_c_path))
            model_c.eval()

    # --- Model D ---
    model_d = None
    net_t7_d = None
    # Treat D as "in use" if a model path, type, or magenta style is provided
    use_d = bool(
        getattr(args, "model_d", None)
        or getattr(args, "model_d_type", None)
        or getattr(args, "magenta_style_d", None)
    )
    d_type_resolved = None
    ipd_resolved = None
    if use_d:
        model_d_path_raw = getattr(args, "model_d", None)
        model_d_type = getattr(args, "model_d_type", None)

        # Infer type if not explicit
        if model_d_type is None:
            if model_d_path_raw and Path(model_d_path_raw).suffix.lower() == ".t7":
                model_d_type = "torch7"
            elif model_d_path_raw:
                model_d_type = args.model_type
            else:
                # No path provided, assume magenta if magenta_style_d exists
                model_d_type = "magenta"

        model_d_path = Path(model_d_path_raw).resolve() if model_d_path_raw else None
        io_preset_d = getattr(args, "io_preset_d", None) or io_preset
        d_type_resolved = model_d_type
        ipd_resolved = io_preset_d
        print(f"[backend] D: type={d_type_resolved} path={model_d_path or '(n/a)'} device={device} io_preset_d={ipd_resolved}")

        if model_d_type == "magenta":
            pass  # Magenta uses magenta_style_d per-frame; no model to load
        elif model_d_type == "torch7":
            if not model_d_path:
                print("[error] model_d path is required when model_d_type=torch7")
                sys.exit(2)
            try:
                net_t7_d = cv2.dnn.readNetFromTorch(str(model_d_path))
                try:
                    net_t7_d.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_d.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][D][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][D][ERROR] Could not load .t7 network from {model_d_path}: {e}")
                sys.exit(2)
        else:
            if not model_d_path:
                print("[error] model_d path is required for transformer/reconet D")
                sys.exit(2)
            if model_d_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_d_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_d = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_d = TransformerNet().to(device)
            _load_checkpoint_compat(model_d, str(model_d_path))
            model_d.eval()

    # --- Model E ---
    model_e = None
    net_t7_e = None
    use_e = bool(
        getattr(args, "model_e", None)
        or getattr(args, "model_e_type", None)
        or getattr(args, "magenta_style_e", None)
    )
    e_type_resolved = None
    ipe_resolved = None
    if use_e:
        model_e_path_raw = getattr(args, "model_e", None)
        model_e_type = getattr(args, "model_e_type", None)
        if model_e_type is None:
            if model_e_path_raw and Path(model_e_path_raw).suffix.lower() == ".t7":
                model_e_type = "torch7"
            elif model_e_path_raw:
                model_e_type = args.model_type
            else:
                model_e_type = "magenta"
        model_e_path = Path(model_e_path_raw).resolve() if model_e_path_raw else None
        io_preset_e = getattr(args, "io_preset_e", None) or io_preset
        e_type_resolved = model_e_type
        ipe_resolved = io_preset_e
        print(f"[backend] E: type={e_type_resolved} path={model_e_path or '(n/a)'} device={device} io_preset_e={ipe_resolved}")
        if model_e_type == "magenta":
            pass
        elif model_e_type == "torch7":
            if not model_e_path:
                print("[error] model_e path is required when model_e_type=torch7")
                sys.exit(2)
            try:
                net_t7_e = cv2.dnn.readNetFromTorch(str(model_e_path))
                try:
                    net_t7_e.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_e.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][E][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][E][ERROR] Could not load .t7 network from {model_e_path}: {e}")
                sys.exit(2)
        else:
            if not model_e_path:
                print("[error] model_e path is required for transformer/reconet E")
                sys.exit(2)
            if model_e_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_e_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_e = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_e = TransformerNet().to(device)
            _load_checkpoint_compat(model_e, str(model_e_path))
            model_e.eval()

    # --- Model F ---
    model_f = None
    net_t7_f = None
    use_f = bool(
        getattr(args, "model_f", None)
        or getattr(args, "model_f_type", None)
        or getattr(args, "magenta_style_f", None)
    )
    f_type_resolved = None
    ipf_resolved = None
    if use_f:
        model_f_path_raw = getattr(args, "model_f", None)
        model_f_type = getattr(args, "model_f_type", None)
        if model_f_type is None:
            if model_f_path_raw and Path(model_f_path_raw).suffix.lower() == ".t7":
                model_f_type = "torch7"
            elif model_f_path_raw:
                model_f_type = args.model_type
            else:
                model_f_type = "magenta"
        model_f_path = Path(model_f_path_raw).resolve() if model_f_path_raw else None
        io_preset_f = getattr(args, "io_preset_f", None) or io_preset
        f_type_resolved = model_f_type
        ipf_resolved = io_preset_f
        print(f"[backend] F: type={f_type_resolved} path={model_f_path or '(n/a)'} device={device} io_preset_f={ipf_resolved}")
        if model_f_type == "magenta":
            pass
        elif model_f_type == "torch7":
            if not model_f_path:
                print("[error] model_f path is required when model_f_type=torch7")
                sys.exit(2)
            try:
                net_t7_f = cv2.dnn.readNetFromTorch(str(model_f_path))
                try:
                    net_t7_f.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_f.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][F][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][F][ERROR] Could not load .t7 network from {model_f_path}: {e}")
                sys.exit(2)
        else:
            if not model_f_path:
                print("[error] model_f path is required for transformer/reconet F")
                sys.exit(2)
            if model_f_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_f_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_f = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_f = TransformerNet().to(device)
            _load_checkpoint_compat(model_f, str(model_f_path))
            model_f.eval()

    # --- Model G ---
    model_g = None
    net_t7_g = None
    use_g = bool(
        getattr(args, "model_g", None)
        or getattr(args, "model_g_type", None)
        or getattr(args, "magenta_style_g", None)
    )
    g_type_resolved = None
    ipg_resolved = None
    if use_g:
        model_g_path_raw = getattr(args, "model_g", None)
        model_g_type = getattr(args, "model_g_type", None)
        if model_g_type is None:
            if model_g_path_raw and Path(model_g_path_raw).suffix.lower() == ".t7":
                model_g_type = "torch7"
            elif model_g_path_raw:
                model_g_type = args.model_type
            else:
                model_g_type = "magenta"
        model_g_path = Path(model_g_path_raw).resolve() if model_g_path_raw else None
        io_preset_g = getattr(args, "io_preset_g", None) or io_preset
        g_type_resolved = model_g_type
        ipg_resolved = io_preset_g
        print(f"[backend] G: type={g_type_resolved} path={model_g_path or '(n/a)'} device={device} io_preset_g={ipg_resolved}")
        if model_g_type == "magenta":
            pass
        elif model_g_type == "torch7":
            if not model_g_path:
                print("[error] model_g path is required when model_g_type=torch7")
                sys.exit(2)
            try:
                net_t7_g = cv2.dnn.readNetFromTorch(str(model_g_path))
                try:
                    net_t7_g.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_g.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][G][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][G][ERROR] Could not load .t7 network from {model_g_path}: {e}")
                sys.exit(2)
        else:
            if not model_g_path:
                print("[error] model_g path is required for transformer/reconet G")
                sys.exit(2)
            if model_g_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_g_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_g = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_g = TransformerNet().to(device)
            _load_checkpoint_compat(model_g, str(model_g_path))
            model_g.eval()

    # --- Model H ---
    model_h = None
    net_t7_h = None
    use_h = bool(
        getattr(args, "model_h", None)
        or getattr(args, "model_h_type", None)
        or getattr(args, "magenta_style_h", None)
    )
    h_type_resolved = None
    iph_resolved = None
    if use_h:
        model_h_path_raw = getattr(args, "model_h", None)
        model_h_type = getattr(args, "model_h_type", None)
        if model_h_type is None:
            if model_h_path_raw and Path(model_h_path_raw).suffix.lower() == ".t7":
                model_h_type = "torch7"
            elif model_h_path_raw:
                model_h_type = args.model_type
            else:
                model_h_type = "magenta"
        model_h_path = Path(model_h_path_raw).resolve() if model_h_path_raw else None
        io_preset_h = getattr(args, "io_preset_h", None) or io_preset
        h_type_resolved = model_h_type
        iph_resolved = io_preset_h
        print(f"[backend] H: type={h_type_resolved} path={model_h_path or '(n/a)'} device={device} io_preset_h={iph_resolved}")
        if model_h_type == "magenta":
            pass
        elif model_h_type == "torch7":
            if not model_h_path:
                print("[error] model_h path is required when model_h_type=torch7")
                sys.exit(2)
            try:
                net_t7_h = cv2.dnn.readNetFromTorch(str(model_h_path))
                try:
                    net_t7_h.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net_t7_h.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception as e:
                    print(f"[torch7][H][WARN] backend/target not set: {e}")
            except Exception as e:
                print(f"[torch7][H][ERROR] Could not load .t7 network from {model_h_path}: {e}")
                sys.exit(2)
        else:
            if not model_h_path:
                print("[error] model_h path is required for transformer/reconet H")
                sys.exit(2)
            if model_h_type == "reconet":
                if not _HAS_RECONET:
                    print("[error] model_h_type=reconet but lib.ReCoNet is not available.")
                    sys.exit(2)
                model_h = _ReCoNet().to(device)
            else:
                TransformerNet = _get_transformer_net()
                model_h = TransformerNet().to(device)
            _load_checkpoint_compat(model_h, str(model_h_path))
            model_h.eval()

    blend = float(max(0.0, min(1.0, blend)))
    print(f">> smoothing: {smooth_lightness}  alpha={smooth_alpha}  |  blend={blend}")

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    print(f"[debug] scanning frames in {frames_dir}")
    # Collect frames
    frame_files = sorted([p for p in frames_dir.iterdir()
                          if p.is_file() and p.name.startswith("frame_")
                          and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if stride and stride > 1:
        frame_files = frame_files[::stride]
    if max_frames:
        frame_files = frame_files[:max_frames]

    print(f"[debug] found {len(frame_files)} staged frame(s)")

    if not frame_files:
        print(f"[error] No frames found to style in: {frames_dir}")
        print("[error] Expect files named like frame_0001.jpg/.png (.jpeg allowed).")
        try:
            listing = sorted([p.name for p in frames_dir.iterdir()])
            print("[error] Current directory listing (first 40):", listing[:40])
        except Exception:
            pass
        sys.exit(1)

    if save_map is None:
        save_map = {}

    # If a mask directory was requested (and no single --mask is set), validate that masks exist.
    if getattr(args, "mask_dir", None) and not getattr(args, "mask", None):
        try:
            md = Path(args.mask_dir)
            missing = []
            total = 0
            for p in frame_files:
                stem_num = p.stem.split("_")[-1]  # e.g., 0001 from frame_0001.png
                mpath = md / f"mask_{stem_num}.png"
                total += 1
                if not mpath.exists():
                    missing.append(p.name)
            if total > 0 and len(missing) == total:
                print(f"[mask][ERROR] --mask_dir set to {md} but no masks like mask_0001.png were found.")
                print("               Refusing to run unmasked; generate masks or remove --mask_dir.")
                sys.exit(2)
            elif missing:
                print(f"[mask][WARN] {len(missing)}/{total} mask(s) missing under {md}.")
                print("            Missing-mask frames will be fully stylized unless a global --mask is provided.")
        except Exception as _e:
            print(f"[mask][WARN] could not validate --mask_dir: {_e}")

    # Temporal caches
    prev_gray = None  # uint8 HxW (content grayscale)
    prev_styled01 = None  # float32 HxWx3 in [0,1] (pre-LAB/pre-blend)
    prev_L = None  # LAB Lightness EMA cache
    last_flow = None  # float32 HxWx2
    # LAB smoothing chroma caches
    prev_aC = None
    prev_bC = None
    prev_frame_size = None  # (H,W) of previous frame after staging; used to reset caches on size changes

    # Motion-adaptive blend knobs (internal; tweak here if needed)
    MOTION_NORM = 8.0  # px; larger => less area considered "high motion"
    MIN_ALPHA = 0.40  # floor style weight in high motion
    GAUSS_SIGMA = 3.0  # blur sigma for motion mask smoothing

    # --- per frame loop ---
    for idx, frame_path in enumerate(frame_files, start=1):
        t0 = time.perf_counter()
        if idx == 1 or idx % 25 == 0:
            print(f"[frame][{idx}] START  rss={_rss_mb():.1f} MB  file={frame_path.name}")
        else:
            print(f"[frame][{idx}] START")
        pil_rgb = Image.open(frame_path).convert("RGB")
        # Optional downscale for inference to avoid OOM on huge frames
        pil_src = pil_rgb
        infer_res = int(getattr(args, "inference_res", 0) or 0)
        if infer_res > 0:
            w0, h0 = pil_rgb.size
            m0 = max(w0, h0)
            if m0 > infer_res:
                r = infer_res / float(m0)
                w1, h1 = int(round(w0 * r)), int(round(h0 * r))
                pil_src = pil_rgb.resize((w1, h1), Image.Resampling.LANCZOS)
                print(f"[A][{idx}] inference_res active: {w0}x{h0} -> {w1}x{h1}")
        x_orig01 = to_tensor(pil_rgb).unsqueeze(0)  # [1,3,H,W] 0..1
        x_src01 = to_tensor(pil_src).unsqueeze(0)  # [1,3,h,w] 0..1 (may be smaller than original)
        gray = np.array(pil_rgb.convert("L"), dtype=np.uint8)  # HxW
        H0, W0 = x_orig01.shape[-2], x_orig01.shape[-1]
        # If frame size changed (e.g., mixed orientations or different paddings),
        # reset temporal caches to avoid shape/broadcast errors and invalid flow.
        curr_size = (H0, W0)
        if (prev_frame_size is not None) and (prev_frame_size != curr_size):
            print(f"[size][reset] frame dims changed {prev_frame_size} -> {curr_size}; resetting EMA/flow caches")
            prev_gray = None
            prev_styled01 = None
            prev_L = None
            prev_aC = None
            prev_bC = None
            last_flow = None
        prev_frame_size = curr_size
        # Early diagnostics for A forward (after dimensions known)
        try:
            th = torch.get_num_threads()
        except Exception:
            th = -1

        # ---- Check for optimized region styling ----
        region_optimize = getattr(args, "region_optimize", False)
        region_mode = getattr(args, "region_mode", None)

        if region_optimize and region_mode and _HAS_REGION_BLEND:
            # OPTIMIZED PATH: Only style the pixels needed for each region
            region_count = getattr(args, "region_count", None) or 4
            region_feather = getattr(args, "region_feather", 20)
            region_assignment = getattr(args, "region_assignment", "sequential")
            region_original = getattr(args, "region_original", 0.0)
            region_rotate = getattr(args, "region_rotate", 0.0)
            region_blend_spec = getattr(args, "region_blend_spec", None)
            region_scales = getattr(args, "region_scales", None)
            region_padding = getattr(args, "region_padding", 64)
            region_morph_spec = getattr(args, "region_morph", None)
            region_sizes_spec = getattr(args, "region_sizes", None)

            # Parse blend animation settings
            blend_animate_spec = getattr(args, "blend_animate", None)
            blend_animate_regions_spec = getattr(args, "blend_animate_regions", None)

            # Parse scale animation settings
            scale_animate_spec = getattr(args, "scale_animate", None)
            scale_animate_regions_spec = getattr(args, "scale_animate_regions", None)

            # Parse morph animation settings (once per video)
            if not hasattr(args, '_morph_animation'):
                args._morph_animation = parse_morph_animation(region_morph_spec) if region_morph_spec else MorphAnimation(enabled=False)

            # Parse seed - for optimized mode, default to fixed seed for stable regions
            seed_str = getattr(args, "region_seed", None)
            if seed_str is None:
                region_seed = 42  # Default to fixed seed for stable regions across frames
            elif seed_str.lower() == "random":
                region_seed = None
            elif seed_str.lower() == "fixed":
                region_seed = 42
            else:
                try:
                    region_seed = int(seed_str)
                except ValueError:
                    region_seed = 42

            # Count active models
            num_models = sum([1, use_b, use_c, use_d, use_e, use_f, use_g, use_h])

            # Parse region sizes (once per video)
            if not hasattr(args, '_region_sizes'):
                args._region_sizes = parse_region_sizes(region_sizes_spec, region_count) if region_sizes_spec else None

            region_sizes = args._region_sizes

            # Generate masks (or use cached) - always cache in optimized mode for consistency
            cache_key = (H0, W0, region_mode, region_count, region_seed, region_feather, region_sizes_spec)
            if hasattr(args, '_region_cache') and cache_key in args._region_cache:
                base_masks, configs = args._region_cache[cache_key]
            else:
                base_masks = generate_region_masks(H0, W0, region_mode, region_count, region_seed, region_feather, region_sizes)
                configs = parse_region_configs(
                    num_regions=len(base_masks),
                    num_models=num_models,
                    assignment=region_assignment,
                    blend_spec=region_blend_spec,
                    scale_spec=region_scales,
                    seed=region_seed,
                    original_chance=region_original
                )
                # Always cache masks in optimized mode
                if not hasattr(args, '_region_cache'):
                    args._region_cache = {}
                args._region_cache[cache_key] = (base_masks, configs)

            # Start with base masks
            masks = base_masks

            # Apply rotation if needed
            if region_rotate != 0:
                angle = idx * region_rotate
                masks = rotate_all_masks(masks, angle)
                masks = [feather_mask(m, region_feather // 2) for m in masks]

            # Apply organic morph animation if enabled
            morph_anim = args._morph_animation
            if morph_anim.enabled:
                masks = warp_all_masks_organic(masks, morph_anim, idx)
                masks = [feather_mask(m, max(5, region_feather // 4)) for m in masks]

            # Prepare crops
            crops = prepare_region_crops(masks, configs, H0, W0, region_padding)
            coverage = compute_crop_coverage(crops, H0, W0)
            models_needed = get_models_needed_for_regions(crops)

            # Parse blend animations (once per video, cache on args)
            if not hasattr(args, '_blend_animations'):
                if blend_animate_regions_spec:
                    args._blend_animations = parse_region_blend_animations(
                        blend_animate_regions_spec, len(crops)
                    )
                elif blend_animate_spec:
                    args._blend_animations = parse_region_blend_animations(
                        blend_animate_spec, len(crops)
                    )
                else:
                    args._blend_animations = None

            blend_animations = args._blend_animations

            # Parse scale animations (once per video, cache on args)
            if not hasattr(args, '_scale_animations'):
                if scale_animate_regions_spec:
                    args._scale_animations = parse_region_scale_animations(
                        scale_animate_regions_spec, len(crops)
                    )
                elif scale_animate_spec:
                    args._scale_animations = parse_region_scale_animations(
                        scale_animate_spec, len(crops)
                    )
                else:
                    args._scale_animations = None

            scale_animations = args._scale_animations

            if idx <= 2:
                anim_info = ""
                if blend_animations and any(a.enabled for a in blend_animations):
                    enabled_count = sum(1 for a in blend_animations if a.enabled)
                    anim_info = f" blend_anim={enabled_count}/{len(blend_animations)}"
                scale_anim_info = ""
                if scale_animations and any(a.enabled for a in scale_animations):
                    enabled_count = sum(1 for a in scale_animations if a.enabled)
                    scale_anim_info = f" scale_anim={enabled_count}/{len(scale_animations)}"
                morph_info = f" morph={morph_anim.mode}" if morph_anim.enabled else ""
                rotate_info = f" rotate={region_rotate}°/f" if region_rotate != 0 else ""
                sizes_info = f" sizes={region_sizes}" if region_sizes else ""
                print(f"[region-opt][{idx}] mode={region_mode} regions={len(crops)} models_needed={models_needed} "
                      f"coverage={coverage:.1%} padding={region_padding}px{anim_info}{scale_anim_info}{morph_info}{rotate_info}{sizes_info}")

            # Build model info lookup: model_idx -> (model, net_t7, type, io_preset, style_img)
            model_info = {}
            # Model A = 0
            model_info[0] = (model, net_t7, args.model_type, io_preset, getattr(args, "magenta_style", None))
            # Model B = 1
            if use_b:
                style_b = getattr(args, "magenta_style_b", None) or getattr(args, "magenta_style", None)
                model_info[1] = (model_b, net_t7_b, b_type_resolved, ipb_resolved, style_b)
            # Model C = 2
            if use_c:
                style_c = getattr(args, "magenta_style_c", None) or getattr(args, "magenta_style", None)
                model_info[2] = (model_c, net_t7_c, c_type_resolved, ipc_resolved, style_c)
            # Model D = 3
            if use_d:
                style_d = getattr(args, "magenta_style_d", None) or getattr(args, "magenta_style", None)
                model_info[3] = (model_d, net_t7_d, d_type_resolved, ipd_resolved, style_d)
            # Model E = 4
            if use_e:
                style_e = getattr(args, "magenta_style_e", None) or getattr(args, "magenta_style", None)
                model_info[4] = (model_e, net_t7_e, e_type_resolved, ipe_resolved, style_e)
            # Model F = 5
            if use_f:
                style_f = getattr(args, "magenta_style_f", None) or getattr(args, "magenta_style", None)
                model_info[5] = (model_f, net_t7_f, f_type_resolved, ipf_resolved, style_f)
            # Model G = 6
            if use_g:
                style_g = getattr(args, "magenta_style_g", None) or getattr(args, "magenta_style", None)
                model_info[6] = (model_g, net_t7_g, g_type_resolved, ipg_resolved, style_g)
            # Model H = 7
            if use_h:
                style_h = getattr(args, "magenta_style_h", None) or getattr(args, "magenta_style", None)
                model_info[7] = (model_h, net_t7_h, h_type_resolved, iph_resolved, style_h)

            # Style crops for each model
            styled_crops = {}  # {model_idx: {region_idx: styled_tensor}}

            for model_idx in models_needed:
                if model_idx not in model_info:
                    print(f"[region-opt][WARN] Model {model_idx} requested but not loaded, skipping")
                    continue

                mdl, net7, mtype, mpreset, mstyle = model_info[model_idx]
                model_letter = chr(ord('A') + model_idx)
                regions_for_model = get_regions_for_model(crops, model_idx)

                styled_crops[model_idx] = {}

                for crop_info in regions_for_model:
                    x1, y1, x2, y2 = crop_info.padded_bbox
                    crop_h, crop_w = y2 - y1, x2 - x1

                    # Extract crop from full-resolution original (not potentially downscaled x_src01)
                    crop_tensor = extract_crop(x_orig01.squeeze(0), crop_info.padded_bbox)

                    # Apply per-region scale - use animated scale if available
                    base_scale = crop_info.config.scale
                    if scale_animations and crop_info.region_idx < len(scale_animations):
                        scale_anim = scale_animations[crop_info.region_idx]
                        region_scale = compute_animated_scale(base_scale, idx, scale_anim)
                    else:
                        region_scale = base_scale

                    if region_scale < 1.0:
                        # Downscale crop for inference
                        scaled_h = max(1, int(crop_h * region_scale))
                        scaled_w = max(1, int(crop_w * region_scale))
                        crop_tensor_scaled = F.interpolate(
                            crop_tensor.unsqueeze(0), size=(scaled_h, scaled_w),
                            mode='bilinear', align_corners=False
                        ).squeeze(0)
                        crop_pil = Image.fromarray((crop_tensor_scaled.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                        infer_h, infer_w = scaled_h, scaled_w
                    else:
                        crop_pil = Image.fromarray((crop_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                        crop_tensor_scaled = crop_tensor
                        infer_h, infer_w = crop_h, crop_w

                    if idx <= 2:
                        scale_info = f" scale={region_scale:.2f} infer={infer_w}x{infer_h}" if region_scale < 1.0 else ""
                        print(f"[region-opt][{idx}] Styling region {crop_info.region_idx} with model {model_letter} "
                              f"crop={crop_w}x{crop_h}{scale_info} (of {W0}x{H0})")

                    # Style the crop
                    try:
                        if mtype == "magenta":
                            # Magenta needs a file path - save temp crop
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                                crop_pil.save(tmp.name)
                                out_pil = _magenta_style_pil(
                                    tmp.name, mstyle,
                                    tile_size=int(args.magenta_tile),
                                    overlap=int(args.magenta_overlap),
                                    target_resolution=None,
                                    model_root=str(getattr(args, "magenta_model_root", "/app/models/magenta")),
                                )
                                import os
                                os.unlink(tmp.name)
                            styled_crop = to_tensor(out_pil).clamp(0, 1)
                        elif mtype == "torch7":
                            pil_out = _torch7_forward_openv(net7, crop_pil)
                            styled_crop = to_tensor(pil_out).clamp(0, 1)
                        else:
                            # Transformer/ReCoNet
                            crop_in = crop_tensor_scaled.unsqueeze(0)
                            with torch.no_grad():
                                if mpreset == "imagenet_255":
                                    x255 = crop_in * 255.0
                                    x_norm = ((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0)).to(device)
                                    y = mdl(x_norm).cpu()
                                    styled_crop = (y / 255.0).clamp(0, 1).squeeze(0)
                                elif mpreset == "imagenet_01":
                                    x_norm = ((crop_in - IMAGENET_MEAN) / IMAGENET_STD).to(device)
                                    y = mdl(x_norm).cpu()
                                    styled_crop = (y * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).squeeze(0)
                                else:
                                    x_in = (crop_in * 255.0).to(device)
                                    y = mdl(x_in).cpu()
                                    styled_crop = (y / 255.0).clamp(0, 1).squeeze(0)

                        # Upscale back to full crop size if we downscaled
                        if region_scale < 1.0:
                            styled_crop = F.interpolate(
                                styled_crop.unsqueeze(0), size=(crop_h, crop_w),
                                mode='bilinear', align_corners=False
                            ).squeeze(0)

                        styled_crops[model_idx][crop_info.region_idx] = styled_crop

                    except Exception as e:
                        print(f"[region-opt][{idx}][ERROR] Failed to style region {crop_info.region_idx} "
                              f"with model {model_letter}: {e}")
                        # Fallback to original crop
                        styled_crops[model_idx][crop_info.region_idx] = crop_tensor

            # Composite all crops (with optional animated blend weights)
            out01 = composite_from_crops(
                styled_crops=styled_crops,
                crops=crops,
                original=x_orig01.squeeze(0) if region_original > 0 or (region_blend_spec and 'O' in region_blend_spec.upper()) else None,
                H=H0, W=W0,
                frame_idx=idx,
                blend_animations=blend_animations
            )

            t1 = time.perf_counter()
            print(f"[region-opt][{idx}] done dt={t1 - t0:.3f}s out range=({float(out01.min()):.3f}..{float(out01.max()):.3f})")

            # Skip normal model processing - go straight to post-processing
            # (need to set these for later smoothing/flow code)
            num_models = 1  # Pretend single model for downstream code

        else:
            # STANDARD PATH: Full-frame styling
            print(
                f"[A][{idx}] forward start: res={W0}x{H0} (src={pil_src.size[0]}x{pil_src.size[1]}) backend={args.model_type} device={device} threads={th} rss={_rss_mb():.1f} MB")
            print(f"[A][{idx}] using io_preset='{io_preset}'")
            sys.stdout.flush()

            # ---- Per-frame stylization backend (Model A) ----
            try:
                if use_magenta:
                    out_mag_pil = _magenta_style_pil(
                        str(frame_path),
                        args.magenta_style,
                        tile_size=int(args.magenta_tile),
                        overlap=int(args.magenta_overlap),
                        target_resolution=(int(args.magenta_target_res) if args.magenta_target_res else None),
                        model_root=str(getattr(args, "magenta_model_root", "/app/models/magenta")),
                    )
                    out01 = to_tensor(out_mag_pil).clamp(0, 1)
                elif use_torch7:
                    try:
                        print(f"[torch7] styling frame {idx}/{len(frame_files)} …")
                        pil_out_t7 = _torch7_forward_openv(net_t7, pil_src)
                    except Exception as e:
                        print(f"[torch7][ERROR] forward failed on frame {idx}: {e} — retrying at half-size")
                        try:
                            pil_small = pil_src.resize((max(1, pil_src.width // 2), max(1, pil_src.height // 2)),
                                                       Image.Resampling.BILINEAR)
                            pil_out_small = _torch7_forward_openv(net_t7, pil_small)
                            pil_out_t7 = pil_out_small.resize(pil_src.size, Image.Resampling.BILINEAR)
                            print("[torch7] retry succeeded at half-size")
                        except Exception as e2:
                            print(f"[torch7][FALLBACK] second attempt failed on frame {idx}: {e2}; using original frame")
                            pil_out_t7 = pil_src
                    out01 = to_tensor(pil_out_t7).clamp(0, 1)
                else:
                    with torch.no_grad():
                        print(f"[A][{idx}] io_preset branch start: {io_preset}")
                        if io_preset == "tanh":
                            x_in = (x_src01 * 2.0 - 1.0).to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            out01 = ((y + 1.0) * 0.5).clamp(0, 1).squeeze(0)
                        elif io_preset == "imagenet_01":
                            x_in = ((x_src01 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            out01 = (y * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).squeeze(0)
                        elif io_preset == "imagenet_255":
                            x255 = x_src01 * 255.0
                            x_in = ((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0)).to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            out01 = (y / 255.0).clamp(0, 1).squeeze(0)
                        elif io_preset == "caffe_bgr":
                            x255 = x_src01 * 255.0
                            x_bgr255 = x255[:, [2, 1, 0], :, :]
                            CAFFE_MEAN = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
                            x_in = (x_bgr255 - CAFFE_MEAN).to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            y_rgb255 = y[:, [2, 1, 0], :, :]
                            out01 = (y_rgb255 / 255.0).clamp(0, 1).squeeze(0)
                        elif io_preset == "raw_255":
                            x_in = (x_src01 * 255.0).to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            out01 = (y / 255.0).clamp(0, 1).squeeze(0)
                        elif io_preset == "raw_01":
                            x_in = x_src01.to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            out01 = y.clamp(0, 1).squeeze(0)
                        else:
                            x_in = (x_src01 * 255.0).to(device)
                            y = model(x_in).cpu()
                            print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                            out01 = (y / 255.0).clamp(0, 1).squeeze(0)
            except Exception as e:
                print(f"[A][{idx}][ERROR] forward crashed: {type(e).__name__}: {e}")
                sys.stdout.flush()
                raise

            t1 = time.perf_counter()
            o_min = float(out01.min())
            o_max = float(out01.max())
            print(
                f"[frame][{idx}] A-done  dt={t1 - t0:.3f}s  out01 range=({o_min:.3f}..{o_max:.3f})  rss={_rss_mb():.1f} MB")
            sys.stdout.flush()

            # Save first couple of A outputs for debugging
            if idx <= 2:
                debug_dir = (frames_dir.parent / "debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                try:
                    Image.fromarray(
                        (out01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    ).save(debug_dir / f"A_out_{idx:04d}.jpg", quality=92)
                    pil_rgb.save(debug_dir / f"IN_{idx:04d}.jpg", quality=92)
                    print(f"[debug] wrote {debug_dir}/A_out_{idx:04d}.jpg and IN_{idx:04d}.jpg")
                except Exception as e:
                    print(f"[debug][WARN] could not save A debug frames: {e}")

            # Lock stylized to content size (avoid later mismatches)
            H1, W1 = out01.shape[-2], out01.shape[-1]
            if (H1, W1) != (H0, W0):
                out01 = F.interpolate(out01.unsqueeze(0), size=(H0, W0),
                                      mode="bilinear", align_corners=False).squeeze(0)

            # ---- Four-model blending ----
            num_models = sum([1, use_b, use_c, use_d])  # Count active models
        if num_models > 1:
            outputs = [out01]  # Start with model A's output
            model_names = ["A"]

            # Helper function to process model output
            def process_model_output(model, net_t7, model_type, io_preset, frame_path, pil_rgb, pil_src, x_orig01, x_src01, device, magenta_style,
                                     model_name, idx, model_root="/app/models/magenta"):
                t0 = time.perf_counter()
                print(
                    f"[{model_name}][{idx}] forward start: res={pil_rgb.size[0]}x{pil_rgb.size[1]} (src={pil_src.size[0]}x{pil_src.size[1]}) backend={model_type} device={device} threads={torch.get_num_threads()} rss={_rss_mb():.1f} MB")
                print(f"[{model_name}][{idx}] using io_preset='{io_preset}'")
                sys.stdout.flush()

                try:
                    if model_type == "magenta":
                        out_pil = _magenta_style_pil(
                            str(frame_path),
                            magenta_style,
                            tile_size=int(args.magenta_tile),
                            overlap=int(args.magenta_overlap),
                            target_resolution=(int(args.magenta_target_res) if args.magenta_target_res else None),
                            model_root=model_root,
                        )
                        out = to_tensor(out_pil).clamp(0, 1)
                    elif model_type == "torch7":
                        try:
                            print(f"[torch7][{model_name}] styling frame {idx}/{len(frame_files)} …")
                            pil_out = _torch7_forward_openv(net_t7, pil_src)
                        except Exception as e:
                            print(f"[torch7][{model_name}][ERROR] forward failed: {e} — retrying at half-size")
                            try:
                                pil_small = pil_src.resize((max(1, pil_src.width // 2), max(1, pil_src.height // 2)),
                                                           Image.Resampling.BILINEAR)
                                pil_out_small = _torch7_forward_openv(net_t7, pil_small)
                                pil_out = pil_out_small.resize(pil_src.size, Image.Resampling.BILINEAR)
                                print(f"[torch7][{model_name}] retry succeeded at half-size")
                            except Exception as e2:
                                print(
                                    f"[torch7][{model_name}][FALLBACK] second attempt failed: {e2}; using original frame")
                                pil_out = pil_src
                        out = to_tensor(pil_out).clamp(0, 1)
                    else:
                        with torch.no_grad():
                            print(f"[{model_name}][{idx}] io_preset branch start: {io_preset}")
                            if io_preset == "tanh":
                                x_in = (x_src01 * 2.0 - 1.0).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = ((y + 1.0) * 0.5).clamp(0, 1).squeeze(0)
                            elif io_preset == "imagenet_01":
                                x_in = ((x_src01 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).squeeze(0)
                            elif io_preset == "imagenet_255":
                                x255 = x_src01 * 255.0
                                x_in = ((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0)).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y / 255.0).clamp(0, 1).squeeze(0)
                            elif io_preset == "caffe_bgr":
                                x255 = x_src01 * 255.0
                                x_bgr255 = x255[:, [2, 1, 0], :, :]
                                CAFFE_MEAN = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
                                x_in = (x_bgr255 - CAFFE_MEAN).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                y_rgb255 = y[:, [2, 1, 0], :, :]
                                out = (y_rgb255 / 255.0).clamp(0, 1).squeeze(0)
                            elif io_preset == "raw_255":
                                x_in = (x_src01 * 255.0).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y / 255.0).clamp(0, 1).squeeze(0)
                            elif io_preset == "raw_01":
                                x_in = x_src01.to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = y.clamp(0, 1).squeeze(0)
                            else:
                                x_in = (x_src01 * 255.0).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y / 255.0).clamp(0, 1).squeeze(0)
                    t1 = time.perf_counter()
                    o_min = float(out.min())
                    o_max = float(out.max())
                    print(
                        f"[frame][{idx}] {model_name}-done dt={t1 - t0:.3f}s out range=({o_min:.3f}..{o_max:.3f}) rss={_rss_mb():.1f} MB")
                    sys.stdout.flush()
                    return out
                except Exception as e:
                    print(
                        f"[{model_name}][{idx}][ERROR] forward failed: {type(e).__name__}: {e} — falling back to original frame")
                    return to_tensor(pil_rgb).clamp(0, 1)

            # Process model B
            if use_b:
                style_img_b = getattr(args, "magenta_style_b", None) or getattr(args, "magenta_style", None)
                if b_type_resolved == "magenta" and not style_img_b:
                    raise ValueError("--magenta_style_b or --magenta_style required for model B when type=magenta")
                out01_b = process_model_output(model_b, net_t7_b, b_type_resolved, ipb_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_b, "B", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_b)
                model_names.append("B")

            # Process model C
            if use_c:
                style_img_c = getattr(args, "magenta_style_c", None) or getattr(args, "magenta_style", None)
                if c_type_resolved == "magenta" and not style_img_c:
                    raise ValueError("--magenta_style_c or --magenta_style required for model C when type=magenta")
                out01_c = process_model_output(model_c, net_t7_c, c_type_resolved, ipc_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_c, "C", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_c)
                model_names.append("C")

            # Process model D
            if use_d:
                style_img_d = getattr(args, "magenta_style_d", None) or getattr(args, "magenta_style", None)
                if d_type_resolved == "magenta" and not style_img_d:
                    raise ValueError("--magenta_style_d or --magenta_style required for model D when type=magenta")
                out01_d = process_model_output(model_d, net_t7_d, d_type_resolved, ipd_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_d, "D", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_d)
                model_names.append("D")

            # Process model E
            if use_e:
                style_img_e = getattr(args, "magenta_style_e", None) or getattr(args, "magenta_style", None)
                if e_type_resolved == "magenta" and not style_img_e:
                    raise ValueError("--magenta_style_e or --magenta_style required for model E when type=magenta")
                out01_e = process_model_output(model_e, net_t7_e, e_type_resolved, ipe_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_e, "E", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_e)
                model_names.append("E")

            # Process model F
            if use_f:
                style_img_f = getattr(args, "magenta_style_f", None) or getattr(args, "magenta_style", None)
                if f_type_resolved == "magenta" and not style_img_f:
                    raise ValueError("--magenta_style_f or --magenta_style required for model F when type=magenta")
                out01_f = process_model_output(model_f, net_t7_f, f_type_resolved, ipf_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_f, "F", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_f)
                model_names.append("F")

            # Process model G
            if use_g:
                style_img_g = getattr(args, "magenta_style_g", None) or getattr(args, "magenta_style", None)
                if g_type_resolved == "magenta" and not style_img_g:
                    raise ValueError("--magenta_style_g or --magenta_style required for model G when type=magenta")
                out01_g = process_model_output(model_g, net_t7_g, g_type_resolved, ipg_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_g, "G", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_g)
                model_names.append("G")

            # Process model H
            if use_h:
                style_img_h = getattr(args, "magenta_style_h", None) or getattr(args, "magenta_style", None)
                if h_type_resolved == "magenta" and not style_img_h:
                    raise ValueError("--magenta_style_h or --magenta_style required for model H when type=magenta")
                out01_h = process_model_output(model_h, net_t7_h, h_type_resolved, iph_resolved, frame_path, pil_rgb, pil_src, x_orig01, x_src01,
                                               device, style_img_h, "H", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_h)
                model_names.append("H")

            # Debug save for first two frames of additional models
            if idx <= 2:
                debug_dir = frames_dir.parent / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                for i, (out, name) in enumerate(zip(outputs[1:], model_names[1:]), 1):
                    try:
                        Image.fromarray(
                            (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                        ).save(debug_dir / f"{name}_out_{idx:04d}.jpg", quality=92)
                        print(f"[debug] wrote {debug_dir}/{name}_out_{idx:04d}.jpg")
                    except Exception as e:
                        print(f"[debug][WARN] could not save {name} debug frames: {e}")

            # Ensure all outputs match content size
            for i, out in enumerate(outputs):
                H, W = out.shape[-2], out.shape[-1]
                if (H, W) != (H0, W0):
                    outputs[i] = F.interpolate(out.unsqueeze(0), size=(H0, W0), mode="bilinear",
                                               align_corners=False).squeeze(0)

            # Blend outputs
            region_mode = getattr(args, "region_mode", None)

            if region_mode and _HAS_REGION_BLEND:
                # Region-based spatial blending
                region_count = getattr(args, "region_count", None) or num_models
                region_feather = getattr(args, "region_feather", 20)
                region_assignment = getattr(args, "region_assignment", "random")
                region_original = getattr(args, "region_original", 0.0)
                region_rotate = getattr(args, "region_rotate", 0.0)
                region_blend_spec = getattr(args, "region_blend_spec", None)
                region_scales = getattr(args, "region_scales", None)
                region_morph_spec = getattr(args, "region_morph", None)

                # Parse morph animation (once, cached on args)
                if not hasattr(args, '_morph_animation_blend'):
                    args._morph_animation_blend = parse_morph_animation(region_morph_spec) if region_morph_spec else MorphAnimation(enabled=False)
                morph_anim = args._morph_animation_blend

                # Parse seed - default to "fixed" if rotating or morphing for consistency
                seed_str = getattr(args, "region_seed", None)
                if seed_str is None:
                    if region_rotate != 0 or morph_anim.enabled:
                        region_seed = 42  # Default to fixed seed when animating
                    else:
                        region_seed = None  # Random each frame
                elif seed_str.lower() == "random":
                    region_seed = None
                elif seed_str.lower() == "fixed":
                    region_seed = 42
                else:
                    try:
                        region_seed = int(seed_str)
                    except ValueError:
                        region_seed = None

                # Parse weights for weighted assignment
                weights = None
                if region_assignment == "weighted":
                    try:
                        weights = parse_blend_weights(getattr(args, "blend_models_weights", None), num_models)
                    except Exception:
                        weights = None

                # Check if we need advanced blending (per-region blends or scales)
                use_advanced = region_blend_spec or region_scales

                if use_advanced:
                    # Advanced blending with per-region model blends and/or scales
                    required_scales = get_required_scales(
                        num_regions=region_count,
                        num_models=num_models,
                        assignment=region_assignment,
                        blend_spec=region_blend_spec,
                        scale_spec=region_scales,
                        seed=region_seed,
                        original_chance=region_original
                    )

                    # Build styled outputs at each required scale
                    styled_outputs_by_scale = {}
                    for scale in required_scales:
                        if scale == 1.0:
                            # Use already-computed full resolution outputs
                            styled_outputs_by_scale[1.0] = outputs
                        else:
                            # Re-style at lower resolution for this scale
                            # For now, just downscale the full-res outputs (true multi-scale would re-run model)
                            scaled_outputs = []
                            scaled_H, scaled_W = int(H0 * scale), int(W0 * scale)
                            for out in outputs:
                                # Downscale then upscale to simulate lower-res styling
                                out_small = F.interpolate(out.unsqueeze(0), size=(scaled_H, scaled_W),
                                                          mode="bilinear", align_corners=False)
                                # Keep at small size - composite_regions_advanced will upscale as needed
                                scaled_outputs.append(out_small.squeeze(0))
                            styled_outputs_by_scale[scale] = scaled_outputs

                    if idx <= 2:
                        print(f"[blend][{idx}] Advanced region blend: scales={list(styled_outputs_by_scale.keys())} blend_spec={region_blend_spec}")

                    out01 = blend_by_regions_advanced(
                        styled_outputs_by_scale=styled_outputs_by_scale,
                        H=H0, W=W0,
                        mode=region_mode,
                        region_count=region_count,
                        assignment=region_assignment,
                        blend_spec=region_blend_spec,
                        scale_spec=region_scales,
                        weights=weights,
                        feather=region_feather,
                        seed=region_seed,
                        original=x_orig01.squeeze(0) if region_original > 0 or (region_blend_spec and 'O' in region_blend_spec.upper()) else None,
                        original_chance=region_original,
                        frame_idx=idx,
                        rotation_rate=region_rotate,
                        morph=morph_anim
                    )
                else:
                    # Simple region blending (original behavior)
                    out01 = blend_by_regions(
                        styled_outputs=outputs,
                        H=H0, W=W0,
                        mode=region_mode,
                        region_count=region_count,
                        assignment=region_assignment,
                        weights=weights,
                        feather=region_feather,
                        seed=region_seed,
                        original=x_orig01.squeeze(0) if region_original > 0 else None,
                        original_chance=region_original,
                        frame_idx=idx,
                        rotation_rate=region_rotate,
                        morph=morph_anim
                    )

                if idx <= 2:
                    rotate_info = f" rotate={region_rotate}°/frame" if region_rotate != 0 else ""
                    morph_info = f" morph={morph_anim.mode}" if morph_anim.enabled else ""
                    print(f"[blend][{idx}] Region blend: mode={region_mode} regions={region_count} assignment={region_assignment} original_chance={region_original:.0%}{rotate_info}{morph_info}")

            elif getattr(args, "blend_models_lab", False):
                # LAB blending: L from A, a/b from weighted mix of B, C, D
                wL, wab = parse_lab_weights(getattr(args, "blend_models_lab_weights", None))
                weights_rest = parse_blend_weights(getattr(args, "blend_models_weights", None), max(num_models - 1, 1))
                if len(weights_rest) == 1:
                    weights_rest = [1.0]
                elif len(weights_rest) == 2:
                    weights_rest = weights_rest
                elif len(weights_rest) == 3:
                    weights_rest = weights_rest
                else:
                    weights_rest = [1.0 / max(num_models - 1, 1)] * max(num_models - 1, 1)

                pil_a = to_pil(outputs[0].clamp(0, 1))
                lab_a = np.array(pil_a.convert("LAB"), dtype=np.uint8).astype(np.float32)
                lab_mix = lab_a.copy()

                a_mix = np.zeros_like(lab_a[..., 1])
                b_mix = np.zeros_like(lab_a[..., 2])
                for out, w in zip(outputs[1:], weights_rest):
                    pil_out = to_pil(out.clamp(0, 1))
                    lab_out = np.array(pil_out.convert("LAB"), dtype=np.uint8).astype(np.float32)
                    a_mix += w * lab_out[..., 1]
                    b_mix += w * lab_out[..., 2]

                lab_mix[..., 1] = np.clip(wL * lab_a[..., 1] + wab * a_mix, 0, 255)
                lab_mix[..., 2] = np.clip(wL * lab_a[..., 2] + wab * b_mix, 0, 255)
                out_img_abcd = Image.fromarray(lab_mix.astype(np.uint8), mode="LAB").convert("RGB")
                out01 = to_tensor(out_img_abcd).clamp(0, 1)
                print(f"[blend][{idx}] LAB blend: L={wL},ab={wab},rest={weights_rest}")

            else:
                # RGB blending
                weights = parse_blend_weights(getattr(args, "blend_models_weights", None), num_models)
                out01 = torch.zeros_like(outputs[0])
                for out, w in zip(outputs, weights):
                    out01 += w * out
                out01 = out01.clamp(0, 1)
                print(f"[blend][{idx}] RGB blend: {','.join(model_names)} weights={weights}")

        if idx <= 3:
            print(f"[debug] frame {idx}  out01 range=({float(out01.min()):.3f}..{float(out01.max()):.3f})")

        # ---- Flow-guided EMA (pre-LAB/pre-blend) ----
        if args.flow_ema and (prev_gray is not None) and (prev_styled01 is not None):
            ds = max(1, int(args.flow_downscale))
            if ds > 1:
                gray_small = cv2.resize(gray, (W0 // ds, H0 // ds), interpolation=cv2.INTER_AREA)
                prev_gray_small = cv2.resize(prev_gray, (W0 // ds, H0 // ds), interpolation=cv2.INTER_AREA)
            else:
                gray_small, prev_gray_small = gray, prev_gray

            flow_small = None
            if args.flow_method == "farneback":
                try:
                    flow_small = cv2.calcOpticalFlowFarneback(
                        prev_gray_small, gray_small,
                        None, 0.5, 3, 15, 3, 5, 1.1, 0
                    )
                except Exception as _e:
                    print(f"[flow][warn] Farneback failed: {_e}; skipping this frame")
                    flow_small = None
            else:
                # DIS flow with safety checks for size changes
                try:
                    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
                    if prev_gray_small is None or prev_gray_small.shape != gray_small.shape:
                        if prev_gray_small is None:
                            print("[flow] init prev_gray_small")
                        else:
                            print(f"[flow][skip] size change {getattr(prev_gray_small, 'shape', None)} -> {gray_small.shape}")
                        flow_small = None
                    else:
                        flow_small = dis.calc(prev_gray_small, gray_small, None)
                except Exception as _e:
                    print(f"[flow][warn] DIS failed: {_e}; skipping this frame")
                    flow_small = None

            if flow_small is not None:
                if ds > 1:
                    flow = cv2.resize(flow_small, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    flow *= ds
                else:
                    flow = flow_small.astype(np.float32)

                curr_styled01 = out01.detach().permute(1, 2, 0).cpu().numpy().astype("float32")
                prev_warp01 = _warp_with_flow(prev_styled01, flow)

                a = float(max(0.0, min(1.0, args.flow_alpha)))
                fused01 = (a * curr_styled01 + (1.0 - a) * prev_warp01).clip(0.0, 1.0).astype("float32")
                out01 = torch.from_numpy(fused01).permute(2, 0, 1)
                last_flow = flow
            else:
                last_flow = None
        else:
            last_flow = None

        # Update flow caches
        prev_gray = gray
        prev_styled01 = out01.permute(1, 2, 0).cpu().numpy().astype("float32")

        # ---- LAB smoothing (lightness/chroma) ----
        out_img = transforms.ToPILImage()(out01)
        if smooth_lightness or smooth_chroma:
            lab = out_img.convert("LAB")
            lab_np = np.array(lab, dtype=np.uint8).astype(np.float32)
            L = lab_np[..., 0]
            aC = lab_np[..., 1]
            bC = lab_np[..., 2]

            if smooth_lightness:
                if (prev_L is None) or (prev_L.shape != L.shape):
                    prev_L = L.copy()
                try:
                    L_sm = smooth_alpha * L + (1.0 - smooth_alpha) * prev_L
                except Exception:
                    # In case something still went wrong (e.g., unexpected shape), fall back and reset cache
                    prev_L = L.copy()
                    L_sm = L
                prev_L = L_sm
                lab_np[..., 0] = np.clip(L_sm, 0, 255)

            if smooth_chroma:
                if (prev_aC is None) or (prev_aC.shape != aC.shape):
                    prev_aC = aC.copy()
                if (prev_bC is None) or (prev_bC.shape != bC.shape):
                    prev_bC = bC.copy()
                try:
                    a_sm = chroma_alpha * aC + (1.0 - chroma_alpha) * prev_aC
                    b_sm = chroma_alpha * bC + (1.0 - chroma_alpha) * prev_bC
                except Exception:
                    prev_aC, prev_bC = aC.copy(), bC.copy()
                    a_sm, b_sm = aC, bC
                prev_aC, prev_bC = a_sm, b_sm
                lab_np[..., 1] = np.clip(a_sm, 0, 255)
                lab_np[..., 2] = np.clip(b_sm, 0, 255)

            out_img = Image.fromarray(lab_np.astype(np.uint8), mode="LAB").convert("RGB")

        # ---- Original-image blend (uniform or motion-adaptive) ----
        out01_smooth = to_tensor(out_img).unsqueeze(0)
        # ---- Optional region mask composite (foreground/background) ----
        mask_used = False
        # Resolve mask source: explicit --mask takes priority, else look in --mask_dir/mask_%04d.png
        mask_file = getattr(args, "mask", None)
        if not mask_file and getattr(args, "mask_dir", None):
            try:
                stem_num = frame_path.stem.split("_")[-1]  # e.g., 0001 from frame_0001.jpg
                cand = Path(args.mask_dir) / f"mask_{stem_num}.png"
                if cand.exists():
                    mask_file = str(cand)
            except Exception:
                mask_file = None
        if mask_file:
            try:
                # Align mask to requested stage
                if args.fit_mask_to == "output":
                    ref_H, ref_W = out01_smooth.shape[-2], out01_smooth.shape[-1]
                else:
                    ref_H, ref_W = x_orig01.shape[-2], x_orig01.shape[-1]
                fpx = _pct_to_px(getattr(args, "mask_feather_pct", 0.0) or 0.0, ref_H)
                if getattr(args, "mask_feather", 0) and getattr(args, "mask_feather", 0) > 0:
                    fpx = max(fpx, int(args.mask_feather))
                try:
                    print(f"[mask] using {Path(mask_file).name} -> fit_to={args.fit_mask_to} target={ref_W}x{ref_H} invert={bool(getattr(args,'mask_invert',False))} feather_px={int(fpx)} (autofix={bool(getattr(args,'mask_autofix',True))})")
                except Exception:
                    pass
                alpha = _load_mask_fit(
                    mask_file,
                    (ref_H, ref_W),
                    bool(getattr(args, "mask_invert", False)),
                    int(fpx),
                    bool(getattr(args, "mask_autofix", True)),
                    bool(getattr(args, "mask_force_transpose", False))
                )  # HxWx1
                # Debug: save fitted alpha as 8-bit PNG for inspection
                if bool(getattr(args, "mask_debug_alpha", False)) or bool(getattr(args, "mask_debug_overlay", False)):
                    try:
                        debug_dir = (frames_dir.parent / "debug")
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        a_u8 = (alpha.squeeze(-1) * 255.0).clip(0, 255).astype(np.uint8)
                        Image.fromarray(a_u8, mode="L").save(debug_dir / f"mask_fit_{idx:04d}.png")
                        print(f"[mask][debug] wrote {debug_dir}/mask_fit_{idx:04d}.png")
                    except Exception as _e:
                        print(f"[mask][debug][WARN] could not save mask_fit: {_e}")


                # Prepare source tensors as numpy
                S = out01_smooth.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
                # Ensure original matches current size
                if (x_orig01.shape[-2], x_orig01.shape[-1]) != (ref_H, ref_W):
                    x_orig01_rs = F.interpolate(x_orig01, size=(ref_H, ref_W), mode="bilinear", align_corners=False)
                else:
                    x_orig01_rs = x_orig01
                O = x_orig01_rs.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

                # Optional debug overlay (original + red tint where alpha>0)
                # (Moved below after S and O are defined)

                if args.composite_mode == "keep":
                    C = (alpha * S + (1.0 - alpha) * O).clip(0.0, 1.0)
                else:  # 'replace' => stylize unmasked region
                    C = ((1.0 - alpha) * S + alpha * O).clip(0.0, 1.0)

                out01_smooth = torch.from_numpy(C).permute(2, 0, 1).unsqueeze(0)
                mask_used = True
            except Exception as _e:
                print(f"[mask][WARN] mask composite skipped: {_e}")
        if out01_smooth.shape[-2:] != (H0, W0):
            x_orig01_rs = F.interpolate(x_orig01, size=out01_smooth.shape[-2:],
                                        mode="bilinear", align_corners=False)
        else:
            x_orig01_rs = x_orig01

        # Optional debug overlay (original + red tint where alpha>0)
        S = out01_smooth.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        O = x_orig01_rs.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        if bool(getattr(args, "mask_debug_overlay", False)):
            try:
                debug_dir = (frames_dir.parent / "debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                base_u8 = (O * 255.0).clip(0, 255).astype(np.uint8)
                tint = np.zeros_like(base_u8, dtype=np.uint8)
                tint[..., 0] = 255  # red
                a3 = np.repeat(alpha, 3, axis=2)
                overlay = (base_u8.astype(np.float32) * (1.0 - 0.35 * a3) + tint.astype(np.float32) * (0.35 * a3)).clip(0, 255).astype(np.uint8)
                Image.fromarray(overlay).save(debug_dir / f"overlay_{idx:04d}.jpg", quality=92)
                print(f"[mask][debug] wrote {debug_dir}/overlay_{idx:04d}.jpg")
            except Exception as _e:
                print(f"[mask][debug][WARN] could not save overlay: {_e}")

        if (getattr(args, "motion_blend", False) and (last_flow is not None) and not mask_used):
            mag = np.sqrt(last_flow[..., 0] ** 2 + last_flow[..., 1] ** 2).astype(np.float32)
            m = np.clip(mag / MOTION_NORM, 0.0, 1.0)
            m = cv2.GaussianBlur(m, (0, 0), GAUSS_SIGMA)

            base_alpha = float(blend)
            min_alpha = float(MIN_ALPHA)
            max_alpha = base_alpha
            alpha_map = max_alpha - (max_alpha - min_alpha) * m
            alpha_map3 = np.repeat(alpha_map[..., None], 3, axis=2)

            s_np = out01_smooth.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            o_np = x_orig01_rs.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            fused = (alpha_map3 * s_np + (1.0 - alpha_map3) * o_np).clip(0.0, 1.0).astype(np.float32)
            out01_final = torch.from_numpy(fused).permute(2, 0, 1)
        else:
            #  Uniform global blend still applies after region composite (if any)
            if 0.0 <= blend < 1.0:
                out01_final = (blend * out01_smooth + (1.0 - blend) * x_orig01_rs).clamp(0, 1).squeeze(0)
            else:
                out01_final = out01_smooth.squeeze(0)

        out_img = to_pil(out01_final)

        t2 = time.perf_counter()
        print(f"[frame][{idx}] postproc  dt={t2 - t1:.3f}s  rss={_rss_mb():.1f} MB")

        # ---- Save ----
        save_as_jpg = (image_ext_out.lower() == "jpg")
        if image_mode and idx in save_map:
            out_path = Path(save_map[idx])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_as_jpg = out_path.suffix.lower() in [".jpg", ".jpeg"]
        else:
            idx_str = frame_path.stem.split("_")[-1]
            out_stem = f"{output_prefix}_{idx_str}"
            out_path = (frames_dir / out_stem).with_suffix(".jpg" if save_as_jpg else ".png")

        print(f"[save][plan] writing to: {out_path}")
        try:
            if save_as_jpg:
                out_img.save(out_path, format="JPEG", quality=int(jpeg_quality))
            else:
                out_img.save(out_path)
            print(f"[save] -> {out_path}")
        except Exception as e:
            print(f"[save][ERROR] failed writing {out_path}: {e}")
            raise

        if idx % 10 == 0 or idx == 1:
            print(f"Styled {idx}/{len(frame_files)} frames...")


# ---------------------------
# Assemble video
# ---------------------------
def assemble_video(frames_dir: Path, output_video: Path,
                   in_fps: Optional[int], out_fps: Optional[int], prefix: str):
    """
    Assemble an image sequence into a video.
    - in_fps controls how ffmpeg interprets the image sequence timestamps.
    - out_fps (if set) requests an encoder output rate; frames will be duplicated/dropped to match.
    """
    fr_in = f"-framerate {in_fps}" if in_fps else ""
    fr_out = f"-r {out_fps}" if out_fps else ""
    jpgs = sorted(frames_dir.glob(f"{prefix}_*.jpg"))
    pngs = sorted(frames_dir.glob(f"{prefix}_*.png"))

    if jpgs:
        pattern = frames_dir / f"{prefix}_%04d.jpg"
    elif pngs:
        pattern = frames_dir / f"{prefix}_%04d.png"
    else:
        print("No styled frames found (jpg/png).")
        sys.exit(1)

    print(f'>> assembling from {pattern.suffix} sequence…')
    cmd = f'ffmpeg -y {fr_in} -i "{pattern}" {fr_out} -c:v libx264 -pix_fmt yuv420p "{output_video}"'
    sh(cmd)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract → Style → Assemble (with temporal smoothing)")
    ap.add_argument("--input_video", required=False, default=None)
    ap.add_argument("--output_video", required=False, default=None)
    ap.add_argument("--model", required=False,
                    help="Path to a PyTorch checkpoint (.pth/.pt) when using transformer/reconet; not required for --model_type magenta.")
    ap.add_argument("--work_dir", default="./_work")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--pre_fps", type=int, default=None,
                    help="Reduce the frame rate of the INPUT VIDEO before extraction by first writing a temp video at this FPS (preprocess step).")
    ap.add_argument("--scale", type=int, default=None,
                    help="Long-side resolution (e.g., 720, 1080). Keep original if omitted.")
    ap.add_argument("--canvas", type=str, default=None,
                    help="If set like 'WxH' (e.g., 1920x1080), extract frames to a fixed canvas using scale(...:force_original_aspect_ratio=decrease)+pad to exactly WxH. Ensures constant dimensions for optical flow.")
    ap.add_argument("--image_ext", choices=["png", "jpg"], default="png", help="Intermediate extraction format.")
    ap.add_argument("--jpeg_quality", type=int, default=85, help="JPG quality when saving styled frames (1–95).")
    ap.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (2 = every other frame).")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit number of frames for quick tests.")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--gpu_memory_limit", type=int, default=32000,
                    help="GPU memory limit in MB (default: 32000 = 32GB). Set lower to share GPU between processes.")
    ap.add_argument("--inference_res", type=int, default=0,
                    help="If >0, downscale the frame's long side to this before model inference, then upsample back for post-processing. Helps avoid OOM on very large images.")

    ap.add_argument("--io_preset",
                    choices=["auto", "imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default="auto",
                    help="I/O preset for the primary model. Use 'auto' to pick based on backend: transformer->imagenet_255, torch7->caffe_bgr, magenta->imagenet_01, reconet->imagenet_01. Use 'raw_01' for models expecting 0-1 input/output.")

    ap.add_argument("--input_image", type=str, help="Path to a single input image.")
    ap.add_argument("--output_image", type=str, help="Path to a single output image.")
    ap.add_argument("--input_dir", type=str, help="Directory of input images for batch processing.")
    ap.add_argument("--output_dir", type=str, help="Output directory for batch image processing.")
    ap.add_argument("--pattern", type=str, default=None, help="Glob pattern for --input_dir (e.g., *.jpg, *.png). If omitted, it defaults to *.{image_ext}.")
    ap.add_argument("--keep_ext", action="store_true", help="Preserve each input file's extension in batch mode.")
    ap.add_argument("--output_suffix", type=str, default="",
                    help="Optional suffix to add before the extension in batch mode.")
    ap.add_argument("--output_prefix", type=str, default="styled_frame", help="Base name for styled outputs in batch mode (e.g., styled_frame). Used when input files are numbered like frame_0001.* so the outputs align with downstream assemblers.")

    # Flicker control .
    ap.add_argument("--smooth_lightness", action="store_true", default=True,
                    help="LAB Lightness EMA (on by default).")
    ap.add_argument("--no-smooth_lightness", action="store_false", dest="smooth_lightness")
    ap.add_argument("--smooth_alpha", type=float, default=0.7, help="EMA alpha 0–1 (higher = more current frame).")
    ap.add_argument("--smooth_chroma", action="store_true", default=False, help="LAB a/b EMA as well.")
    ap.add_argument("--chroma_alpha", type=float, default=0.85, help="EMA alpha for a/b when chroma smoothing is on.")

    ap.add_argument("--blend", type=float, default=1.0,
                    help="Uniform blend with original: 0.0=original, 1.0=fully stylized.")

    # Region mask compositing (e.g., sky/non-sky)
    ap.add_argument("--mask", type=str, default=None, help="Path to an 8-bit mask image (255 = process region).")
    ap.add_argument("--mask_invert", action="store_true", help="Invert mask so 255 selects background instead.")
    ap.add_argument("--mask_feather", type=int, default=0, help="Feather radius in pixels (applied after resize).")
    ap.add_argument("--mask_dir", type=str, default=None,
                    help="Directory of per-frame masks named mask_%04d.png (255 = process region). When set (and no --mask), frames without a corresponding mask are rejected (hard error if none exist; warning if some are missing).")
    ap.add_argument("--mask_feather_pct", type=float, default=0.0, help="Feather radius as percent of image height.")
    ap.add_argument("--mask_autofix", action="store_true", default=True,
                    help="Auto-correct 90°-transposed masks to match the target orientation before resize.")
    ap.add_argument("--mask_force_transpose", action="store_true",
                    help="Force 90° transpose of masks before resize (overrides/augments --mask_autofix).")
    ap.add_argument("--mask_debug_overlay", action="store_true",
                    help="Save per-frame overlay of mask alignment to _work/debug/overlay_%04d.jpg")
    ap.add_argument("--mask_debug_alpha", action="store_true",
                    help="Save the fitted mask (after autofix/transpose, resize, invert, and feather) as 8-bit PNG to _work/debug/mask_fit_%04d.png")
    ap.add_argument("--fit_mask_to", choices=["input", "output"], default="input", help="Resize mask to match input or output dimensions before blending.")
    ap.add_argument("--composite_mode", choices=["keep", "replace"], default="keep", help="keep = stylize masked region only; replace = stylize unmasked region.")

    # Optical flow EMA
    ap.add_argument("--flow_ema", action="store_true", default=False,
                    help="Enable flow-guided EMA (warp prev stylized, then fuse).")
    ap.add_argument("--flow_alpha", type=float, default=0.85,
                    help="Flow EMA alpha (higher = more current frame).")
    ap.add_argument("--flow_method", choices=["farneback", "dis"], default="dis",
                    help="Optical flow backend.")
    ap.add_argument("--flow_downscale", type=int, default=1,
                    help="Downscale factor for flow (1=no downscale; 2/3=faster).")

    # Model type
    ap.add_argument("--model_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default="transformer",
                    help="Select backend: PyTorch (transformer/reconet), TensorFlow-Hub (magenta), or Torch7 via OpenCV DNN (torch7). If --model ends with .t7, the torch7 backend is auto-selected.")
    ap.add_argument("--model_b", type=str, default=None,
                    help="Optional second model checkpoint to blend with the primary output (.pth/.pt or .t7).")
    ap.add_argument("--model_b_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the second model. Defaults to auto (.t7 -> torch7) or the same as --model_type.")
    ap.add_argument("--io_preset_b",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the second model (defaults to --io_preset).")
    ap.add_argument("--model_c", type=str, default=None, help="Path to third model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_c_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the third model. Defaults to auto (.t7 -> torch7) or same as --model_type.")
    ap.add_argument("--io_preset_c",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the third model (defaults to --io_preset).")
    ap.add_argument("--model_d", type=str, default=None, help="Path to fourth model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_d_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the fourth model. Defaults to auto (.t7 -> torch7) or same as --model_type.")
    ap.add_argument("--io_preset_d",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the fourth model (defaults to --io_preset).")

    # Model E
    ap.add_argument("--model_e", type=str, default=None, help="Path to fifth model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_e_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the fifth model.")
    ap.add_argument("--io_preset_e",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the fifth model (defaults to --io_preset).")
    ap.add_argument("--magenta_style_e", type=str, default=None,
                    help="Style image for fifth model when --model_e_type magenta.")

    # Model F
    ap.add_argument("--model_f", type=str, default=None, help="Path to sixth model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_f_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the sixth model.")
    ap.add_argument("--io_preset_f",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the sixth model (defaults to --io_preset).")
    ap.add_argument("--magenta_style_f", type=str, default=None,
                    help="Style image for sixth model when --model_f_type magenta.")

    # Model G
    ap.add_argument("--model_g", type=str, default=None, help="Path to seventh model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_g_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the seventh model.")
    ap.add_argument("--io_preset_g",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the seventh model (defaults to --io_preset).")
    ap.add_argument("--magenta_style_g", type=str, default=None,
                    help="Style image for seventh model when --model_g_type magenta.")

    # Model H
    ap.add_argument("--model_h", type=str, default=None, help="Path to eighth model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_h_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the eighth model.")
    ap.add_argument("--io_preset_h",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255", "raw_01"],
                    default=None,
                    help="I/O preset for the eighth model (defaults to --io_preset).")
    ap.add_argument("--magenta_style_h", type=str, default=None,
                    help="Style image for eighth model when --model_h_type magenta.")

    ap.add_argument("--blend_models_weights", type=str, default=None,
                    help="Comma-separated weights for blending models (e.g., '0.25,0.25,0.25,0.25'). Must sum to 1.0. Defaults to equal weights.")
    ap.add_argument("--blend_models_lab", action="store_true", default=False,
                    help="Blend in LAB: take Lightness from model A and chroma (a/b) from model B,C,D.")
    ap.add_argument("--blend_models_lab_weights", type=str, default=None,
                    help="If --blend_models_lab, weights for L (from A) and a/b (from B,C,D) in format 'wL,wab' (e.g., '0.5,0.5').")

    # Region-based spatial blending
    ap.add_argument("--region_mode", type=str, default=None,
                    choices=["grid", "diagonal", "voronoi", "fractal", "radial", "waves", "spiral", "concentric", "random"],
                    help="Spatial region pattern for blending models. Each region uses a different model instead of weighted blend.")
    ap.add_argument("--region_count", type=int, default=None,
                    help="Number of regions to create (defaults to number of active models).")
    ap.add_argument("--region_sizes", type=str, default=None,
                    help="Relative sizes for each region (voronoi mode only). Format: '1,1,1,0.2' makes last "
                         "region ~5x smaller. '1,1,1,1,0.1' gives ~5%% for last region. Higher = larger region.")
    ap.add_argument("--region_seed", type=str, default=None,
                    help="Random seed for region generation. Use 'random' or omit for random, or an integer for reproducibility.")
    ap.add_argument("--region_feather", type=int, default=20,
                    help="Edge softness in pixels for region boundaries (default: 20).")
    ap.add_argument("--region_assignment", type=str, default="random",
                    choices=["sequential", "random", "weighted"],
                    help="How to assign models to regions: sequential (A,B,C,D,A,...), random, or weighted by blend_models_weights.")
    ap.add_argument("--region_original", type=float, default=0.0,
                    help="Probability (0.0-1.0) that a region stays unstyled (shows original frame). E.g., 0.25 = ~25%% of regions unstyled.")
    ap.add_argument("--region_rotate", type=float, default=0.0,
                    help="Degrees to rotate the region pattern per frame. E.g., 2.0 = smooth 2° rotation per frame. Use with --region_seed fixed for consistency.")
    ap.add_argument("--region_blend_spec", type=str, default=None,
                    help="Per-region model blend specification. Format: 'A|B|C|D' for one model per region, "
                         "'A+B|C+D' for 50/50 blends, 'A:0.7+B:0.3|C' for weighted blends. "
                         "Use 'O' for original (unstyled). Regions cycle through specs.")
    ap.add_argument("--region_scales", type=str, default=None,
                    help="Per-region scale factors (resolution). Format: '1.0,0.5,0.25' - regions cycle through scales. "
                         "1.0 = full resolution, 0.5 = half resolution (styled at lower res, upscaled).")
    ap.add_argument("--region_optimize", action="store_true", default=False,
                    help="Enable crop-based region optimization. Only styles the pixels needed for each region "
                         "instead of full frames. Can be 2-4x faster with multiple regions.")
    ap.add_argument("--region_padding", type=int, default=64,
                    help="Padding pixels around region crops for convolution context (default: 64). "
                         "Only used with --region_optimize.")
    ap.add_argument("--blend_animate", type=str, default=None,
                    help="Animate blend weights harmonically. Format: 'period,waveform,phase,min,max'. "
                         "Examples: '120' (120-frame sine cycle), '60,triangle' (60-frame triangle wave), "
                         "'90,sine,45,0.2,0.8' (90 frames, 45° phase, 0.2-0.8 opacity range). "
                         "Waveforms: sine, triangle, sawtooth, sawtooth_down, square.")
    ap.add_argument("--blend_animate_regions", type=str, default=None,
                    help="Per-region blend animation specs. Format: 'spec1|spec2|spec3|...' where each spec "
                         "follows --blend_animate format. Use 'static' for no animation on a region. "
                         "Example: '120,sine|60,triangle|static|90,sawtooth'")
    ap.add_argument("--scale_animate", type=str, default=None,
                    help="Animate region scales (resolution) harmonically. Format: 'period,waveform,phase,min,max'. "
                         "Examples: '60' (60-frame sine cycle, 0.5-1.0 scale), '60,triangle,0,0.3,0.8'. "
                         "Creates pulsing detail effect as regions grow/shrink in resolution.")
    ap.add_argument("--scale_animate_regions", type=str, default=None,
                    help="Per-region scale animation specs. Format: 'spec1|spec2|spec3|...' where each spec "
                         "follows --scale_animate format. Use 'static' for no animation on a region. "
                         "Example: '60,sine,0,0.5,1.0|30,triangle|static|90,sawtooth'")
    ap.add_argument("--region_morph", type=str, default=None,
                    help="Enable organic morphing/weaving region boundaries. Makes regions animate like "
                         "blobs or tentacles. Format: 'mode' or 'speed,amplitude,frequency,mode'. "
                         "Modes: blob (organic), tentacle (elongated), wave (sinusoidal), pulse (radial). "
                         "Examples: 'blob', 'tentacle', '1.5,0.2,4.0,blob', '2.0,0.1,3.0,tentacle'. "
                         "Speed=animation speed, amplitude=how far boundaries move (0.1=10%% of frame), "
                         "frequency=detail level (higher=more tentacles).")

    ap.add_argument("--magenta_style_b", type=str, default=None,
                    help="Optional style image for second model when --model_b_type magenta (defaults to --magenta_style if omitted).")
    ap.add_argument("--magenta_style_c", type=str, default=None,
                    help="Style image for third model when --model_c_type magenta (defaults to --magenta_style).")
    ap.add_argument("--magenta_style_d", type=str, default=None,
                    help="Style image for fourth model when --model_d_type magenta (defaults to --magenta_style).")

    # Magenta backend options
    ap.add_argument("--magenta_style", type=str, default=None,
                    help="Path to the style image to use with Magenta backend.")
    ap.add_argument("--magenta_model_root", type=str, default="/app/models/magenta",
                    help="Directory that contains a TF-Hub model subdir.")
    ap.add_argument("--magenta_tile", type=int, default=256, help="Tile size for Magenta tiling.")
    ap.add_argument("--magenta_overlap", type=int, default=32, help="Overlap (pixels) for Magenta tiles.")
    ap.add_argument("--magenta_target_res", type=int, default=None,
                    help="Optional long-side target resolution before tiling.")

    # Motion-adaptive blend
    ap.add_argument("--motion_blend", action="store_true", default=False,
                    help="Per-pixel alpha from flow magnitude (high motion = more original).")

    ap.add_argument("--clean_frames", action="store_true",
                    help="Delete existing frame_*.{png|jpg} and styled_frame_*.{png|jpg} before extracting.")
    ap.add_argument("--clean_work_dir", action="store_true", default=False,
                    help="Remove the unique work directory after job completion (image modes only). Default: keep for debugging.")

    args = ap.parse_args()

    # Set GPU memory limit early, before any model loading
    set_gpu_memory_limit(args.gpu_memory_limit)

    def _parse_canvas(s: Optional[str]) -> Optional[tuple[int,int]]:
        if not s:
            return None
        m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", str(s))
        if not m:
            print(f"[canvas][ERROR] Expected WxH (e.g., 1920x1080), got: {s}")
            sys.exit(2)
        w, h = int(m.group(1)), int(m.group(2))
        if w <= 0 or h <= 0:
            print(f"[canvas][ERROR] Invalid canvas size: {w}x{h}")
            sys.exit(2)
        return (w, h)

    canvas_wh = _parse_canvas(getattr(args, "canvas", None))

    # Normalize blank I/O args to None (so mode detection is unambiguous)
    for k in ("input_video", "output_video", "input_image", "output_image", "input_dir", "output_dir"):
        v = getattr(args, k, None)
        if v is not None and str(v).strip() == "":
            setattr(args, k, None)

    # Default pattern to *.{image_ext} when batch image mode uses implicit extension
    if getattr(args, "pattern", None) in (None, ""):
        args.pattern = f"*.{args.image_ext}"

    # Extra safety for OpenCV DNN stability in some environments
    os.environ.setdefault('OPENCV_OPENCL_DEVICE', 'disabled')
    os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

    # Decide mode
    image_mode_single = bool(getattr(args, "input_image", None)) and bool(getattr(args, "output_image", None))
    image_mode_batch = bool(getattr(args, "input_dir", None)) and bool(getattr(args, "output_dir", None))
    video_mode = bool(getattr(args, "input_video", None)) and bool(getattr(args, "output_video", None))

    if (image_mode_single or image_mode_batch) and video_mode:
        print("Provide exactly one of: (input_video & output_video) OR (input_image & output_image) OR (input_dir & output_dir).")
        sys.exit(2)
    if not (image_mode_single or image_mode_batch or video_mode):
        print("Specify (input_video & output_video) OR (input_image & output_image) OR (input_dir & output_dir).")
        sys.exit(2)

    # Require --model except for Magenta backend
    if args.model_type != "magenta":
        if not getattr(args, "model", None):
            print("[error] --model is required unless --model_type magenta")
            sys.exit(2)
    else:
        if not getattr(args, "magenta_style", None):
            print("[magenta][ERROR] --magenta_style is required when --model_type magenta")
            sys.exit(2)

    # Warn: motion features are video-only
    if image_mode_single or image_mode_batch:
        if getattr(args, "motion_blend", False):
            print("[warn] --motion_blend ignored in image mode.")
        if getattr(args, "flow_ema", False):
            print("[warn] --flow_ema ignored in image mode.")

    # Use unique work directory per job to prevent cross-contamination
    base_work_dir = Path(args.work_dir).resolve()
    if image_mode_single or image_mode_batch:
        # Create unique subdirectory for this job using short UUID
        job_id = uuid.uuid4().hex[:8]
        work_dir = base_work_dir / f"job_{job_id}"
        print(f"[work_dir] Using isolated work directory: {work_dir}")
    else:
        work_dir = base_work_dir
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing staged frames depending on mode
    # IMPORTANT: In image batch mode the input_dir may be the same as frames_dir.
    # Do not delete `frame_*` in that case or we nuke the inputs before we can read them.
    def _purge(patterns: list[str]):
        for pat in patterns:
            for p in frames_dir.glob(pat):
                p.unlink(missing_ok=True)

    if video_mode or image_mode_single:
        # We are about to (re)create frame_* via extraction/staging; safe to clear both.
        _purge(["frame_*.png", "frame_*.jpg", "frame_*.jpeg",
                "styled_frame_*.png", "styled_frame_*.jpg", "styled_frame_*.jpeg"])
    else:
        # image_mode_batch:
        # If the staging dir is NOT the same as the input_dir, it's safe to clear stale staged frames.
        input_dir_path = Path(args.input_dir).resolve() if getattr(args, "input_dir", None) else None
        if input_dir_path and input_dir_path != frames_dir.resolve():
            _purge(["frame_*.png", "frame_*.jpg", "frame_*.jpeg"])
        # Always remove old styled outputs.
        _purge(["styled_frame_*.png", "styled_frame_*.jpg", "styled_frame_*.jpeg"])

    if args.model_type != "magenta":
        model_path = Path(args.model).resolve()
    else:
        model_path = None

    if args.model_type != "magenta" and model_path is not None and model_path.suffix.lower() == ".t7":
        print(f"[auto] Detected .t7 checkpoint ({model_path.name}); switching backend to torch7 (OpenCV DNN).")
        args.model_type = "torch7"

    # Resolve I/O preset if requested
    IO_PRESETS = {
        "transformer": "imagenet_255",
        "torch7": "caffe_bgr",
        "magenta": "imagenet_01",
        "reconet": "imagenet_01",
    }
    if getattr(args, "io_preset", "auto") == "auto":
        resolved = IO_PRESETS.get(args.model_type, "imagenet_01")
        print(f"[auto] io_preset resolved to '{resolved}' for backend '{args.model_type}'")
        args.io_preset = resolved

    save_map = {}
    if video_mode:
        input_video = Path(args.input_video).resolve()
        output_video = Path(args.output_video).resolve()
        print(
            f"[cfg] scale={args.scale}  fps={args.fps}  pre_fps={getattr(args, 'pre_fps', None)}  image_ext={args.image_ext}")
        if canvas_wh:
            print(f"[cfg] canvas={canvas_wh[0]}x{canvas_wh[1]} (scale+pad)")

        if getattr(args, "pre_fps", None):
            pre_fps = int(args.pre_fps)
            tmp_pre = (work_dir / f"_pre_fps_{pre_fps}.mp4").resolve()
            print(f"[pre-fps] Writing temp video at {pre_fps} fps: {tmp_pre}")
            cmd = f'ffmpeg -y -i "{input_video}" -filter:v "fps={pre_fps}" -c:v libx264 -pix_fmt yuv420p "{tmp_pre}"'
            print(cmd)
            sh(cmd)
            input_video = tmp_pre

        extract_fps = None if getattr(args, "pre_fps", None) else args.fps
        if getattr(args, "pre_fps", None) and args.fps:
            print(
                f"[note] --pre_fps is set; ignoring --fps during extraction. Assembly will interpret frames at pre_fps={args.pre_fps} and write output at fps={args.fps}.")
        extract_frames(input_video, frames_dir, extract_fps, args.scale, args.image_ext, args.jpeg_quality, canvas_wh)
    elif image_mode_single:
        src = Path(args.input_image).resolve()
        ext = src.suffix.lower()
        dst = frames_dir / f"frame_0001{ext}"
        print(f"[stage][single] staging {src} -> {dst}")
        try:
            pil_norm = _get_image_with_exif_pil(str(src))
            if ext in [".jpg", ".jpeg"]:
                pil_norm.save(dst, format="JPEG", quality=max(1, min(95, int(args.jpeg_quality))))
            else:
                pil_norm.save(dst)
        except Exception as e:
            print(f"[stage][WARN] EXIF-normalize failed ({e}); copying instead")
            shutil.copy2(src, dst)
        if not dst.exists():
            print(f"[stage][ERROR] expected staged file missing: {dst}")
        else:
            try:
                print(f"[stage][ok] {dst} exists ({dst.stat().st_size} bytes)")
            except Exception:
                print(f"[stage][ok] {dst} exists")
        save_map[1] = str(Path(args.output_image).resolve())
    elif image_mode_batch:
        in_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not in_files:
            print(f"No files matched: {args.input_dir}/{args.pattern}")
            sys.exit(2)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(in_files, start=1):
            src = Path(f).resolve()
            ext = src.suffix.lower()
            dst = frames_dir / f"frame_{i:04d}{ext}"
            try:
                pil_norm = _get_image_with_exif_pil(str(src))
                if ext in [".jpg", ".jpeg"]:
                    pil_norm.save(dst, format="JPEG", quality=max(1, min(95, int(args.jpeg_quality))))
                else:
                    pil_norm.save(dst)
            except Exception as e:
                print(f"[stage][WARN] EXIF-normalize failed for {src} ({e}); copying instead")
                shutil.copy2(src, dst)
            print(f"[stage][batch] {src} -> {dst}")
            base = src.stem
            out_ext = ext if args.keep_ext else (".jpg" if args.image_ext.lower() == "jpg" else ".png")
            suffix = args.output_suffix or ""
            # If inputs are numbered like frame_0001, honor that index and rename to {output_prefix}_0001.*
            m = re.match(r"^frame_(\d+)$", base)
            if m:
                idx_str = m.group(1)
                out_stem = f"{args.output_prefix}_{idx_str}"
            else:
                out_stem = f"{base}{suffix}"
            save_map[i] = str((Path(args.output_dir) / f"{out_stem}{out_ext}").resolve())

    # 2) Style
    style_frames(
        args,
        frames_dir, model_path, output_prefix="styled_frame",
        image_ext_out=args.image_ext, device_str=args.device,
        threads=args.threads, stride=args.stride, max_frames=args.max_frames,
        smooth_lightness=args.smooth_lightness, smooth_alpha=args.smooth_alpha,
        jpeg_quality=args.jpeg_quality, io_preset=args.io_preset,
        smooth_chroma=args.smooth_chroma, chroma_alpha=args.chroma_alpha,
        blend=args.blend,
        image_mode=(image_mode_single or image_mode_batch),
        save_map=save_map,
    )

    # 3) Assemble (video only)
    if video_mode:
        if getattr(args, "pre_fps", None):
            in_fps = int(args.pre_fps)
            out_fps = int(args.fps) if args.fps else in_fps
            if args.fps:
                print(
                    f"[assemble] input framerate={in_fps} (sequence pacing), output fps={out_fps} (dup frames as needed).")
            else:
                print(f"[assemble] input framerate={in_fps}; output fps not set (keeps {in_fps}).")
        else:
            in_fps = int(args.fps) if args.fps else None
            out_fps = None
        assemble_video(frames_dir, output_video, in_fps, out_fps, prefix="styled_frame")
        print(f"\n✅ Done. Styled video at: {output_video}")
        print(f"   Frames in: {frames_dir}")
    else:
        print("\n✅ Image mode complete.")
        print(f"   Frames staged in: {frames_dir}")

        try:
            if image_mode_single:
                outp = Path(args.output_image).resolve()
                print(f"✅ Wrote stylized image to: {outp}")
                print(f"   Exists? {outp.exists()}")
            elif image_mode_batch:
                out_dir = Path(args.output_dir).resolve()
                planned = len(save_map)
                written = 0
                for v in save_map.values():
                    p = Path(v)
                    if p.exists():
                        written += 1
                print(f"✅ Wrote {written}/{planned} images into: {out_dir}")
        except Exception as e:
            print(f"[report][WARN] Could not verify outputs: {e}")

    if args.clean_frames:
        for p in frames_dir.glob("frame_*.png"): p.unlink(missing_ok=True)
        for p in frames_dir.glob("frame_*.jpg"): p.unlink(missing_ok=True)
        for p in frames_dir.glob("styled_frame_*.png"): p.unlink(missing_ok=True)
        for p in frames_dir.glob("styled_frame_*.jpg"): p.unlink(missing_ok=True)

    # Optionally clean up unique work directory for image modes
    if args.clean_work_dir and (image_mode_single or image_mode_batch) and work_dir != base_work_dir:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"[cleanup] Removed isolated work directory: {work_dir}")
        except Exception as e:
            print(f"[cleanup][WARN] Could not remove work directory: {e}")
    elif (image_mode_single or image_mode_batch) and work_dir != base_work_dir:
        print(f"[work_dir] Kept isolated work directory (use --clean_work_dir to remove): {work_dir}")


if __name__ == "__main__":
    main()