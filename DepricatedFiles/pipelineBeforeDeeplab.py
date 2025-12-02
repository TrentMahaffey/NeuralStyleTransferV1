#!/usr/bin/env python3
import argparse, subprocess, sys
import os, glob, shutil
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import faulthandler, gc
import time
import resource

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


def _try_import_tf():
    global tf, hub
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_hub as hub  # type: ignore
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
                   img_ext: str, jpeg_quality: int):
    frames_dir.mkdir(parents=True, exist_ok=True)

    vf_parts: List[str] = []
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

    if use_magenta:
        model = None
        print("[magenta] Using TensorFlow-Hub backend; style image:", args.magenta_style)
        if not args.magenta_style:
            raise ValueError("--magenta_style is required when --model_type magenta")
    elif use_torch7:
        model = None
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
        else:
            TransformerNet = _get_transformer_net()
            model = TransformerNet().to(device)
        _load_checkpoint_compat(model, str(model_path))
        model.eval()

    print(f"[backend] A: type={args.model_type} path={model_path if model_path else '(n/a)'}  device={device}")

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
    use_c = bool(getattr(args, "model_c", None))
    c_type_resolved = None
    ipc_resolved = None
    if use_c:
        model_c_path = Path(args.model_c).resolve()
        auto_c_type = "torch7" if model_c_path.suffix.lower() == ".t7" else args.model_type
        model_c_type = getattr(args, "model_c_type", None) or auto_c_type
        io_preset_c = getattr(args, "io_preset_c", None) or io_preset
        c_type_resolved = model_c_type
        ipc_resolved = io_preset_c
        print(f"[backend] C: type={c_type_resolved} path={model_c_path} device={device} io_preset_c={ipc_resolved}")

        if model_c_type == "magenta":
            pass
        elif model_c_type == "torch7":
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
    use_d = bool(getattr(args, "model_d", None))
    d_type_resolved = None
    ipd_resolved = None
    if use_d:
        model_d_path = Path(args.model_d).resolve()
        auto_d_type = "torch7" if model_d_path.suffix.lower() == ".t7" else args.model_type
        model_d_type = getattr(args, "model_d_type", None) or auto_d_type
        io_preset_d = getattr(args, "io_preset_d", None) or io_preset
        d_type_resolved = model_d_type
        ipd_resolved = io_preset_d
        print(f"[backend] D: type={d_type_resolved} path={model_d_path} device={device} io_preset_d={ipd_resolved}")

        if model_d_type == "magenta":
            pass
        elif model_d_type == "torch7":
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

    # Temporal caches
    prev_gray = None  # uint8 HxW (content grayscale)
    prev_styled01 = None  # float32 HxWx3 in [0,1] (pre-LAB/pre-blend)
    prev_L = None  # LAB Lightness EMA cache
    last_flow = None  # float32 HxWx2

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
        x_orig01 = to_tensor(pil_rgb).unsqueeze(0)  # [1,3,H,W] 0..1
        gray = np.array(pil_rgb.convert("L"), dtype=np.uint8)  # HxW
        H0, W0 = x_orig01.shape[-2], x_orig01.shape[-1]
        # Early diagnostics for A forward (after dimensions known)
        try:
            th = torch.get_num_threads()
        except Exception:
            th = -1
        print(
            f"[A][{idx}] forward start: res={W0}x{H0} backend={args.model_type} device={device} threads={th} rss={_rss_mb():.1f} MB")
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
                    pil_out_t7 = _torch7_forward_openv(net_t7, pil_rgb)
                except Exception as e:
                    print(f"[torch7][ERROR] forward failed on frame {idx}: {e} — retrying at half-size")
                    try:
                        pil_small = pil_rgb.resize((max(1, pil_rgb.width // 2), max(1, pil_rgb.height // 2)),
                                                   Image.Resampling.BILINEAR)
                        pil_out_small = _torch7_forward_openv(net_t7, pil_small)
                        pil_out_t7 = pil_out_small.resize(pil_rgb.size, Image.Resampling.BILINEAR)
                        print("[torch7] retry succeeded at half-size")
                    except Exception as e2:
                        print(f"[torch7][FALLBACK] second attempt failed on frame {idx}: {e2}; using original frame")
                        pil_out_t7 = pil_rgb
                out01 = to_tensor(pil_out_t7).clamp(0, 1)
            else:
                with torch.no_grad():
                    print(f"[A][{idx}] io_preset branch start: {io_preset}")
                    if io_preset == "tanh":
                        x_in = (x_orig01 * 2.0 - 1.0).to(device)
                        y = model(x_in).cpu()
                        print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                        out01 = ((y + 1.0) * 0.5).clamp(0, 1).squeeze(0)
                    elif io_preset == "imagenet_01":
                        x_in = ((x_orig01 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
                        y = model(x_in).cpu()
                        print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                        out01 = (y * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).squeeze(0)
                    elif io_preset == "imagenet_255":
                        x255 = x_orig01 * 255.0
                        x_in = ((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0)).to(device)
                        y = model(x_in).cpu()
                        print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                        out01 = (y / 255.0).clamp(0, 1).squeeze(0)
                    elif io_preset == "caffe_bgr":
                        x255 = x_orig01 * 255.0
                        x_bgr255 = x255[:, [2, 1, 0], :, :]
                        CAFFE_MEAN = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
                        x_in = (x_bgr255 - CAFFE_MEAN).to(device)
                        y = model(x_in).cpu()
                        print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                        y_rgb255 = y[:, [2, 1, 0], :, :]
                        out01 = (y_rgb255 / 255.0).clamp(0, 1).squeeze(0)
                    elif io_preset == "raw_255":
                        x_in = (x_orig01 * 255.0).to(device) .        
                        y = model(x_in).cpu()
                        print(f"[A][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                        out01 = (y / 255.0).clamp(0, 1).squeeze(0) .
                    else:
                        x_in = (x_orig01 * 255.0).to(device)
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
            def process_model_output(model, net_t7, model_type, io_preset, frame_path, pil_rgb, device, magenta_style,
                                     model_name, idx, model_root="/app/models/magenta"):
                t0 = time.perf_counter()
                print(
                    f"[{model_name}][{idx}] forward start: res={pil_rgb.size[0]}x{pil_rgb.size[1]} backend={model_type} device={device} threads={torch.get_num_threads()} rss={_rss_mb():.1f} MB")
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
                            pil_out = _torch7_forward_openv(net_t7, pil_rgb)
                        except Exception as e:
                            print(f"[torch7][{model_name}][ERROR] forward failed: {e} — retrying at half-size")
                            try:
                                pil_small = pil_rgb.resize((max(1, pil_rgb.width // 2), max(1, pil_rgb.height // 2)),
                                                           Image.Resampling.BILINEAR)
                                pil_out_small = _torch7_forward_openv(net_t7, pil_small)
                                pil_out = pil_out_small.resize(pil_rgb.size, Image.Resampling.BILINEAR)
                                print(f"[torch7][{model_name}] retry succeeded at half-size")
                            except Exception as e2:
                                print(
                                    f"[torch7][{model_name}][FALLBACK] second attempt failed: {e2}; using original frame")
                                pil_out = pil_rgb
                        out = to_tensor(pil_out).clamp(0, 1)
                    else:
                        with torch.no_grad():
                            print(f"[{model_name}][{idx}] io_preset branch start: {io_preset}")
                            if io_preset == "tanh":
                                x_in = (x_orig01 * 2.0 - 1.0).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = ((y + 1.0) * 0.5).clamp(0, 1).squeeze(0)
                            elif io_preset == "imagenet_01":
                                x_in = ((x_orig01 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).squeeze(0)
                            elif io_preset == "imagenet_255":
                                x255 = x_orig01 * 255.0
                                x_in = ((x255 - IMAGENET_MEAN * 255.0) / (IMAGENET_STD * 255.0)).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y / 255.0).clamp(0, 1).squeeze(0)
                            elif io_preset == "caffe_bgr":
                                x255 = x_orig01 * 255.0
                                x_bgr255 = x255[:, [2, 1, 0], :, :]
                                CAFFE_MEAN = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
                                x_in = (x_bgr255 - CAFFE_MEAN).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                y_rgb255 = y[:, [2, 1, 0], :, :]
                                out = (y_rgb255 / 255.0).clamp(0, 1).squeeze(0)
                            elif io_preset == "raw_255":
                                x_in = (x_orig01 * 255.0).to(device)
                                y = model(x_in).cpu()
                                print(
                                    f"[{model_name}][{idx}] y-range pre-denorm: ({float(y.min()):.3f}..{float(y.max()):.3f})")
                                out = (y / 255.0).clamp(0, 1).squeeze(0)
                            else:
                                x_in = (x_orig01 * 255.0).to(device)
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
                out01_b = process_model_output(model_b, net_t7_b, b_type_resolved, ipb_resolved, frame_path, pil_rgb,
                                               device, style_img_b, "B", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_b)
                model_names.append("B")

            # Process model C
            if use_c:
                style_img_c = getattr(args, "magenta_style_c", None) or getattr(args, "magenta_style", None)
                if c_type_resolved == "magenta" and not style_img_c:
                    raise ValueError("--magenta_style_c or --magenta_style required for model C when type=magenta")
                out01_c = process_model_output(model_c, net_t7_c, c_type_resolved, ipc_resolved, frame_path, pil_rgb,
                                               device, style_img_c, "C", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_c)
                model_names.append("C")

            # Process model D
            if use_d:
                style_img_d = getattr(args, "magenta_style_d", None) or getattr(args, "magenta_style", None)
                if d_type_resolved == "magenta" and not style_img_d:
                    raise ValueError("--magenta_style_d or --magenta_style required for model D when type=magenta")
                out01_d = process_model_output(model_d, net_t7_d, d_type_resolved, ipd_resolved, frame_path, pil_rgb,
                                               device, style_img_d, "D", idx, model_root=str(
                        getattr(args, "magenta_model_root", "/app/models/magenta")))
                outputs.append(out01_d)
                model_names.append("D")

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
            if getattr(args, "blend_models_lab", False):
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
            else:
                # RGB blending
                weights = parse_blend_weights(getattr(args, "blend_models_weights", None), num_models)
                out01 = torch.zeros_like(outputs[0])
                for out, w in zip(outputs, weights):
                    out01 += w * out
                out01 = out01.clamp(0, 1)

            print(
                f"[blend][{idx}] Blended {','.join(model_names)} with weights={weights if not getattr(args, 'blend_models_lab', False) else f'L={wL},ab={wab},rest={weights_rest}'}")

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

            if args.flow_method == "farneback":
                flow_small = cv2.calcOpticalFlowFarneback(
                    prev_gray_small, gray_small,
                    None, 0.5, 3, 15, 3, 5, 1.1, 0
                )
            else:
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
                flow_small = dis.calc(prev_gray_small, gray_small, None)

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
                if prev_L is None:
                    prev_L = L.copy()
                L_sm = smooth_alpha * L + (1.0 - smooth_alpha) * prev_L
                prev_L = L_sm
                lab_np[..., 0] = np.clip(L_sm, 0, 255)

            if smooth_chroma:
                if 'prev_aC' not in locals(): prev_aC = aC.copy()
                if 'prev_bC' not in locals(): prev_bC = bC.copy()
                a_sm = chroma_alpha * aC + (1.0 - chroma_alpha) * prev_aC
                b_sm = chroma_alpha * bC + (1.0 - chroma_alpha) * prev_bC
                prev_aC, prev_bC = a_sm, b_sm
                lab_np[..., 1] = np.clip(a_sm, 0, 255)
                lab_np[..., 2] = np.clip(b_sm, 0, 255)

            out_img = Image.fromarray(lab_np.astype(np.uint8), mode="LAB").convert("RGB")

        # ---- Original-image blend (uniform or motion-adaptive) ----
        out01_smooth = to_tensor(out_img).unsqueeze(0)
        if out01_smooth.shape[-2:] != (H0, W0):
            x_orig01_rs = F.interpolate(x_orig01, size=out01_smooth.shape[-2:],
                                        mode="bilinear", align_corners=False)
        else:
            x_orig01_rs = x_orig01

        if getattr(args, "motion_blend", False) and (last_flow is not None):
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
    ap.add_argument("--input_video", required=False)
    ap.add_argument("--output_video", required=False)
    ap.add_argument("--model", required=False,
                    help="Path to a PyTorch checkpoint (.pth/.pt) when using transformer/reconet; not required for --model_type magenta.")
    ap.add_argument("--work_dir", default="./_work")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--pre_fps", type=int, default=None,
                    help="Reduce the frame rate of the INPUT VIDEO before extraction by first writing a temp video at this FPS (preprocess step).")
    ap.add_argument("--scale", type=int, default=None,
                    help="Long-side resolution (e.g., 720, 1080). Keep original if omitted.")
    ap.add_argument("--image_ext", choices=["png", "jpg"], default="png", help="Intermediate extraction format.")
    ap.add_argument("--jpeg_quality", type=int, default=85, help="JPG quality when saving styled frames (1–95).")
    ap.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads.")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (2 = every other frame).")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit number of frames for quick tests.")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")

    ap.add_argument("--io_preset",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255"],
                    default="imagenet_01",
                    help="Match to checkpoint's expected I/O (ReCoNet usually 'tanh'; Johnson transformer often 'imagenet_01').")

    ap.add_argument("--input_image", type=str, help="Path to a single input image.")
    ap.add_argument("--output_image", type=str, help="Path to a single output image.")
    ap.add_argument("--input_dir", type=str, help="Directory of input images for batch processing.")
    ap.add_argument("--output_dir", type=str, help="Output directory for batch image processing.")
    ap.add_argument("--pattern", type=str, default="*.jpg", help="Glob pattern for --input_dir (e.g., *.jpg, *.png).")
    ap.add_argument("--keep_ext", action="store_true", help="Preserve each input file's extension in batch mode.")
    ap.add_argument("--output_suffix", type=str, default="",
                    help="Optional suffix to add before the extension in batch mode.")

    # Flicker control .
    ap.add_argument("--smooth_lightness", action="store_true", default=True,
                    help="LAB Lightness EMA (on by default).")
    ap.add_argument("--no-smooth_lightness", action="store_false", dest="smooth_lightness")
    ap.add_argument("--smooth_alpha", type=float, default=0.7, help="EMA alpha 0–1 (higher = more current frame).")
    ap.add_argument("--smooth_chroma", action="store_true", default=False, help="LAB a/b EMA as well.")
    ap.add_argument("--chroma_alpha", type=float, default=0.85, help="EMA alpha for a/b when chroma smoothing is on.")

    ap.add_argument("--blend", type=float, default=1.0,
                    help="Uniform blend with original: 0.0=original, 1.0=fully stylized.")

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
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255"],
                    default=None,
                    help="I/O preset for the second model (defaults to --io_preset).")
    ap.add_argument("--model_c", type=str, default=None, help="Path to third model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_c_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the third model. Defaults to auto (.t7 -> torch7) or same as --model_type.")
    ap.add_argument("--io_preset_c",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255"],
                    default=None,
                    help="I/O preset for the third model (defaults to --io_preset).")
    ap.add_argument("--model_d", type=str, default=None, help="Path to fourth model checkpoint (.pth/.pt or .t7).")
    ap.add_argument("--model_d_type",
                    choices=["transformer", "reconet", "magenta", "torch7"],
                    default=None,
                    help="Backend for the fourth model. Defaults to auto (.t7 -> torch7) or same as --model_type.")
    ap.add_argument("--io_preset_d",
                    choices=["imagenet_255", "imagenet_01", "tanh", "caffe_bgr", "raw_255"],
                    default=None,
                    help="I/O preset for the fourth model (defaults to --io_preset).")
    ap.add_argument("--blend_models_weights", type=str, default=None,
                    help="Comma-separated weights for blending A,B,C,D (e.g., '0.4,0.3,0.2,0.1'). Must sum to 1.0. Defaults to equal weights.")
    ap.add_argument("--blend_models_lab", action="store_true", default=False,
                    help="Blend in LAB: take Lightness from model A and chroma (a/b) from model B,C,D.")
    ap.add_argument("--blend_models_lab_weights", type=str, default=None,
                    help="If --blend_models_lab, weights for L (from A) and a/b (from B,C,D) in format 'wL,wab' (e.g., '0.5,0.5').")
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

    args = ap.parse_args()

    # Extra safety for OpenCV DNN stability in some environments
    os.environ.setdefault('OPENCV_OPENCL_DEVICE', 'disabled')
    os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

    # Decide mode
    image_mode_single = bool(getattr(args, "input_image", None)) and bool(getattr(args, "output_image", None))
    image_mode_batch = bool(getattr(args, "input_dir", None)) and bool(getattr(args, "output_dir", None))
    video_mode = bool(getattr(args, "input_video", None)) and bool(getattr(args, "output_video", None))

    if (image_mode_single or image_mode_batch) and video_mode:
        print("Provide either video args OR image args, not both.")
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

    work_dir = Path(args.work_dir).resolve()
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing staged frames (png/jpg/jpeg)
    for p in frames_dir.glob("frame_*.png"): p.unlink(missing_ok=True)
    for p in frames_dir.glob("frame_*.jpg"): p.unlink(missing_ok=True)
    for p in frames_dir.glob("frame_*.jpeg"): p.unlink(missing_ok=True)
    for p in frames_dir.glob("styled_frame_*.png"): p.unlink(missing_ok=True)
    for p in frames_dir.glob("styled_frame_*.jpg"): p.unlink(missing_ok=True)
    for p in frames_dir.glob("styled_frame_*.jpeg"): p.unlink(missing_ok=True)

    if args.model_type != "magenta":
        model_path = Path(args.model).resolve()
    else:
        model_path = None

    if args.model_type != "magenta" and model_path is not None and model_path.suffix.lower() == ".t7":
        print(f"[auto] Detected .t7 checkpoint ({model_path.name}); switching backend to torch7 (OpenCV DNN).")
        args.model_type = "torch7"

    save_map = {}
    if video_mode:
        input_video = Path(args.input_video).resolve()
        output_video = Path(args.output_video).resolve()
        print(
            f"[cfg] scale={args.scale}  fps={args.fps}  pre_fps={getattr(args, 'pre_fps', None)}  image_ext={args.image_ext}")

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
        extract_frames(input_video, frames_dir, extract_fps, args.scale, args.image_ext, args.jpeg_quality)
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
            save_map[i] = str((Path(args.output_dir) / f"{base}{suffix}{out_ext}").resolve())

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


if __name__ == "__main__":
    main()