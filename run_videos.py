#!/usr/bin/env python3
"""
run_videos.py — thin adapter between drive_videos.py and pipeline.py

Behavior (restored & improved):
- Accepts a single positional VIDEO PATH. No model parsing here.
- Reads env prepared by drive_videos.py: MODEL_A..D, MODEL_*_TYPE, MAGENTA_STYLE[_B|_C|_D], IO_PRESET[_B|_C|_D],
  BLEND_WEIGHTS, BLEND_MODELS_LAB, BLEND_MODELS_LAB_WEIGHTS, plus core knobs (SCALE, FPS, PRE_FPS, etc.).
- Calls /app/pipeline.py once, mapping A..D slots to real flags.
- Always supplies --output_video alongside --input_video (pipeline.py requires it).
- Passes --max_frames when MAX_FRAMES env is set.

This keeps drive_videos.py as the source of truth for model picking/shuffling.
"""

import os
import sys
import shlex
import pathlib
import subprocess
from typing import Optional

# ---- Core env helpers -------------------------------------------------------

def getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def getbool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


# ---- Resolution helpers -----------------------------------------------------

MAGENTA_STYLES_DIR = getenv("MAGENTA_STYLES_DIR", "/app/models/magenta_styles")
PYTORCH_DIR        = getenv("PYTORCH_DIR",        "/app/models/pytorch")
TORCH_DIR          = getenv("TORCH_DIR",          "/app/models/torch")
TRANSFORMER_DIR    = getenv("TRANSFORMER_DIR",    "/app/models/transformers")

# Pipeline accepts only: transformer, reconet, magenta, torch7
def canonical_model_type(t: Optional[str]) -> str:
    t = (t or "").lower()
    # Map external naming to pipeline's accepted choices
    if t == "pytorch":
        return "transformer"
    return t


def resolve_nonmagnet_model(path_or_name: str, model_type: str) -> str:
    """Resolve a non-magenta model path, respecting absolute paths and existing extensions.
    drive_videos.py usually passes absolute paths already; we pass through unchanged.
    """
    p = pathlib.Path(path_or_name)
    if p.is_absolute():
        return str(p)
    # Fallbacks only if relative names were provided (unlikely with your planner)
    mt = canonical_model_type(model_type)
    if mt in {"pytorch", "transformer"}:
        # Treat PyTorch-style models as transformer backends; keep files under PYTORCH_DIR
        return str(pathlib.Path(PYTORCH_DIR) / (path_or_name if pathlib.Path(path_or_name).suffix else f"{path_or_name}.pth"))
    if mt == "torch7":
        return str(pathlib.Path(TORCH_DIR) / (path_or_name if pathlib.Path(path_or_name).suffix else f"{path_or_name}.t7"))
    if mt in {"transformer", "reconet"}:  # fallback for explicit transformer dir setups
        return str(pathlib.Path(TRANSFORMER_DIR) / path_or_name)
    return str(p)


def resolve_magenta_style(style_name: Optional[str]) -> Optional[str]:
    if not style_name:
        return None
    p = pathlib.Path(style_name)
    return str(p if p.is_absolute() else pathlib.Path(MAGENTA_STYLES_DIR) / style_name)


# ---- Pipeline command construction -----------------------------------------

IN_DIR  = getenv("IN_DIR",  "/app/input_videos")
OUT_DIR = getenv("OUT_DIR", "/app/output")
SCALE   = getenv("SCALE",   "720")
FPS     = getenv("FPS",     "24")
PRE_FPS = getenv("PRE_FPS")
BLEND   = getenv("BLEND",   "0.9")
FLOW_METHOD    = getenv("FLOW_METHOD",    "dis")
FLOW_DOWNSCALE = getenv("FLOW_DOWNSCALE", "1")
SMOOTH_LIGHTNESS = getbool("SMOOTH_LIGHTNESS", False)
SMOOTH_ALPHA     = getenv("SMOOTH_ALPHA", "0.65")
FLOW_EMA   = getbool("FLOW_EMA", False)
FLOW_ALPHA = getenv("FLOW_ALPHA", "0.7")
SMOOTH_CHROMA = getbool("SMOOTH_CHROMA", False)
CHROMA_ALPHA  = getenv("CHROMA_ALPHA")
IO_PRESET_GLOBAL = getenv("IO_PRESET")  # optional global, applies to A if A has no per-slot preset
MAX_FRAMES = getenv("MAX_FRAMES")  # pass through when set .
STRIDE        = getenv("STRIDE")
JPEG_QUALITY  = getenv("JPEG_QUALITY")
CLEAN_FRAMES  = getbool("CLEAN_FRAMES", False)
MAGENTA_TILE       = getenv("MAGENTA_TILE")
MAGENTA_OVERLAP    = getenv("MAGENTA_OVERLAP")
MAGENTA_TARGET_RES = getenv("MAGENTA_TARGET_RES")
MAGENTA_MODEL_ROOT = getenv("MAGENTA_MODEL_ROOT")

# --- Extra optional pass-throughs (add flags if present) ---
DEVICE        = getenv("DEVICE")         # e.g. cpu / cuda
THREADS       = getenv("THREADS")        # worker threads for some backends
IMAGE_EXT     = getenv("IMAGE_EXT")      # png/jpg
SEED          = getenv("SEED")           # deterministic seed if supported
MOTION_BLEND  = getbool("MOTION_BLEND", False)  # temporal motion/content blend (pipeline flag: --motion_blend)
SMOOTHING     = getenv("SMOOTHING")      # '0'/'1' to force enable/disable smoothing when supported
PIPELINE_ARGS = getenv("PIPELINE_ARGS")  # free-form extra args appended verbatim (shlex-split)


def add_slot(cmd: list[str], slot_suffix: str, model_val: Optional[str], model_type: Optional[str],
             magenta_style: Optional[str], io_preset: Optional[str]):
    """Add one slot (A:"", B:"_b", C:"_c", D:"_d") to the pipeline command.
    For magenta, the model is the engine name 'magenta' and style passed with --magenta_style*
    For others, we pass a resolved file path.
    """
    if not (model_val or model_type or magenta_style):
        return  # slot not present

    t = canonical_model_type(model_type)
    if t == "magenta":
        cmd += [f"--model{slot_suffix}", "magenta", f"--model{slot_suffix}_type", "magenta"]
        sty_flag = "--magenta_style" if slot_suffix == "" else f"--magenta_style{slot_suffix}"
        resolved_style = resolve_magenta_style(magenta_style)
        if resolved_style:
            cmd += [sty_flag, resolved_style]
    else:
        # non-magenta
        if not model_val:
            return
        resolved_path = resolve_nonmagnet_model(model_val, t)
        cmd += [f"--model{slot_suffix}", resolved_path, f"--model{slot_suffix}_type", t]

    # Per-slot IO preset (A uses --io_preset; B/C/D use suffixed flags)
    if io_preset:
        preset_flag = "--io_preset" if slot_suffix == "" else f"--io_preset{slot_suffix}"
        cmd += [preset_flag, io_preset]


def build_pipeline_cmd(video_path: str) -> list[str]:
    stem = pathlib.Path(video_path).stem
    output_suffix = getenv("OUTPUT_SUFFIX", "")  # optional suffix if planner sets it
    output_video = str(pathlib.Path(OUT_DIR) / f"{stem}{output_suffix}.mp4")

    cmd = [
        "python3", "/app/pipeline.py",
        "--input_video", video_path,
        "--output_video", output_video,
        "--output_dir", OUT_DIR,
        "--scale", str(SCALE),
        "--fps", str(FPS),
        "--blend", str(BLEND),
        "--flow_method", FLOW_METHOD,
        "--flow_downscale", str(FLOW_DOWNSCALE),
    ]

    if PRE_FPS:
        cmd += ["--pre_fps", str(PRE_FPS)]

    # Smoothing controls: always pass alpha when provided
    if getbool("SMOOTH_LIGHTNESS", False):
        cmd += ["--smooth_lightness"]
    if SMOOTH_ALPHA is not None:
        cmd += ["--smooth_alpha", str(SMOOTH_ALPHA)]

    if SMOOTH_CHROMA:
        cmd += ["--smooth_chroma"]
    if CHROMA_ALPHA is not None:
        cmd += ["--chroma_alpha", str(CHROMA_ALPHA)]

    if FLOW_EMA:
        cmd += ["--flow_ema", "--flow_alpha", str(FLOW_ALPHA)]
    if MAX_FRAMES:
        cmd += ["--max_frames", str(MAX_FRAMES)]

    if STRIDE:
        cmd += ["--stride", str(STRIDE)]
    if JPEG_QUALITY:
        cmd += ["--jpeg_quality", str(JPEG_QUALITY)]

    if MAGENTA_TILE:
        cmd += ["--magenta_tile", str(MAGENTA_TILE)]
    if MAGENTA_OVERLAP:
        cmd += ["--magenta_overlap", str(MAGENTA_OVERLAP)]
    if MAGENTA_TARGET_RES:
        cmd += ["--magenta_target_res", str(MAGENTA_TARGET_RES)]
    if MAGENTA_MODEL_ROOT:
        cmd += ["--magenta_model_root", str(MAGENTA_MODEL_ROOT)]

    if CLEAN_FRAMES:
        cmd += ["--clean_frames"]

    # Optional blend controls across multiple models
    blend_weights = getenv("BLEND_WEIGHTS")
    if blend_weights:
        cmd += ["--blend_models_weights", blend_weights]
    if getbool("BLEND_MODELS_LAB", False):
        cmd += ["--blend_models_lab"]
    blend_lab_weights = getenv("BLEND_MODELS_LAB_WEIGHTS")
    if blend_lab_weights:
        cmd += ["--blend_models_lab_weights", blend_lab_weights]

    # Optional general controls if present
    if DEVICE:
        cmd += ["--device", str(DEVICE)]
    if THREADS:
        cmd += ["--threads", str(THREADS)]
    if IMAGE_EXT:
        cmd += ["--image_ext", str(IMAGE_EXT)]
    if SEED:
        cmd += ["--seed", str(SEED)]
    if MOTION_BLEND:
        cmd += ["--motion_blend"]
    # Allow explicit enable/disable of smoothing if pipeline supports it
    if SMOOTHING is not None:
        if SMOOTHING.lower() in {"0", "false", "no", "off"}:
            cmd += ["--no_smoothing"]
        else:
            cmd += ["--smoothing"]
    # Free-form argument passthrough for anything else supported by pipeline.py
    if PIPELINE_ARGS:
        cmd += shlex.split(PIPELINE_ARGS)

    # Per-slot IO presets (slot-global IO_PRESET_X overrides IO_PRESET_GLOBAL for that slot)
    ioA = getenv("IO_PRESET_A", IO_PRESET_GLOBAL)
    ioB = getenv("IO_PRESET_B")
    ioC = getenv("IO_PRESET_C")
    ioD = getenv("IO_PRESET_D")

    # Slot A (unsuffixed flags)
    add_slot(
        cmd,
        "",
        getenv("MODEL_A"),
        getenv("MODEL_A_TYPE"),
        getenv("MAGENTA_STYLE"),
        ioA,
    )

    # Slot B
    add_slot(
        cmd,
        "_b",
        getenv("MODEL_B"),
        getenv("MODEL_B_TYPE"),
        getenv("MAGENTA_STYLE_B"),
        ioB,
    )

    # Slot C
    add_slot(
        cmd,
        "_c",
        getenv("MODEL_C"),
        getenv("MODEL_C_TYPE"),
        getenv("MAGENTA_STYLE_C"),
        ioC,
    )

    # Slot D
    add_slot(
        cmd,
        "_d",
        getenv("MODEL_D"),
        getenv("MODEL_D_TYPE"),
        getenv("MAGENTA_STYLE_D"),
        ioD,
    )

    return cmd


# ---- Main -------------------------------------------------------------------

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: run_videos.py <video_path>")
        return 2

    video_path = argv[1]
    cmd = build_pipeline_cmd(video_path)

    # Debug prints
    print(f"[run] MAX_FRAMES={MAX_FRAMES or ''}")
    if DEVICE or THREADS or IMAGE_EXT or SEED or MOTION_BLEND or PIPELINE_ARGS:
        print("[run][extras]",
              f"device={DEVICE or ''} threads={THREADS or ''} img_ext={IMAGE_EXT or ''} seed={SEED or ''} "
              f"motion_blend={'1' if MOTION_BLEND else '0'} extra='{PIPELINE_ARGS or ''}'")
    print("[run]", " ".join(shlex.quote(x) for x in cmd))

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
