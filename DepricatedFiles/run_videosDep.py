
#!/usr/bin/env python3
import os
import sys
import subprocess
import shlex
import pathlib
from itertools import chain

# Artifact version
ARTIFACT_VERSION = "b2c3d4e5-f6a7-8901-cdef-123456789abc"
print(f"[debug] run_videos.py artifact version: {ARTIFACT_VERSION}")

# Inputs / outputs
IN_DIR = os.getenv("IN_DIR", "/app/input_videos")
OUT_DIR = os.getenv("OUT_DIR", "/app/output")
WORK_ROOT = os.getenv("WORK_ROOT", "/app/_work")

# Models
PYTORCH_DIR = os.getenv("PYTORCH_DIR", "/app/models/pytorch")
TORCH_DIR = os.getenv("TORCH_DIR", "/app/models/torch")
MAGENTA_DIR = os.getenv("MAGENTA_DIR", "/app/models/magenta")
MAGENTA_STYLES_DIR = os.getenv("MAGENTA_STYLES_DIR", "/app/models/magenta_styles")

# Video I/O
SCALE = os.getenv("SCALE", "720")
FPS = os.getenv("FPS", "24")
PRE_FPS = os.getenv("PRE_FPS", "")  # if set, we pass --pre_fps
IMG_EXT = os.getenv("IMG_EXT", "jpg")
JPEG_QUALITY = os.getenv("JPEG_QUALITY", "85")

# Model I/O presets (per-model)
IO_A = os.getenv("IO_PRESET_A", "imagenet_255")
IO_B = os.getenv("IO_PRESET_B", IO_A)
IO_C = os.getenv("IO_PRESET_C", IO_A)
IO_D = os.getenv("IO_PRESET_D", IO_A)

# Model selection
MODEL_A = os.getenv("MODEL_A", "")  # Model name/path for A
MODEL_B = os.getenv("MODEL_B", "")  # Model name/path for B
MODEL_C = os.getenv("MODEL_C", "")  # Model name/path for C
MODEL_D = os.getenv("MODEL_D", "")  # Model name/path for D
MODEL_A_TYPE = os.getenv("MODEL_A_TYPE", "transformer")  # transformer, torch7, or magenta
MODEL_B_TYPE = os.getenv("MODEL_B_TYPE", "transformer")
MODEL_C_TYPE = os.getenv("MODEL_C_TYPE", "transformer")
MODEL_D_TYPE = os.getenv("MODEL_D_TYPE", "magenta")
USE_B = os.getenv("USE_B", "1") == "1"  # Enable model B
USE_C = os.getenv("USE_C", "1") == "1"  # Enable model C
USE_D = os.getenv("USE_D", "1") == "1"  # Enable model D

# Magenta style images
MAGENTA_STYLE = os.getenv("MAGENTA_STYLE", "")      # Style for model A if magenta
MAGENTA_STYLE_B = os.getenv("MAGENTA_STYLE_B", "")  # Style for model B if magenta
MAGENTA_STYLE_C = os.getenv("MAGENTA_STYLE_C", "")  # Style for model C if magenta
MAGENTA_STYLE_D = os.getenv("MAGENTA_STYLE_D", "")  # Style for model D if magenta

# Model blend weights
BLEND_WEIGHTS = os.getenv("BLEND_WEIGHTS", "")  # e.g., "0.25,0.25,0.25,0.25"
BLEND_LAB = os.getenv("BLEND_LAB", "0") == "1"  # Enable LAB blending
BLEND_LAB_WEIGHTS = os.getenv("BLEND_LAB_WEIGHTS", "0.5,0.5")  # e.g., "0.5,0.5" for L, a/b

# Stylized↔original blend
BLEND = float(os.getenv("BLEND", "1.0"))  # for --blend (1.0 = fully stylized)

# Temporal smoothing (Lightness EMA)
SMOOTH_LIGHT = os.getenv("SMOOTH_LIGHTNESS", "1") == "1"  # 1=on (pipeline default), 0=off
SMOOTH_ALPHA = os.getenv("SMOOTH_ALPHA", "0.7")  # EMA alpha (higher = less smoothing)

# Optical-flow EMA (disabled by default)
FLOW_EMA = os.getenv("FLOW_EMA", "0") == "1"
FLOW_ALPHA = os.getenv("FLOW_ALPHA", "0.85")
FLOW_METHOD = os.getenv("FLOW_METHOD", "dis")  # dis|farneback
FLOW_DOWNSCALE = os.getenv("FLOW_DOWNSCALE", "1")  # 1,2,3…

# Ensure output dir exists
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Validate Magenta model directory
if pathlib.Path(MAGENTA_DIR).exists():
    magenta_subdirs = [d for d in pathlib.Path(MAGENTA_DIR).glob("*") if d.is_dir()]
    if not magenta_subdirs:
        print(f"[warning] No model subdirectory found in {MAGENTA_DIR}; Magenta models will not be available unless specified")
        magenta_available = False
    else:
        magenta_available = True
else:
    print(f"[warning] Magenta directory {MAGENTA_DIR} does not exist; Magenta models will not be available")
    magenta_available = False

# Determine number of models to use
num_models = 1  # Always use model A
if USE_B and MODEL_B:
    num_models += 1
if USE_C and MODEL_C:
    num_models += 1
if USE_D and MODEL_D:
    num_models += 1

def _resolve_model(root: str, val: str) -> pathlib.Path:
    """
    Resolve a model path that may be absolute or relative to a given root.
    """
    p = pathlib.Path(val)
    return p if p.is_absolute() else pathlib.Path(root) / p

# Generate or parse blend weights
def generate_blend_weights(num_models: int):
    if BLEND_WEIGHTS:
        weights = [float(w) for w in BLEND_WEIGHTS.split(",")]
        if len(weights) != num_models:
            raise ValueError(f"BLEND_WEIGHTS must have {num_models} values, got {len(weights)}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"BLEND_WEIGHTS must sum to 1.0, got {sum(weights):.6f}")
        return weights
    return [1.0 / num_models] * num_models  # Equal weights

# Parse LAB blend weights
def parse_lab_weights():
    if BLEND_LAB and BLEND_LAB_WEIGHTS:
        wL, wab = [float(w) for w in BLEND_LAB_WEIGHTS.split(",")]
        if abs(wL + wab - 1.0) > 1e-6:
            raise ValueError(f"BLEND_LAB_WEIGHTS must sum to 1.0, got {wL + wab:.6f}")
        return wL, wab
    return 0.5, 0.5  # Default to equal contribution

# Accept a single input path (preferred), else fall back to scanning IN_DIR
if len(sys.argv) > 1:
    vids = [pathlib.Path(sys.argv[1])]
    print(f"[debug] Single-input mode: {vids[0].name}")
else:
    vids = chain(pathlib.Path(IN_DIR).glob("*.mp4"),
                 pathlib.Path(IN_DIR).glob("*.mov"))
    print(f"[debug] Directory-scan mode: IN_DIR={IN_DIR}")

for vid in vids:
    # Log input video and environment variables
    print(f"[debug] Processing video: {vid.name}")
    print(f"[debug] MODEL_A={MODEL_A}, MODEL_A_TYPE={MODEL_A_TYPE}, MAGENTA_STYLE={MAGENTA_STYLE}, IO_PRESET_A={IO_A}")
    print(f"[debug] MODEL_B={MODEL_B}, MODEL_B_TYPE={MODEL_B_TYPE}, MAGENTA_STYLE_B={MAGENTA_STYLE_B}, IO_PRESET_B={IO_B}")
    print(f"[debug] MODEL_C={MODEL_C}, MODEL_C_TYPE={MODEL_C_TYPE}, MAGENTA_STYLE_C={MAGENTA_STYLE_C}, IO_PRESET_C={IO_C}")
    print(f"[debug] MODEL_D={MODEL_D}, MODEL_D_TYPE={MODEL_D_TYPE}, MAGENTA_STYLE_D={MAGENTA_STYLE_D}, IO_PRESET_D={IO_D}")
    print(f"[debug] BLEND_WEIGHTS={BLEND_WEIGHTS}")

    # Validate required environment variables
    if not MODEL_A:
        raise ValueError("MODEL_A must be specified")
    if num_models >= 2 and not MODEL_B:
        raise ValueError("MODEL_B must be specified when USE_B=1")
    if num_models >= 3 and not MODEL_C:
        raise ValueError("MODEL_C must be specified when USE_C=1")
    if num_models >= 4 and not MODEL_D:
        raise ValueError("MODEL_D must be specified when USE_D=1")

    # Select models
    selected_models = []

    # Model A (required)
    if MODEL_A_TYPE == "magenta":
        if not MAGENTA_STYLE:
            raise ValueError("MAGENTA_STYLE required for model A when MODEL_A_TYPE=magenta")
        if not magenta_available:
            raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
        mA = pathlib.Path(MAGENTA_DIR)
        style_a = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE
        if not style_a.exists():
            raise ValueError(f"Style image {style_a} not found")
    elif MODEL_A_TYPE == "torch7":
        mA = _resolve_model(TORCH_DIR, MODEL_A)
        if not mA.exists():
            raise ValueError(f"Torch7 model {mA} not found")
    else:  # transformer
        mA = _resolve_model(PYTORCH_DIR, MODEL_A)
        if not mA.exists():
            raise ValueError(f"PyTorch model {mA} not found")
    selected_models.append(("A", mA, MODEL_A_TYPE, MAGENTA_STYLE))

    # Model B
    mB = None
    if num_models >= 2 and USE_B:
        if MODEL_B_TYPE == "magenta":
            if not MAGENTA_STYLE_B:
                raise ValueError("MAGENTA_STYLE_B required for model B when MODEL_B_TYPE=magenta")
            if not magenta_available:
                raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
            mB = pathlib.Path(MAGENTA_DIR)
            style_b = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_B
            if not style_b.exists():
                raise ValueError(f"Style image {style_b} not found")
        elif MODEL_B_TYPE == "torch7":
            mB = _resolve_model(TORCH_DIR, MODEL_B)
            if not mB.exists():
                raise ValueError(f"Torch7 model {mB} not found")
        else:
            mB = _resolve_model(PYTORCH_DIR, MODEL_B)
            if not mB.exists():
                raise ValueError(f"PyTorch model {mB} not found")
        selected_models.append(("B", mB, MODEL_B_TYPE, MAGENTA_STYLE_B))

    # Model C
    mC = None
    if num_models >= 3 and USE_C:
        if MODEL_C_TYPE == "magenta":
            if not MAGENTA_STYLE_C:
                raise ValueError("MAGENTA_STYLE_C required for model C when MODEL_C_TYPE=magenta")
            if not magenta_available:
                raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
            mC = pathlib.Path(MAGENTA_DIR)
            style_c = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_C
            if not style_c.exists():
                raise ValueError(f"Style image {style_c} not found")
        elif MODEL_C_TYPE == "torch7":
            mC = _resolve_model(TORCH_DIR, MODEL_C)
            if not mC.exists():
                raise ValueError(f"Torch7 model {mC} not found")
        else:
            mC = _resolve_model(PYTORCH_DIR, MODEL_C)
            if not mC.exists():
                raise ValueError(f"PyTorch model {mC} not found")
        selected_models.append(("C", mC, MODEL_C_TYPE, MAGENTA_STYLE_C))

    # Model D
    mD = None
    if num_models >= 4 and USE_D:
        if MODEL_D_TYPE == "magenta":
            if not MAGENTA_STYLE_D:
                raise ValueError("MAGENTA_STYLE_D required for model D when MODEL_D_TYPE=magenta")
            if not magenta_available:
                raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
            mD = pathlib.Path(MAGENTA_DIR)
            style_d = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_D
            if not style_d.exists():
                raise ValueError(f"Style image {style_d} not found")
        elif MODEL_D_TYPE == "torch7":
            mD = _resolve_model(TORCH_DIR, MODEL_D)
            if not mD.exists():
                raise ValueError(f"Torch7 model {mD} not found")
        else:
            mD = _resolve_model(PYTORCH_DIR, MODEL_D)
            if not mD.exists():
                raise ValueError(f"PyTorch model {mD} not found")
        selected_models.append(("D", mD, MODEL_D_TYPE, MAGENTA_STYLE_D))

    # Ensure at least one model is selected
    if not selected_models:
        raise SystemExit("At least one model must be selected")

    # Generate blend weights
    weights = generate_blend_weights(num_models)
    weight_str = ",".join(f"{w:.3f}" for w in weights)
    print(f"[debug] Generated blend weights: {weight_str}")

    # Generate output filename
    stem = vid.stem
    model_str = "_".join(
        f"{name}-{model.stem if model_type != 'magenta' else style.split('/')[-1].split('.')[0] if style else 'magenta'}"
        for name, model, model_type, style in selected_models
    )
    outp = pathlib.Path(OUT_DIR) / f"{stem}_{model_str}_w-{weight_str}.mp4"
    print(f"[debug] Expected output filename: {outp}")

    # Create work directory
    work = pathlib.Path(WORK_ROOT) / stem
    work.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python3", "-u", "/app/pipeline.py",
        "--model_type", str(MODEL_A_TYPE),
        "--model", str(mA),
        "--io_preset", str(IO_A),
        "--blend", str(BLEND),
        "--input_video", str(vid),
        "--output_video", str(outp),
        "--work_dir", str(work),
        "--image_ext", str(IMG_EXT),
        "--jpeg_quality", str(JPEG_QUALITY),
        "--scale", str(SCALE),
        "--fps", str(FPS),
        "--smooth_alpha", str(SMOOTH_ALPHA),
        "--flow_method", str(FLOW_METHOD),
        "--flow_downscale", str(FLOW_DOWNSCALE),
        "--magenta_model_root", str(MAGENTA_DIR),
        # "--max_frames", "1",
    ]

    # Add Magenta style for model A
    if MODEL_A_TYPE == "magenta" and MAGENTA_STYLE:
        cmd += ["--magenta_style", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE)]

    # Add model B
    if mB:
        cmd += ["--model_b", str(mB), "--model_b_type", str(MODEL_B_TYPE), "--io_preset_b", str(IO_B)]
        if MODEL_B_TYPE == "magenta" and MAGENTA_STYLE_B:
            cmd += ["--magenta_style_b", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_B)]

    # Add model C
    if mC:
        cmd += ["--model_c", str(mC), "--model_c_type", str(MODEL_C_TYPE), "--io_preset_c", str(IO_C)]
        if MODEL_C_TYPE == "magenta" and MAGENTA_STYLE_C:
            cmd += ["--magenta_style_c", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_C)]

    # Add model D
    if mD:
        cmd += ["--model_d", str(mD), "--model_d_type", str(MODEL_D_TYPE), "--io_preset_d", str(IO_D)]
        if MODEL_D_TYPE == "magenta" and MAGENTA_STYLE_D:
            cmd += ["--magenta_style_d", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_D)]

    # Add blend weights
    if num_models > 1:
        cmd += ["--blend_models_weights", str(weight_str)]
        if BLEND_LAB:
            wL, wab = parse_lab_weights()
            cmd += ["--blend_models_lab", "--blend_models_lab_weights", f"{wL},{wab}"]

    # Lightness EMA toggle
    if SMOOTH_LIGHT:
        cmd.append("--smooth_lightness")
    else:
        cmd.append("--no-smooth_lightness")

    # Flow EMA toggle + alpha
    if FLOW_EMA:
        cmd += ["--flow_ema", "--flow_alpha", str(FLOW_ALPHA)]

    # Pre-FPS (pre-resample input before extraction)
    if PRE_FPS:
        cmd += ["--pre_fps", str(PRE_FPS)]

    print("[run]", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)