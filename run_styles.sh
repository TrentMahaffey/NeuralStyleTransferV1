#!/usr/bin/env python3
import os, random, subprocess, shlex, pathlib
from itertools import chain

# Inputs / outputs
IN_DIR       = os.getenv("IN_DIR", "/app/input_videos")
OUT_DIR      = os.getenv("OUT_DIR", "/app/output")
WORK_ROOT    = os.getenv("WORK_ROOT", "/app/_work")

# Models
PYTORCH_DIR  = os.getenv("PYTORCH_DIR", "/app/models/pytorch")
TORCH_DIR    = os.getenv("TORCH_DIR", "/app/models/torch")
MAGENTA_DIR  = os.getenv("MAGENTA_DIR", "/app/models/magenta")
MAGENTA_STYLES_DIR = os.getenv("MAGENTA_STYLES_DIR", "/app/models/magenta_styles")

# Video I/O
SCALE        = os.getenv("SCALE", "720")
FPS          = os.getenv("FPS", "24")
PRE_FPS      = os.getenv("PRE_FPS", "")  # if set, we pass --pre_fps
IMG_EXT      = os.getenv("IMG_EXT", "jpg")
JPEG_QUALITY = os.getenv("JPEG_QUALITY", "85")

# Model I/O presets (per-model)
IO_A         = os.getenv("IO_PRESET_A", "imagenet_255")
IO_B         = os.getenv("IO_PRESET_B", IO_A)
IO_C         = os.getenv("IO_PRESET_C", IO_A)
IO_D         = os.getenv("IO_PRESET_D", IO_A)

# Model selection
MODEL_A      = os.getenv("MODEL_A", "")  # Model name for A (e.g., candy.pth, la_muse_eccv16.t7, or magenta)
MODEL_B      = os.getenv("MODEL_B", "")  # Model name for B
MODEL_C      = os.getenv("MODEL_C", "")  # Model name for C
MODEL_D      = os.getenv("MODEL_D", "")  # Model name for D
MODEL_A_TYPE = os.getenv("MODEL_A_TYPE", "transformer")  # transformer, torch7, or magenta
MODEL_B_TYPE = os.getenv("MODEL_B_TYPE", "transformer")
MODEL_C_TYPE = os.getenv("MODEL_C_TYPE", "transformer")
MODEL_D_TYPE = os.getenv("MODEL_D_TYPE", "magenta")
USE_B        = os.getenv("USE_B", "1") == "1"  # Enable model B
USE_C        = os.getenv("USE_C", "1") == "1"  # Enable model C
USE_D        = os.getenv("USE_D", "1") == "1"  # Enable model D

# Magenta style images
MAGENTA_STYLE   = os.getenv("MAGENTA_STYLE", "")    # Style for model A if magenta
MAGENTA_STYLE_B = os.getenv("MAGENTA_STYLE_B", "")  # Style for model B if magenta
MAGENTA_STYLE_C = os.getenv("MAGENTA_STYLE_C", "")  # Style for model C if magenta
MAGENTA_STYLE_D = os.getenv("MAGENTA_STYLE_D", "")  # Style for model D if magenta

# Model blend weights
RANDOM_WEIGHTS = os.getenv("RANDOM_WEIGHTS", "0") == "1"  # Randomize blend weights
BLEND_WEIGHTS  = os.getenv("BLEND_WEIGHTS", "")  # e.g., "0.4,0.3,0.2,0.1"
BLEND_LAB      = os.getenv("BLEND_LAB", "0") == "1"  # Enable LAB blending
BLEND_LAB_WEIGHTS = os.getenv("BLEND_LAB_WEIGHTS", "0.5,0.5")  # e.g., "0.5,0.5" for L, a/b

# Stylized↔original blend
BLEND        = float(os.getenv("BLEND", "1.0"))  # for --blend (1.0 = fully stylized)

# Temporal smoothing (Lightness EMA)
SMOOTH_LIGHT = os.getenv("SMOOTH_LIGHTNESS", "1") == "1"  # 1=on (pipeline default), 0=off
SMOOTH_ALPHA = os.getenv("SMOOTH_ALPHA", "0.7")           # EMA alpha (higher = less smoothing)

# Optical-flow EMA (disabled by default)
FLOW_EMA       = os.getenv("FLOW_EMA", "0") == "1"
FLOW_ALPHA     = os.getenv("FLOW_ALPHA", "0.85")
FLOW_METHOD    = os.getenv("FLOW_METHOD", "dis")          # dis|farneback
FLOW_DOWNSCALE = os.getenv("FLOW_DOWNSCALE", "1")         # 1,2,3…

# Ensure output dir exists
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Collect models and styles
pytorch_models = [p for p in pathlib.Path(PYTORCH_DIR).glob("*.pth")]
torch_models = [p for p in pathlib.Path(TORCH_DIR).glob("*.t7")]
magenta_styles = [p for p in pathlib.Path(MAGENTA_STYLES_DIR).glob("*.jpg")]

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

if not (pytorch_models or torch_models or (magenta_available and magenta_styles)):
    raise SystemExit(f"No models found in {PYTORCH_DIR}, {TORCH_DIR}, or {MAGENTA_DIR} with styles in {MAGENTA_STYLES_DIR}")

# Determine number of models to use
num_models = 1  # Always use model A
if USE_B and (MODEL_B or len(pytorch_models) + len(torch_models) >= 2 or (MODEL_B_TYPE == "magenta" and magenta_styles and magenta_available)):
    num_models += 1
if USE_C and (MODEL_C or len(pytorch_models) + len(torch_models) >= 3 or (MODEL_C_TYPE == "magenta" and magenta_styles and magenta_available)):
    num_models += 1
if USE_D and (MODEL_D or len(pytorch_models) + len(torch_models) >= 4 or (MODEL_D_TYPE == "magenta" and magenta_styles and magenta_available)):
    num_models += 1

# Generate or parse blend weights
def generate_blend_weights(num_models):
    if BLEND_WEIGHTS:
        weights = [float(w) for w in BLEND_WEIGHTS.split(",")]
        if len(weights) != num_models:
            raise ValueError(f"BLEND_WEIGHTS must have {num_models} values, got {len(weights)}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"BLEND_WEIGHTS must sum to 1.0, got {sum(weights):.6f}")
        return weights
    if RANDOM_WEIGHTS:
        weights = [random.random() for _ in range(num_models)]
        total = sum(weights)
        weights = [w / total for w in weights]  # Normalize to sum to 1.0
        # Round to 3 decimal places and adjust last weight to ensure sum=1.0
        weights = [round(w, 3) for w in weights]
        if num_models > 1:
            weights[-1] = round(1.0 - sum(weights[:-1]), 3)
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

vids = chain(pathlib.Path(IN_DIR).glob("*.mp4"),
             pathlib.Path(IN_DIR).glob("*.mov"))

for vid in vids:
    # Select models
    available_pytorch = pytorch_models.copy()
    available_torch = torch_models.copy()
    available_styles = magenta_styles.copy()
    selected_models = []

    # Model A (required)
    if MODEL_A:
        if MODEL_A_TYPE == "magenta":
            if not MAGENTA_STYLE:
                raise ValueError("MAGENTA_STYLE required for model A when MODEL_A_TYPE=magenta")
            if not magenta_available:
                raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
            mA = MAGENTA_DIR  # Model path is magenta dir
            style_a = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE
            if not style_a.exists():
                raise ValueError(f"Style image {style_a} not found")
        elif MODEL_A_TYPE == "torch7":
            mA = pathlib.Path(TORCH_DIR) / MODEL_A
            if not mA.exists():
                raise ValueError(f"Torch7 model {mA} not found")
        else:  # transformer
            mA = pathlib.Path(PYTORCH_DIR) / MODEL_A
            if not mA.exists():
                raise ValueError(f"PyTorch model {mA} not found")
    else:
        if MODEL_A_TYPE == "magenta" and available_styles and magenta_available:
            mA = MAGENTA_DIR
            style_a = random.choice(available_styles)
            MAGENTA_STYLE = style_a.name
        elif MODEL_A_TYPE == "torch7" and available_torch:
            mA = random.choice(available_torch)
        elif available_pytorch:
            mA = random.choice(available_pytorch)
        else:
            raise ValueError("No suitable model available for A")
    selected_models.append(("A", mA, MODEL_A_TYPE, MAGENTA_STYLE))
    if MODEL_A_TYPE == "magenta" and MAGENTA_STYLE:
        style_path = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE
        if style_path in available_styles:
            available_styles.remove(style_path)
    elif MODEL_A_TYPE == "torch7" and mA in available_torch:
        available_torch.remove(mA)
    elif mA in available_pytorch:
        available_pytorch.remove(mA)

    # Model B
    mB = None
    style_b = None
    if num_models >= 2 and USE_B:
        if MODEL_B:
            if MODEL_B_TYPE == "magenta":
                if not MAGENTA_STYLE_B:
                    raise ValueError("MAGENTA_STYLE_B required for model B when MODEL_B_TYPE=magenta")
                if not magenta_available:
                    raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
                mB = MAGENTA_DIR
                style_b = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_B
                if not style_b.exists():
                    raise ValueError(f"Style image {style_b} not found")
            elif MODEL_B_TYPE == "torch7":
                mB = pathlib.Path(TORCH_DIR) / MODEL_B
                if not mB.exists():
                    raise ValueError(f"Torch7 model {mB} not found")
            else:
                mB = pathlib.Path(PYTORCH_DIR) / MODEL_B
                if not mB.exists():
                    raise ValueError(f"PyTorch model {mB} not found")
        else:
            if MODEL_B_TYPE == "magenta" and available_styles and magenta_available:
                mB = MAGENTA_DIR
                style_b = random.choice(available_styles)
                MAGENTA_STYLE_B = style_b.name
            elif MODEL_B_TYPE == "torch7" and available_torch:
                mB = random.choice(available_torch)
            elif available_pytorch:
                mB = random.choice(available_pytorch)
        if mB:
            selected_models.append(("B", mB, MODEL_B_TYPE, MAGENTA_STYLE_B))
            if MODEL_B_TYPE == "magenta" and style_b and style_b in available_styles:
                available_styles.remove(style_b)
            elif MODEL_B_TYPE == "torch7" and mB in available_torch:
                available_torch.remove(mB)
            elif mB in available_pytorch:
                available_pytorch.remove(mB)

    # Model C
    mC = None
    style_c = None
    if num_models >= 3 and USE_C:
        if MODEL_C:
            if MODEL_C_TYPE == "magenta":
                if not MAGENTA_STYLE_C:
                    raise ValueError("MAGENTA_STYLE_C required for model C when MODEL_C_TYPE=magenta")
                if not magenta_available:
                    raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
                mC = MAGENTA_DIR
                style_c = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_C
                if not style_c.exists():
                    raise ValueError(f"Style image {style_c} not found")
            elif MODEL_C_TYPE == "torch7":
                mC = pathlib.Path(TORCH_DIR) / MODEL_C
                if not mC.exists():
                    raise ValueError(f"Torch7 model {mC} not found")
            else:
                mC = pathlib.Path(PYTORCH_DIR) / MODEL_C
                if not mC.exists():
                    raise ValueError(f"PyTorch model {mC} not found")
        else:
            if MODEL_C_TYPE == "magenta" and available_styles and magenta_available:
                mC = MAGENTA_DIR
                style_c = random.choice(available_styles)
                MAGENTA_STYLE_C = style_c.name
            elif MODEL_C_TYPE == "torch7" and available_torch:
                mC = random.choice(available_torch)
            elif available_pytorch:
                mC = random.choice(available_pytorch)
        if mC:
            selected_models.append(("C", mC, MODEL_C_TYPE, MAGENTA_STYLE_C))
            if MODEL_C_TYPE == "magenta" and style_c and style_c in available_styles:
                available_styles.remove(style_c)
            elif MODEL_C_TYPE == "torch7" and mC in available_torch:
                available_torch.remove(mC)
            elif mC in available_pytorch:
                available_pytorch.remove(mC)

    # Model D
    mD = None
    style_d = None
    if num_models >= 4 and USE_D:
        if MODEL_D:
            if MODEL_D_TYPE == "magenta":
                if not MAGENTA_STYLE_D:
                    raise ValueError("MAGENTA_STYLE_D required for model D when MODEL_D_TYPE=magenta")
                if not magenta_available:
                    raise ValueError(f"Magenta model directory {MAGENTA_DIR} not found or empty")
                mD = MAGENTA_DIR
                style_d = pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_D
                if not style_d.exists():
                    raise ValueError(f"Style image {style_d} not found")
            elif MODEL_D_TYPE == "torch7":
                mD = pathlib.Path(TORCH_DIR) / MODEL_D
                if not mD.exists():
                    raise ValueError(f"Torch7 model {mD} not found")
            else:
                mD = pathlib.Path(PYTORCH_DIR) / MODEL_D
                if not mD.exists():
                    raise ValueError(f"PyTorch model {mD} not found")
        else:
            if MODEL_D_TYPE == "magenta" and available_styles and magenta_available:
                mD = MAGENTA_DIR
                style_d = random.choice(available_styles)
                MAGENTA_STYLE_D = style_d.name
            elif MODEL_D_TYPE == "torch7" and available_torch:
                mD = random.choice(available_torch)
            elif available_pytorch:
                mD = random.choice(available_pytorch)
        if mD:
            selected_models.append(("D", mD, MODEL_D_TYPE, MAGENTA_STYLE_D))
            if MODEL_D_TYPE == "magenta" and style_d and style_d in available_styles:
                available_styles.remove(style_d)
            elif MODEL_D_TYPE == "torch7" and mD in available_torch:
                available_torch.remove(mD)
            elif mD in available_pytorch:
                available_pytorch.remove(mD)

    # Ensure at least one model is selected
    if not selected_models:
        raise SystemExit("At least one model must be selected")

    # Generate blend weights
    weights = generate_blend_weights(num_models)
    weight_str = ",".join(f"{w:.3f}" for w in weights)

    # Generate output filename
    stem = vid.stem
    model_str = "_".join(f"{name}-{model.stem if model_type != 'magenta' else style.split('/')[-1].split('.')[0] if style else 'magenta'}" for name, model, model_type, style in selected_models)
    outp = pathlib.Path(OUT_DIR) / f"{stem}_{model_str}_w-{weight_str}.mp4"

    # Create work directory
    work = pathlib.Path(WORK_ROOT) / stem
    work.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", "-u", "/app/pipeline.py",
        "--model_type", MODEL_A_TYPE,
        "--model", str(mA),
        "--io_preset", IO_A,
        "--blend", f"{BLEND}",
        "--input_video", str(vid),
        "--output_video", str(outp),
        "--work_dir", str(work),
        "--image_ext", IMG_EXT, "--jpeg_quality", JPEG_QUALITY,
        "--scale", SCALE, "--fps", FPS,
        "--smooth_alpha", SMOOTH_ALPHA,
        "--flow_method", FLOW_METHOD, "--flow_downscale", FLOW_DOWNSCALE,
        "--max_frames", "5",
        "--magenta_model_root", MAGENTA_DIR,
    ]

    # Add Magenta style for model A
    if MODEL_A_TYPE == "magenta" and MAGENTA_STYLE:
        cmd += ["--magenta_style", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE)]

    # Add model B
    if mB:
        cmd += ["--model_b", str(mB), "--model_b_type", MODEL_B_TYPE, "--io_preset_b", IO_B]
        if MODEL_B_TYPE == "magenta" and MAGENTA_STYLE_B:
            cmd += ["--magenta_style_b", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_B)]

    # Add model C
    if mC:
        cmd += ["--model_c", str(mC), "--model_c_type", MODEL_C_TYPE, "--io_preset_c", IO_C]
        if MODEL_C_TYPE == "magenta" and MAGENTA_STYLE_C:
            cmd += ["--magenta_style_c", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_C)]

    # Add model D
    if mD:
        cmd += ["--model_d", str(mD), "--model_d_type", MODEL_D_TYPE, "--io_preset_d", IO_D]
        if MODEL_D_TYPE == "magenta" and MAGENTA_STYLE_D:
            cmd += ["--magenta_style_d", str(pathlib.Path(MAGENTA_STYLES_DIR) / MAGENTA_STYLE_D)]

    # Add blend weights
    if num_models > 1:
        cmd += ["--blend_models_weights", weight_str]
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
        cmd += ["--flow_ema", "--flow_alpha", FLOW_ALPHA]

    # Pre-FPS (pre-resample input before extraction)
    if PRE_FPS:
        cmd += ["--pre_fps", PRE_FPS]

    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)