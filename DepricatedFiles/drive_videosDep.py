import os
import random
import subprocess
import shlex
import pathlib
import shutil
import hashlib
from itertools import chain

# Artifact version
ARTIFACT_VERSION = "a1b2c3d4-5678-9012-abcd-34567890cdef"
print(f"[debug] drive_videos.py artifact version: {ARTIFACT_VERSION}")

# Inputs / outputs
IN_DIR = os.getenv("IN_DIR", "/app/input_videos")
OUT_DIR = os.getenv("OUT_DIR", "/app/output")
WORK_ROOT = os.getenv("WORK_ROOT", "/app/_work")
PYTORCH_DIR = os.getenv("PYTORCH_DIR", "/app/models/pytorch")
TORCH_DIR = os.getenv("TORCH_DIR", "/app/models/torch")
MAGENTA_DIR = os.getenv("MAGENTA_DIR", "/app/models/magenta")
MAGENTA_STYLES_DIR = os.getenv("MAGENTA_STYLES_DIR", "/app/models/magenta_styles")

# Video I/O
SCALE = os.getenv("SCALE", "720")
FPS = os.getenv("FPS", "24")
PRE_FPS = os.getenv("PRE_FPS", "15")
IMG_EXT = os.getenv("IMG_EXT", "jpg")
JPEG_QUALITY = os.getenv("JPEG_QUALITY", "85")

# Model I/O presets
IO_PRESETS = {
    "transformer": "imagenet_255",
    "torch7": "caffe_bgr",
    "magenta": "imagenet_01"
}

# Stylized↔original blend
BLEND = float(os.getenv("BLEND", "0.9"))
SMOOTH_LIGHT = os.getenv("SMOOTH_LIGHTNESS", "1") == "1"
SMOOTH_ALPHA = os.getenv("SMOOTH_ALPHA", "0.65")
FLOW_EMA = os.getenv("FLOW_EMA", "0") == "1"
FLOW_METHOD = os.getenv("FLOW_METHOD", "dis")
FLOW_DOWNSCALE = os.getenv("FLOW_DOWNSCALE", "1")

# Fixed blend weights (equal for each model)
BLEND_WEIGHTS = "0.25,0.25,0.25,0.25"

# Ensure output dir exists
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Clear entire work directory
if pathlib.Path(WORK_ROOT).exists():
    shutil.rmtree(WORK_ROOT)
    print(f"[debug] Cleared entire work directory {WORK_ROOT}")
pathlib.Path(WORK_ROOT).mkdir(parents=True, exist_ok=True)

# Collect models and styles
pytorch_models = list(pathlib.Path(PYTORCH_DIR).glob("*.pth"))
torch_models = list(pathlib.Path(TORCH_DIR).glob("*.t7"))
magenta_styles = list(pathlib.Path(MAGENTA_STYLES_DIR).glob("*.jpg"))

# Log available models and styles
print(f"[debug] Available PyTorch models: {[p.name for p in pytorch_models]}")
print(f"[debug] Available Torch7 models: {[p.name for p in torch_models]}")
print(f"[debug] Available Magenta styles: {[p.name for p in magenta_styles]}")

# Validate Magenta model directory
magenta_available = False
if pathlib.Path(MAGENTA_DIR).exists():
    magenta_subdirs = [d for d in pathlib.Path(MAGENTA_DIR).glob("*") if d.is_dir()]
    if magenta_subdirs:
        magenta_available = True
    else:
        print(f"[warning] No model subdirectory found in {MAGENTA_DIR}; Magenta model will not be used")
else:
    print(f"[warning] Magenta directory {MAGENTA_DIR} does not exist; Magenta model will not be used")

# Ensure sufficient models are available
available_models = pytorch_models + torch_models
if len(available_models) < 2 or (magenta_available and len(magenta_styles) < 2):
    raise SystemExit(f"Need at least 2 non-Magenta models in {PYTORCH_DIR} or {TORCH_DIR} and 2 styles in {MAGENTA_STYLES_DIR} for Magenta")

# Log parent environment to detect interference
print("[debug] Parent environment variables (before processing):")
for key, value in sorted(os.environ.items()):
    if key.startswith("MODEL_") or key.startswith("MAGENTA_STYLE") or key.startswith("IO_PRESET") or key == "BLEND_WEIGHTS":
        print(f"[debug]   {key}={value}")

# Process videos
vids = chain(pathlib.Path(IN_DIR).glob("*.mp4"), pathlib.Path(IN_DIR).glob("*.mov"))

for vid in vids:
    # Set random seed based on video name hash
    seed = int(hashlib.sha256(vid.name.encode()).hexdigest(), 16) % (2**32)
    random.seed(seed)
    # print(f"[debug] Video {vid.name}: Set random seed to {seed}")
    # print(f"[debug] Video {vid.name}: Random state: {random.getstate()}")

    # Shuffle model lists for this video
    pytorch_models_copy = pytorch_models.copy()
    torch_models_copy = torch_models.copy()
    magenta_styles_copy = magenta_styles.copy()
    random.shuffle(pytorch_models_copy)
    random.shuffle(torch_models_copy)
    random.shuffle(magenta_styles_copy)
    print(f"[debug] Video {vid.name}: Shuffled PyTorch models: {[p.name for p in pytorch_models_copy]}")
    print(f"[debug] Video {vid.name}: Shuffled Torch7 models: {[p.name for p in torch_models_copy]}")
    print(f"[debug] Video {vid.name}: Shuffled Magenta styles: {[p.name for p in magenta_styles_copy]}")

    # Randomly select two slots for Magenta
    slots = ["A", "B", "C", "D"]
    magenta_slots = random.sample(slots, 2)
    for slot in magenta_slots:
        slots.remove(slot)
    print(f"[debug] Video {vid.name}: Selected Magenta slots: {magenta_slots}")

    # Initialize model configuration
    model_config = {
        "A": {"model": "", "type": "", "style": ""},
        "B": {"model": "", "type": "", "style": ""},
        "C": {"model": "", "type": "", "style": ""},
        "D": {"model": "", "type": "", "style": ""}
    }

    # Assign Magenta to the selected slots with unique styles
    available_styles_copy = magenta_styles_copy.copy()
    if magenta_available:
        for slot in magenta_slots:
            if not available_styles_copy:
                raise SystemExit(f"Not enough unique Magenta styles for slot {slot} in video {vid.name}")
            style = random.choice(available_styles_copy)
            model_config[slot]["model"] = MAGENTA_DIR
            model_config[slot]["type"] = "magenta"
            model_config[slot]["style"] = style.name
            print(f"[debug] Video {vid.name}: Magenta model assigned to slot {slot} with style {model_config[slot]['style']}")
            available_styles_copy.remove(style)
    else:
        raise SystemExit("Magenta model not available; cannot proceed")

    # Assign random PyTorch or Torch7 models to remaining slots
    available_models_copy = pytorch_models_copy + torch_models_copy
    for slot in slots:
        if not available_models_copy:
            print(f"[warning] Video {vid.name}: Not enough unique models for slot {slot}; skipping")
            model_config[slot]["model"] = ""
            model_config[slot]["type"] = ""
            continue
        model = random.choice(available_models_copy)
        model_config[slot]["model"] = str(model)
        model_config[slot]["type"] = "transformer" if model.suffix == ".pth" else "torch7"
        print(f"[debug] Video {vid.name}: Assigned {model_config[slot]['type']} model {model_config[slot]['model']} to slot {slot} with io_preset {IO_PRESETS[model_config[slot]['type']]}")
        available_models_copy.remove(model)

    # Generate output filename
    stem = vid.stem
    model_str = "_".join(
        f"{name}-{model_config[name]['style'].split('.')[0] if model_config[name]['type'] == 'magenta' else pathlib.Path(model_config[name]['model']).stem}"
        for name in ["A", "B", "C", "D"] if model_config[name]["model"]
    )
    outp = pathlib.Path(OUT_DIR) / f"{stem}_{model_str}_w-{BLEND_WEIGHTS}.mp4"
    print(f"[debug] Video {vid.name}: Expected output filename: {outp}")

    # Build environment variables for run_videos.py
    env = {"PATH": os.environ.get("PATH", "")}
    env.update({
        "IN_DIR": IN_DIR,
        "OUT_DIR": OUT_DIR,
        "PYTORCH_DIR": PYTORCH_DIR,
        "TORCH_DIR": TORCH_DIR,
        "MAGENTA_DIR": MAGENTA_DIR,
        "MAGENTA_STYLES_DIR": MAGENTA_STYLES_DIR,
        "SCALE": SCALE,
        "FPS": FPS,
        "PRE_FPS": PRE_FPS,
        "IMG_EXT": IMG_EXT,
        "JPEG_QUALITY": JPEG_QUALITY,
        "BLEND": str(BLEND),
        "SMOOTH_LIGHTNESS": "1" if SMOOTH_LIGHT else "0",
        "SMOOTH_ALPHA": SMOOTH_ALPHA,
        "FLOW_EMA": "1" if FLOW_EMA else "0",
        "FLOW_METHOD": FLOW_METHOD,
        "FLOW_DOWNSCALE": FLOW_DOWNSCALE,
        "OPENCV_OPENCL_DEVICE": "disabled",
        "OPENCV_NUM_THREADS": "1",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "BLEND_WEIGHTS": BLEND_WEIGHTS
    })

    # Add model-specific environment variables
    for slot in ["A", "B", "C", "D"]:
        if model_config[slot]["model"]:
            env[f"MODEL_{slot}"] = model_config[slot]["model"]
            env[f"MODEL_{slot}_TYPE"] = model_config[slot]["type"]
            env[f"IO_PRESET_{slot}"] = IO_PRESETS[model_config[slot]["type"]]
            print(f"[debug] Video {vid.name}: Set IO_PRESET_{slot}={IO_PRESETS[model_config[slot]['type']]} for model {model_config[slot]['model']}")
            if model_config[slot]["type"] == "magenta":
                env_key = "MAGENTA_STYLE" if slot == "A" else f"MAGENTA_STYLE_{slot}"
                env[env_key] = model_config[slot]["style"]
                print(f"[debug] Video {vid.name}: Set {env_key}={model_config[slot]['style']} for slot {slot}")
        else:
            env[f"USE_{slot}"] = "0"

    # Log all environment variables for debugging
    print(f"[debug] Video {vid.name}: Environment variables for run_videos.py:")
    for key, value in sorted(env.items()):
        if key.startswith("MODEL_") or key.startswith("MAGENTA_STYLE") or key.startswith("IO_PRESET") or key == "BLEND_WEIGHTS":
            print(f"[debug]   {key}={value}")

    # Run run_videos.py
    cmd = ["python3", "/app/run_videos.py", str(vid)]
    print(f"[drive] Video {vid.name}: Running: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] Video {vid.name}: run_videos.py failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"[error] Video {vid.name}: Interrupted by user")
        raise