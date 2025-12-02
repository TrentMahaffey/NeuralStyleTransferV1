import os
import random
import subprocess
import shlex
import pathlib
import shutil
import hashlib
import json
from itertools import chain
from decimal import Decimal, getcontext, ROUND_HALF_UP
import time

# --- Helpers for explicit model selection -----------------------------------

def parse_models_env(models_str: str):
    models = []
    if not models_str:
        return models
    parts = [p.strip() for p in models_str.split(',') if p.strip()]
    for raw in parts:
        if ':' in raw:
            kind, ident = raw.split(':', 1)
            kind = kind.strip().lower()
            ident = ident.strip()
        else:
            kind, ident = 'transformer', raw.strip()
        models.append({'type': kind, 'id': ident})
    return models


def resolve_model_for_slot(m, PYTORCH_DIR, TORCH_DIR, MAGENTA_DIR, MAGENTA_STYLES_DIR):
    """Return tuple (model_value, model_type, magenta_style).
    Notes: pipeline expects types in {transformer,reconet,magenta,torch7}; we map pytorch→transformer.
    For magenta, model_value can be MAGENTA_DIR.
    """
    t = m['type'].lower()
    ident = m['id']
    if t == 'magenta':
        # style can be absolute path or a filename under MAGENTA_STYLES_DIR
        style_path = pathlib.Path(ident)
        style_val = str(style_path if style_path.is_absolute() else pathlib.Path(MAGENTA_STYLES_DIR) / ident)
        return (MAGENTA_DIR, 'magenta', pathlib.Path(style_val).name if not style_path.is_absolute() else style_val)
    if t == 'pytorch':
        p = pathlib.Path(ident)
        if not p.is_absolute():
            p = pathlib.Path(PYTORCH_DIR) / (ident if pathlib.Path(ident).suffix else f"{ident}.pth")
        return (str(p), 'transformer', '')
    if t == 'torch7':
        p = pathlib.Path(ident)
        if not p.is_absolute():
            p = pathlib.Path(TORCH_DIR) / (ident if pathlib.Path(ident).suffix else f"{ident}.t7")
        return (str(p), 'torch7', '')
    # transformer / reconet etc. treated like file under TRANSFORMER_DIR if relative; fall back to given ident
    p = pathlib.Path(ident)
    return (str(p), t, '')


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def equal_weights_csv(n: int) -> str:
    """Return CSV of n weights that sum to exactly 1.000000 at 6dp.
    Uses Decimal to avoid float rounding drift that can trip strict validators.
    """
    if n <= 0:
        return ''
    getcontext().prec = 28
    unit = Decimal('1.000000')
    step = (unit / Decimal(n)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
    weights = [step] * (n - 1)
    # force exact remainder on the last term
    last = (unit - sum(weights)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
    weights.append(last)
    # format with 6 fixed decimals
    return ','.join(f"{w:.6f}" for w in weights)

# --- Video duration probe helper using ffprobe ---
def probe_duration_seconds(video_path: pathlib.Path) -> float:
    """Return duration (seconds) using ffprobe; fall back to 0.0 on error."""
    try:
        # ask ffprobe for only the format duration to keep it fast
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "ignore").strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0

# --- Helper to detect new mp4 output file in OUT_DIR ---
def _detect_new_mp4(out_dir: pathlib.Path, before_paths: set[pathlib.Path], baseline_mtime: float) -> pathlib.Path | None:
    """Detect a newly written/updated .mp4 in out_dir.

    We prefer files whose mtime is strictly greater than the baseline_mtime snapshot
    (taken immediately before the render). This handles the case where run_videos
    overwrites a deterministic filename that already existed (so path ∈ before_paths).
    As a fallback, we look for paths not present in before_paths. Finally, we return
    the most recently modified .mp4 if nothing else matches.
    """
    # 1) Freshly modified after the snapshot
    fresh = [p for p in out_dir.glob("*.mp4") if p.stat().st_mtime > baseline_mtime + 1e-6]
    if fresh:
        return max(fresh, key=lambda p: p.stat().st_mtime)

    # 2) New paths that weren't present before (covers first-time runs)
    newcomers = [p for p in out_dir.glob("*.mp4") if p not in before_paths]
    if newcomers:
        return max(newcomers, key=lambda p: p.stat().st_mtime)

    # 3) Fallback to latest modified .mp4
    candidates = list(out_dir.glob("*.mp4"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None

# --- Montage helpers ---------------------------------------------------------

def _truthy(v: str) -> bool:
    return str(v).lower() in {"1","true","yes","on"}


def make_style_label(m: dict) -> str:
    t = m.get("type","")
    i = m.get("id","")
    base = pathlib.Path(i).stem if i else t
    return f"{t}-{base}".replace(" ", "_")


def render_single_model_clip(
    vid: pathlib.Path,
    m: dict,
    idx: int,
    base_env: dict,
    max_frames: int | None = None,
    start_secs: float | None = None,
    dur_secs: float | None = None
) -> pathlib.Path:
    """Render a clip using only slot A for a single model. Returns the output .mp4 path.
    Optionally trims the input to a temporary segment using start_secs and dur_secs.
    """
    env = dict(base_env)
    if max_frames is not None:
        env["MAX_FRAMES"] = str(max_frames)
    model_val, model_type, style = resolve_model_for_slot(m, PYTORCH_DIR, TORCH_DIR, MAGENTA_DIR, MAGENTA_STYLES_DIR)
    env["MODEL_A"] = model_val
    env["MODEL_A_TYPE"] = model_type
    env["IO_PRESET_A"] = IO_PRESETS.get(model_type, IO_PRESETS.get("transformer", "imagenet_255"))
    if model_type == "magenta":
        env["MAGENTA_STYLE"] = style
    label = make_style_label(m)
    env["OUTPUT_SUFFIX"] = f"_{label}"  # run_videos will add its own mbXX/mXX; avoid duplication
    input_for_run = vid
    if start_secs is not None or dur_secs is not None:
        seg_path = pathlib.Path(WORK_ROOT) / f"seg_{vid.stem}_m{idx:02d}.mp4"
        cmd_trim = ["ffmpeg", "-y"]
        if start_secs is not None:
            cmd_trim += ["-ss", f"{start_secs:.3f}"]
        cmd_trim += ["-i", str(vid)]
        if dur_secs is not None:
            cmd_trim += ["-t", f"{dur_secs:.3f}"]
        cmd_trim += ["-c", "copy", str(seg_path)]
        print(f"[montage] Trimming single-model source: start={start_secs} dur={dur_secs} → {seg_path.name}")
        subprocess.run(cmd_trim, check=True)
        input_for_run = seg_path
    cmd = ["python3", "/app/run_videos.py", str(input_for_run)]
    print(f"[montage] Rendering model {idx}: {label} (suffix={env['OUTPUT_SUFFIX']})")
    out_dir_path = pathlib.Path(OUT_DIR)
    before_set = set(out_dir_path.glob("*.mp4"))
    baseline_mtime = max([p.stat().st_mtime for p in before_set], default=0.0)
    subprocess.run(cmd, env=env, check=True)
    detected = _detect_new_mp4(out_dir_path, before_set, baseline_mtime)
    if not detected:
        # Fall back: choose any file whose name contains the label, prefer newest
        pattern = f"*{label}*.mp4"
        matches = sorted(out_dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            detected = matches[0]
    if not detected:
        # Additional best-effort guesses (with/without 'seg_' prefix)
        guesses = [
            out_dir_path / f"{vid.stem}_{label}.mp4",
            out_dir_path / f"seg_{vid.stem}_{label}.mp4",
        ]
        for g in guesses:
            if g.exists():
                detected = g
                break
    if not detected:
        raise RuntimeError(f"Could not locate output file for model {idx} ({label}); check run logs.")
    return detected

# --- Batch clip rendering helper ---
def render_batch_clip(
    vid: pathlib.Path,
    models: list[dict],
    idx: int,
    base_env: dict,
    max_frames: int | None = None,
    start_secs: float | None = None,
    dur_secs: float | None = None
) -> pathlib.Path:
    """Render a clip using up to 4 models blended together (slots A..D). Returns the output .mp4 path.
    Optionally trims the input to a temporary segment using start_secs and dur_secs.
    """
    env = dict(base_env)
    if max_frames is not None:
        env["MAX_FRAMES"] = str(max_frames)
    slots = ["A", "B", "C", "D"]
    resolved = []
    for slot, m in zip(slots, models):
        model_val, model_type, style = resolve_model_for_slot(m, PYTORCH_DIR, TORCH_DIR, MAGENTA_DIR, MAGENTA_STYLES_DIR)
        env[f"MODEL_{slot}"] = model_val
        env[f"MODEL_{slot}_TYPE"] = model_type
        env[f"IO_PRESET_{slot}"] = IO_PRESETS.get(model_type, IO_PRESETS.get("transformer", "imagenet_255"))
        if model_type == "magenta":
            env_key = "MAGENTA_STYLE" if slot == "A" else f"MAGENTA_STYLE_{slot}"
            env[env_key] = style
        resolved.append(f"{m.get('type')}:{m.get('id')}")

    # Blend weights: respect explicit BLEND_WEIGHTS, otherwise equal per active slot count
    user_bw = os.getenv("BLEND_WEIGHTS", "").strip()
    env["BLEND_WEIGHTS"] = user_bw if user_bw else equal_weights_csv(len(models))

    # Label for filename: join short labels of models in this batch
    label = "+".join(make_style_label(m) for m in models)
    env["OUTPUT_SUFFIX"] = f"_{label}"  # let run_videos add its own mbXX to the basename

    input_for_run = vid
    if start_secs is not None or dur_secs is not None:
        seg_path = pathlib.Path(WORK_ROOT) / f"seg_{vid.stem}_mb{idx:02d}.mp4"
        cmd_trim = ["ffmpeg", "-y"]
        if start_secs is not None:
            cmd_trim += ["-ss", f"{start_secs:.3f}"]
        cmd_trim += ["-i", str(vid)]
        if dur_secs is not None:
            cmd_trim += ["-t", f"{dur_secs:.3f}"]
        cmd_trim += ["-c", "copy", str(seg_path)]
        print(f"[montage] Trimming batch-model source: start={start_secs} dur={dur_secs} → {seg_path.name}")
        subprocess.run(cmd_trim, check=True)
        input_for_run = seg_path
    cmd = ["python3", "/app/run_videos.py", str(input_for_run)]
    print(f"[montage] Rendering batch {idx} ({len(models)} model(s)): {resolved} (suffix={env['OUTPUT_SUFFIX']})")
    out_dir_path = pathlib.Path(OUT_DIR)
    before_set = set(out_dir_path.glob("*.mp4"))
    baseline_mtime = max([p.stat().st_mtime for p in before_set], default=0.0)
    subprocess.run(cmd, env=env, check=True)
    detected = _detect_new_mp4(out_dir_path, before_set, baseline_mtime)
    if not detected:
        # Fall back: choose any file whose name contains the label, prefer newest
        pattern = f"*{label}*.mp4"
        matches = sorted(out_dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            detected = matches[0]
    if not detected:
        # Additional best-effort guesses (with/without 'seg_' prefix)
        guesses = [
            out_dir_path / f"{vid.stem}_{label}.mp4",
            out_dir_path / f"seg_{vid.stem}_{label}.mp4",
        ]
        for g in guesses:
            if g.exists():
                detected = g
                break
    if not detected:
        raise RuntimeError(f"Could not locate output file for batch {idx} ({label}); check run logs.")
    return detected


def render_original_clip(
    vid: pathlib.Path,
    idx: int,
    target_scale: str,
    target_fps: str,
    dur_secs: float | None = None
) -> pathlib.Path:
    """Re-encode original to match scale/fps of styled clips for smooth xfade. Duration is exactly 'dur_secs' when provided."""
    out_path = pathlib.Path(OUT_DIR) / f"{vid.stem}_m{idx:02d}_original.mp4"
    vf = f"scale='if(gte(iw,ih),{target_scale},-2)':'if(gte(ih,iw),{target_scale},-2)':flags=lanczos,fps={target_fps}"
    cmd = [
        "ffmpeg","-y","-i", str(vid),
        "-vf", vf,
        "-c:v","libx264","-pix_fmt","yuv420p",
    ]
    if dur_secs is not None:
        cmd.extend(["-t", f"{dur_secs}"])
    cmd.append(str(out_path))
    print(f"[montage] Rendering original clip → {out_path.name}")
    subprocess.run(cmd, check=True)
    return out_path


def assemble_montage(clips: list[pathlib.Path], output_path: pathlib.Path, segment_secs: float, fade_secs: float, intro_secs: float) -> None:
    """Assemble one final montage with equal-length segments and crossfades.
    - clips[0] is the original; use INTRO_SECS for it, SEGMENT_SECS for the rest.
    - Video-only assembly (no audio) to keep the graph simple.
    """
    if not clips:
        raise SystemExit("assemble_montage: no clips to assemble")

    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    # Build filter graph: trim each input, then chain xfades
    filter_parts = []
    labels = []
    for i, _ in enumerate(clips):
        dur = intro_secs if i == 0 else segment_secs
        # Force constant frame rate for xfade stability .
        fps_str = os.getenv("FPS", "24")
        filter_parts.append(
            f"[{i}:v]trim=duration={dur},setpts=PTS-STARTPTS,fps=fps={fps_str}[v{i}]"
        )
        labels.append(f"[v{i}]")

    # Chain xfades: [v0][v1]xfade=... [x1]; [x1][v2]xfade=... [x2] ...
    # IMPORTANT: xfade's `offset` is relative to the FIRST input of each xfade.
    # After each xfade, the resulting clip length becomes (offset + duration),
    # so to preserve total duration we must accumulate offsets across the chain.
    out_label = "[v0]"
    x_idx = 0

    # For the first transition, the first stream is the intro clip of length `intro_secs`.
    # The transition should start at (intro_secs - fade_secs) so that the crossfade
    # overlaps exactly `fade_secs` with the next clip.
    cumulative_offset = max(0.0, intro_secs - fade_secs)

    for i in range(1, len(clips)):
        x_idx += 1
        next_in = f"[v{i}]"
        out = f"[x{x_idx}]"
        # Round for friendlier logs and to avoid float noise in ffmpeg graphs
        off_str = f"{cumulative_offset:.6f}"
        filter_parts.append(
            f"{out_label}{next_in}xfade=transition=fade:duration={fade_secs}:offset={off_str}{out}"
        )
        out_label = out
        # After each xfade, the effective length of the result is (offset + fade_secs).
        # For subsequent transitions, we need to advance by (segment_secs - fade_secs)
        # because each non-intro segment is trimmed to `segment_secs`.
        cumulative_offset += max(0.0, segment_secs - fade_secs)

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg","-y", *inputs, "-filter_complex", filter_complex, "-map", out_label, "-c:v","libx264","-pix_fmt","yuv420p", str(output_path)]
    print(f"[montage] Assembling {len(clips)} clips → {output_path.name}")
    # For debug: print graph
    print(f"[montage] filter_complex=\n{filter_complex}")
    subprocess.run(cmd, check=True)

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
    # --- Deterministic/explicit model selection path (when MODELS is provided) ---
    models_env = os.getenv("MODELS", "").strip()
    if models_env:
        explicit_models = parse_models_env(models_env)
        # Optional shuffle while preserving determinism per video
        if os.getenv("SHUFFLE_MODELS", "0").lower() in {"1","true","yes","on"}:
            seed = int(hashlib.sha256((vid.name + "::models").encode()).hexdigest(), 16) % (2**32)
            random.Random(seed).shuffle(explicit_models)
        # Optional cap
        max_models = int(os.getenv("MAX_MODELS", "0") or 0)
        if max_models > 0:
            explicit_models = explicit_models[:max_models]
        if not explicit_models:
            raise SystemExit("MODELS was set but empty after filtering")

        # Determine how to handle >4 models
        strategy = os.getenv("CHUNK_STRATEGY", "chunk").lower()  # one of: chunk, error, clamp4, pad
        print(f"[plan] {vid.name}: CHUNK_STRATEGY={strategy}")

        if strategy not in {"chunk", "error", "clamp4", "pad"}:
            strategy = "chunk"

        models_for_processing = list(explicit_models)

        # Montage fast-path: render one clip per model or batch, then crossfade into a single montage
        if _truthy(os.getenv("MONTAGE", "0")):
            seg_secs = float(os.getenv("SEGMENT_SECS", "2"))
            fade_secs = float(os.getenv("FADE_SECS", "0.5"))
            intro_secs = float(os.getenv("INTRO_SECS", "1"))

            batch_size = int(os.getenv("MONTAGE_BATCH_SIZE", "4"))
            if batch_size < 1:
                batch_size = 1

            # Compute number of generated styled clips (N)
            if batch_size == 1:
                N = len(models_for_processing)
            else:
                bs = min(4, batch_size)
                N = len(list(chunked(models_for_processing, bs)))

            # Probe source duration
            src_dur = probe_duration_seconds(vid)

            # Optional dynamic segment sizing to fill (approximately) the source after the intro
            # Enabled by default: AUTO_SEGMENT=1; to turn off, set AUTO_SEGMENT=0 and provide SEGMENT_SECS
            auto_seg = os.getenv("AUTO_SEGMENT", "1").lower() in {"1","true","yes","on"}
            if auto_seg and N > 0:
                remaining = max(0.0, src_dur - intro_secs)
                # Compensate for crossfades so total montage duration matches the source:
                # total = intro + N*seg_secs - N*fade_secs == src_dur  =>  seg_secs = remaining/N + fade_secs
                seg_secs = (remaining / float(N)) + fade_secs
                # Ensure seg_secs is at least a minimal positive value and not less than fade length
                seg_secs = max(seg_secs, max(0.1, fade_secs))
                print(f"[montage] AUTO_SEGMENT active: src_dur={src_dur:.3f}s intro={intro_secs:.3f}s N={N} fade={fade_secs:.3f}s → seg_secs={seg_secs:.3f}s (trim per styled clip; fade-compensated)")
            else:
                print(f"[montage] Fixed SEGMENT_SECS={seg_secs:.3f}s (AUTO_SEGMENT disabled or N=0)")

            # --- Per-clip frame budgets so we don't process the whole video for short segments ---
            try:
                pre_fps_val = float(PRE_FPS)
            except Exception:
                pre_fps_val = 15.0

            def _round_half_up(x: float) -> int:
                from decimal import Decimal, ROUND_HALF_UP
                return int(Decimal(str(x)).to_integral_value(rounding=ROUND_HALF_UP))

            # Per-clip frame budgets (do not add fade_secs; overlap is handled by xfade and sequential starts)
            intro_frames = _round_half_up(pre_fps_val * (intro_secs))
            seg_frames   = _round_half_up(pre_fps_val * (seg_secs))

            # If the user set MAX_FRAMES, cap per-clip budgets to that value
            _user_max_frames = os.getenv("MAX_FRAMES", "").strip()
            if _user_max_frames.isdigit():
                _umf = int(_user_max_frames)
                if _umf > 0:
                    intro_frames = min(intro_frames, _umf)
                    seg_frames = min(seg_frames, _umf)

            # Base environment carried to each single-model/batch render
            base_env = {"PATH": os.environ.get("PATH", "")}
            base_env.update({
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
                "MOTION_BLEND": os.getenv("MOTION_BLEND", "0"),
                "FLOW_ALPHA": os.getenv("FLOW_ALPHA", "0.3"),
                "OPENCV_OPENCL_DEVICE": "disabled",
                "OPENCV_NUM_THREADS": "1",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_CPP_MIN_LOG_LEVEL": "2",
            })

            # Render original clip (duration = intro_secs)
            original_clip = render_original_clip(vid, 0, SCALE, FPS, dur_secs=intro_secs)

            # How to pick source segments for each generated clip:
            # - 'sequential' (default): walk forward with (segment_secs - fade_secs) overlaps
            # - 'spread': evenly spread across the full source duration
            layout = os.getenv("MONTAGE_LAYOUT", "sequential").lower()
            if layout not in {"sequential", "spread"}:
                layout = "sequential"

            # Montage: compute start offsets for each generated clip
            clip_len = seg_secs  # render exactly the trim length; overlap is created by xfade and sequential starts
            starts: list[float] = []

            # Start the first styled segment so that its first `fade_secs` overlap
            # aligns with the end of the intro segment. If intro is shorter than
            # fade, clamp to 0.
            base_start = max(0.0, intro_secs - fade_secs)

            if layout == "spread":
                # Evenly distribute styled clips across the *remaining* timeline
                # after the intro, honoring clip length and clamping to the source.
                if N <= 1 or src_dur <= clip_len:
                    starts = [base_start] * max(N, 1)
                    stride = 0.0
                else:
                    # We want the set of starts to span from base_start to the last valid start.
                    last_valid = max(src_dur - clip_len, 0.0)
                    span = max(0.0, last_valid - base_start)
                    stride = span / float(max(N - 1, 1))
                    starts = [round(min(base_start + i * stride, last_valid), 3) for i in range(0, N)]
            else:
                # sequential: step forward by (segment_secs - fade_secs) so there is exactly
                # 'fade_secs' of content overlap between consecutive styled clips.
                step = max(seg_secs - fade_secs, 0.0)
                stride = step  # for logging consistency
                for i in range(N):
                    s = round(base_start + i * step, 3)
                    # clamp so we never seek past the last available segment
                    s = min(s, max(src_dur - clip_len, 0.0))
                    starts.append(s)

            print(f"[montage] Source duration={src_dur:.3f}s  clip_len={clip_len:.3f}s  N={N}  layout={layout}  base_start={base_start:.3f}s  stride/step={stride:.3f}s")

            # Decide between single-model clips vs batch clips .
            clips = []
            if batch_size == 1:
                for i, m in enumerate(models_for_processing, start=1):
                    start = starts[i-1]
                    clip = render_single_model_clip(vid, m, i, base_env, max_frames=seg_frames, start_secs=start, dur_secs=clip_len)
                    clips.append(clip)
            else:
                bs = min(4, batch_size)
                groups = list(chunked(models_for_processing, bs))
                for i, group in enumerate(groups, start=1):
                    start = starts[i-1]
                    clip = render_batch_clip(vid, group, i, base_env, max_frames=seg_frames, start_secs=start, dur_secs=clip_len)
                    clips.append(clip)

            # Assemble final montage from original + generated clips
            final_out = pathlib.Path(OUT_DIR) / f"{vid.stem}_montage.mp4"
            assemble_montage([original_clip] + clips, final_out, seg_secs, fade_secs, intro_secs)
            print(f"✅ Montage created: {final_out}")
            continue

        if strategy == "error" and len(models_for_processing) > 4:
            raise SystemExit(f"More than 4 models ({len(models_for_processing)}) provided and CHUNK_STRATEGY=error")

        if strategy == "clamp4" and len(models_for_processing) > 4:
            models_for_processing = models_for_processing[:4]

        if strategy == "pad" and 1 <= len(models_for_processing) <= 4:
            # duplicate the last model until we have 4; helps keep A..D filled for consistent pipelines
            while len(models_for_processing) < 4:
                models_for_processing.append(models_for_processing[-1])

        # Prepare chunks of up to 4 (pipeline supports A..D)
        chunks = list(chunked(models_for_processing, 4))
        multi = len(chunks) > 1

        # Common base env (carry through important knobs, including MAX_FRAMES if set)
        base_env = {"PATH": os.environ.get("PATH", "")}
        base_env.update({
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
            "MOTION_BLEND": os.getenv("MOTION_BLEND", "0"),
            "FLOW_ALPHA": os.getenv("FLOW_ALPHA", "0.3"),
            "OPENCV_OPENCL_DEVICE": "disabled",
            "OPENCV_NUM_THREADS": "1",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        })
        # Pass through user MAX_FRAMES if provided (single-chunk/explicit path uses inherited env)
        _mf_env = os.getenv("MAX_FRAMES", "").strip()
        if _mf_env:
            base_env["MAX_FRAMES"] = _mf_env

        # Process one or more chunks
        for idx, ch in enumerate(chunks, start=1):
            env = dict(base_env)
            slots = ["A","B","C","D"]
            # Compute blend weights for the number of models in this chunk, respecting user override
            user_bw = os.getenv("BLEND_WEIGHTS", "").strip()
            env["BLEND_WEIGHTS"] = user_bw if user_bw else (equal_weights_csv(len(ch)) or BLEND_WEIGHTS)

            # Map models to slots
            for slot, m in zip(slots, ch):
                model_val, model_type, style = resolve_model_for_slot(m, PYTORCH_DIR, TORCH_DIR, MAGENTA_DIR, MAGENTA_STYLES_DIR)
                env[f"MODEL_{slot}"] = model_val
                env[f"MODEL_{slot}_TYPE"] = model_type
                env[f"IO_PRESET_{slot}"] = IO_PRESETS.get(model_type, IO_PRESETS.get("transformer", "imagenet_255"))
                if model_type == "magenta":
                    env_key = "MAGENTA_STYLE" if slot == "A" else f"MAGENTA_STYLE_{slot}"
                    # If style is absolute path, keep as is; run_videos.py will pass it through
                    env[env_key] = style

            # If multiple chunks, add an OUTPUT_SUFFIX for unique filenames
            if multi:
                env["OUTPUT_SUFFIX"] = f"_set{idx:02d}"

            # Print summary for this chunk: slot count and blend weights
            print(f"[plan] {vid.name}: chunk {idx}/{len(chunks)} uses {len(ch)} model(s), BLEND_WEIGHTS={env['BLEND_WEIGHTS']}")

            # Log environment for debugging
            print(f"[plan] {vid.name}: MODELS override active (chunk {idx}/{len(chunks)}): {[m['type']+':'+m['id'] for m in ch]}")
            print(f"[plan] {vid.name}: BLEND_WEIGHTS={env['BLEND_WEIGHTS']}")
            for key, value in sorted(env.items()):
                if key.startswith("MODEL_") or key.startswith("MAGENTA_STYLE") or key.startswith("IO_PRESET") or key in {"BLEND_WEIGHTS","OUTPUT_SUFFIX","MAX_FRAMES"}:
                    print(f"[debug]   {key}={value}")

            # Run run_videos.py for this chunk
            cmd = ["python3", "/app/run_videos.py", str(vid)]
            print(f"[drive] Video {vid.name}: Running (chunk {idx}/{len(chunks)}): {' '.join(shlex.quote(c) for c in cmd)}")
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] Video {vid.name}: run_videos.py failed with exit code {e.returncode}")
            except KeyboardInterrupt:
                print(f"[error] Video {vid.name}: Interrupted by user")
                raise
        # Done with explicit MODELS path; skip random selection
        continue

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

    # Assign random PyTorch or Torch7 models to remaining slots .
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
        "MOTION_BLEND": os.getenv("MOTION_BLEND", "0"),
        "FLOW_ALPHA": os.getenv("FLOW_ALPHA", "0.3"),
        "OPENCV_OPENCL_DEVICE": "disabled",
        "OPENCV_NUM_THREADS": "1",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "BLEND_WEIGHTS": BLEND_WEIGHTS
    })
    # Pass through user MAX_FRAMES if provided .
    _mf_env2 = os.getenv("MAX_FRAMES", "").strip()
    if _mf_env2:
        env["MAX_FRAMES"] = _mf_env2

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

    # Log all environment variables for debugging .
    print(f"[debug] Video {vid.name}: Environment variables for run_videos.py:")
    for key, value in sorted(env.items()):
        if key.startswith("MODEL_") or key.startswith("MAGENTA_STYLE") or key.startswith("IO_PRESET") or key in {"BLEND_WEIGHTS","MAX_FRAMES"}:
            print(f"[debug]   {key}={value}")

    # Run run_videos.py .
    cmd = ["python3", "/app/run_videos.py", str(vid)]
    print(f"[drive] Video {vid.name}: Running: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] Video {vid.name}: run_videos.py failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"[error] Video {vid.name}: Interrupted by user")
        raise