#!/usr/bin/env bash
set -euo pipefail

# --- Config (env-overridable) ---
IN_DIR=${IN_DIR:-/app/input}
OUT_DIR=${OUT_DIR:-/app/output}
TMP=${TMP:-/app/work}
FPS=${FPS:-24}
SCALE=${SCALE:-720}
IMG_Q=${IMG_Q:-90}
HOLD_ORIG_START=${HOLD_ORIG_START:-1}  # hold on the initial original still .
HOLD_ORIG_END=${HOLD_ORIG_END:-1}      # hold on the final original still
HOLD_MODEL=${HOLD_MODEL:-1}            # hold on each model still .
TRANS=${TRANS:-1}
TRANSITION=${TRANSITION:-fade}   # e.g., dissolve, fade, smoothleft/right/up/down, circleopen/close, wipe*, slide*, pixelize, radial, fadeblack/white, distance, fadegrays
INCLUDE_ALLFOUR=${INCLUDE_ALLFOUR:-0}   # 1 = include 'allfour' composite in the morph sequence; 0 = skip it

CRF=${CRF:-18}
PRESET=${PRESET:-slow}
REV="2025-09-10-magenta"
echo "[script] morph.sh REV=${REV}"

# Default Magenta style directory (env-overridable)
MAGENTA_STYLE_DIR=${MAGENTA_STYLE_DIR:-/app/magenta_styles}
# Default Magenta target resolution (env-overridable, defaults to pipeline scale)
MAGENTA_TARGET_RES=${MAGENTA_TARGET_RES:-$SCALE}

# Cap how many model styles to run per image; randomized selection from the pool
MAX_MODELS=${MAX_MODELS:-20}          # hard cap per image (excludes the two 'orig' entries)
RANDOMIZE_MODELS=${RANDOMIZE_MODELS:-1}  # 1=randomize (seeded per image); 0=take first MAX_MODELS in order
RANDOM_SEED=${RANDOM_SEED:-}          # optional global seed; if empty we seed with the image base name

# Cap and randomize number of 2-model blends per image (env-overridable)
MAX_COMBOS=${MAX_COMBOS:-0}            # 0 = no cap; >0 = limit number of 2-model blends per image
RANDOMIZE_COMBOS=${RANDOMIZE_COMBOS:-1}  # 1=randomize selected combos; 0=take first N in deterministic order

# Normalize numeric envs to avoid test errors if passed empty/non-numeric
case "${MAX_MODELS:-}" in (''|*[!0-9]*) MAX_MODELS=20;; esac
case "${MAX_COMBOS:-}" in (''|*[!0-9]*) MAX_COMBOS=0;; esac
case "${RANDOMIZE_MODELS:-}" in (''|*[!0-9]*) RANDOMIZE_MODELS=1;; esac
case "${RANDOMIZE_COMBOS:-}" in (''|*[!0-9]*) RANDOMIZE_COMBOS=1;; esac

# Enable bash trace and print config when DEBUG=1 .                        .
if [[ "${DEBUG:-0}" -eq 1 ]]; then
  set -x
  echo "[cfg] IN_DIR=${IN_DIR} OUT_DIR=${OUT_DIR} TMP=${TMP} FPS=${FPS} SCALE=${SCALE} HOLD_ORIG_START=${HOLD_ORIG_START} HOLD_ORIG_END=${HOLD_ORIG_END} HOLD_MODEL=${HOLD_MODEL} TRANS=${TRANS} TRANSITION=${TRANSITION} CRF=${CRF} PRESET=${PRESET} MAGENTA_STYLE_DIR=${MAGENTA_STYLE_DIR} MAGENTA_TARGET_RES=${MAGENTA_TARGET_RES} MAX_MODELS=${MAX_MODELS} RANDOMIZE_MODELS=${RANDOMIZE_MODELS} MAX_COMBOS=${MAX_COMBOS} RANDOMIZE_COMBOS=${RANDOMIZE_COMBOS}"
  ffmpeg -version | head -n 1
fi

# Tools & paths sanity
command -v ffmpeg >/dev/null || { echo "[err] ffmpeg not found"; exit 1; }
command -v ffprobe >/dev/null || { echo "[err] ffprobe not found"; exit 1; }
[[ -f /app/pipeline.py ]] || { echo "[err] /app/pipeline.py missing"; exit 1; }
mkdir -p "$OUT_DIR" "$TMP"

# --- Models (transformer checkpoints) ---
declare -A MODEL_PATHS=(
  # PyTorch Transformer (.pth)
  [candy]="/app/models/pytorch/candy.pth"
  [mosaic]="/app/models/pytorch/mosaic.pth"
  [udnie]="/app/models/pytorch/udnie.pth"
  [rain_princess]="/app/models/pytorch/rain_princess.pth"

  # Torch7 (.t7)
  [candy_t7]="/app/models/torch/candy.t7"
  [mosaic_t7]="/app/models/torch/mosaic.t7"
  [composition_vii]="/app/models/torch/composition_vii_eccv16.t7"
  [la_muse]="/app/models/torch/la_muse_eccv16.t7"
  [starry_night]="/app/models/torch/starry_night_eccv16.t7"
  [the_scream]="/app/models/torch/the_scream.t7"
  [the_wave]="/app/models/torch/the_wave_eccv16.t7"
)

MODEL_ORDER=(candy mosaic udnie rain_princess composition_vii la_muse starry_night the_scream the_wave candy_t7 mosaic_t7)

# Optionally append Magenta styles (comma-separated file names or paths) to MODEL_ORDER
# Example: MAGENTA_STYLES="hokusai.jpg,van_gogh.jpg" and optionally MAGENTA_STYLE_DIR=/work/styles
if [[ -n "${MAGENTA_STYLES:-}" ]]; then
  IFS=',' read -r -a _MAG_STYLES <<< "$MAGENTA_STYLES"
  for _s in "${_MAG_STYLES[@]}"; do
    # Keep raw tag "magenta:<style>" -- we'll resolve path later in style_image()
    MODEL_ORDER+=("magenta:${_s}")
  done
  unset _MAG_STYLES
fi

# If MAGENTA_STYLES is not set, auto-discover styles in MAGENTA_STYLE_DIR
if [[ -z "${MAGENTA_STYLES:-}" && -d "${MAGENTA_STYLE_DIR}" ]]; then
  mapfile -t _MAG_FOUND < <(find "${MAGENTA_STYLE_DIR}" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.jpe' \) \
    -printf '%f\n' | sort)
  if (( ${#_MAG_FOUND[@]} > 0 )); then
    echo "[magenta] auto-discovered ${#_MAG_FOUND[@]} style image(s) in ${MAGENTA_STYLE_DIR}"
    for _f in "${_MAG_FOUND[@]}"; do
      MODEL_ORDER+=("magenta:${_f}")
    done
  else
    echo "[magenta] no style images found in ${MAGENTA_STYLE_DIR}"
  fi
  unset _MAG_FOUND
fi

# Helper: does MODEL_PATHS contain this key? (safe with set -u)
is_known_model() {
  local key="$1"
  [[ -v "MODEL_PATHS[$key]" ]]
}

# Helper: does a model path point to a Torch7 file?
is_t7_path() {
  local p="$1"
  shopt -s nocasematch
  if [[ "$p" == *.t7 ]]; then
    shopt -u nocasematch
    return 0
  fi
  shopt -u nocasematch
  return 1
}

# Verify all model files exist (soft check: warn, don't exit)
for key in "${MODEL_ORDER[@]}"; do
  # Skip path checks for magenta:* pseudo-models (they use style images instead)
  if [[ "$key" == magenta:* ]]; then
    continue
  fi
  m="${MODEL_PATHS[$key]:-}"
  if [[ -z "$m" ]]; then
    echo "[warn] no path configured for model key: $key"; continue
  fi
  if [[ ! -f "$m" ]]; then
    echo "[warn] missing model: $key ($m) — will skip this style"
  fi
done

# --- Discover input images ---
if [[ ! -d "$IN_DIR" ]]; then
  echo "[err] input dir not found: $IN_DIR"; exit 1
fi
mapfile -d '' -t IMAGES < <(find "$IN_DIR" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.heic' -o -iname '*.heif' \) \
  -print0 | sort -z)
if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "[err] no images found in $IN_DIR"; exit 1
fi
echo "[info] will process ${#IMAGES[@]} image(s) from $IN_DIR"

# Helper: precise video duration in seconds (float)
get_dur() {
  local f="$1" d
  d=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=nw=1:nk=1 "$f" || true)
  if [[ -z "$d" || "$d" == "N/A" ]]; then
    d=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$f" || true)
  fi
  printf '%s' "${d:-0}"
}

# Style one still using the reliable 1-frame video path
style_image() {
  local src_path="$1" tag="$2" model_key="$3" model_path="$4" work="$5"
  local out_img="$work/stills/${tag}.jpg"
  local base
  base=$(basename "$src_path"); base="${base%.*}"

  if [[ "$model_key" == "orig" ]]; then
    cp -f "$src_path" "$out_img"
    cp -f "$out_img" "$OUT_DIR/${base}_orig.jpg"
    echo "[still] $base:$tag (orig -> $OUT_DIR/${base}_orig.jpg)"
    return 0
  fi

  # Magenta TF-Hub tiling backend (tag format: magenta:<style_filename_or_path>)
  if [[ "$tag" == magenta:* ]]; then
    local style_file="${tag#magenta:}"
    local style_path="$style_file"
    # Prefer MAGENTA_STYLE_DIR (defaults to /work/magenta_styles), then fall back to the given path
    if [[ -n "${MAGENTA_STYLE_DIR:-}" && -f "${MAGENTA_STYLE_DIR}/$style_file" ]]; then
      style_path="${MAGENTA_STYLE_DIR}/$style_file"
    fi
    if [[ ! -f "$style_path" ]]; then
      echo "[magenta][ERROR] style image not found: $style_file (resolved: $style_path)" >&2
      return 1
    fi

    # Build a safe tag and output filename (no colon, no double extension)
    local style_basename
    style_basename="$(basename "$style_file")"
    local style_name="${style_basename%.*}"
    local safe_tag="magenta_${style_name}"
    out_img="$work/stills/${safe_tag}.jpg"

    # Run the magenta backend through pipeline.py (failure-tolerant)
    set +e
    python -u /app/pipeline.py \
      --model_type magenta \
      --magenta_style "$style_path" \
      --magenta_model_root "${MAGENTA_MODEL_ROOT:-/app/models/magenta}" \
      --magenta_tile "${MAGENTA_TILE:-256}" \
      --magenta_overlap "${MAGENTA_OVERLAP:-32}" \
      --magenta_target_res "$MAGENTA_TARGET_RES" \
      --input_image "$src_path" \
      --output_image "$out_img" \
      --image_ext jpg --jpeg_quality "${IMG_Q:-90}" \
      --threads "${THREADS:-4}" \
      --device cpu
    mag_status=$?
    set -e
    if [[ $mag_status -ne 0 || ! -f "$out_img" ]]; then
      echo "[magenta][WARN] failed (status=$mag_status) for style '$style_file' on '$base'; skipping this style"
      return 1
    fi

    # Provide a colon-keyed alias so later checks for "$work/stills/${tag}.jpg" succeed
    ln -sf "$(basename "$out_img")" "$work/stills/${tag}.jpg" 2>/dev/null || cp -f "$out_img" "$work/stills/${tag}.jpg"

    # Export to OUT_DIR with a safe name
    local out_name="${base}_${safe_tag}.jpg"
    cp -f "$out_img" "$OUT_DIR/$out_name"
    echo "[still] $base:$tag (magenta) -> $OUT_DIR/$out_name"
    return 0
  fi

  if [[ -n "$model_path" ]] && is_t7_path "$model_path"; then
    # Run Torch7 via OpenCV DNN with Caffe/BGR preprocessing (subtract means; add back on output).
    # If it crashes or fails, skip this style without aborting the script.
    set +e
    python - "$src_path" "$model_path" "$out_img" "$IMG_Q" <<'PY'
import os, sys, cv2, numpy as np
src_path, model_path, out_path, jpeg_q = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

# Read input (BGR)
img = cv2.imread(src_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Content image not found: {src_path}")

# Optional downscale for stability / memory. Use env SCALE as max height (default 720).
try:
    max_h = int(os.getenv("SCALE", "720"))
    if max_h <= 0: max_h = 720
except Exception:
    max_h = 720
h, w = img.shape[:2]
if h > max_h:
    new_w = int(round(w * (max_h / float(h))))
    if new_w < 16: new_w = 16
    img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

# Caffe-style preprocessing: BGR input, subtract ImageNet means (no scaling)
MEAN_BGR = (103.939, 116.779, 123.680)
blob = cv2.dnn.blobFromImage(
    image=img,
    scalefactor=1.0,
    size=(w, h),
    mean=MEAN_BGR,
    swapRB=False,   # keep BGR for Caffe-trained torch7 nets
    crop=False
)

net = cv2.dnn.readNetFromTorch(model_path)
if hasattr(cv2.dnn, 'DNN_BACKEND_OPENCV'):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
if hasattr(cv2.dnn, 'DNN_TARGET_CPU'):
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

net.setInput(blob)
out = net.forward()  # (1, 3, H, W) in BGR domain, mean-subtracted

if out is None or out.size == 0:
    raise RuntimeError("DNN forward returned empty output")

# Convert to HWC BGR and ADD MEAN BACK (reverse of input mean-subtraction)
out = out.squeeze().transpose(1, 2, 0)  # HWC (BGR)
out = out + np.array(MEAN_BGR, dtype=np.float32)[None, None, :]

# Clamp to displayable range and uint8
out = np.clip(out, 0, 255).astype(np.uint8)

# Save (keep BGR for OpenCV imwrite)
ok = cv2.imwrite(out_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
if not ok:
    raise RuntimeError("Failed to write output image")
PY
    py_status=$?
    set -e
    if [[ $py_status -ne 0 || ! -f "$out_img" ]]; then
      echo "[warn] Torch7 style failed for $base:$tag (status=$py_status); skipping this style"
      return 1
    fi
    cp -f "$out_img" "$OUT_DIR/${base}_${tag}.jpg"
    echo "[still] $base:$tag (.t7) -> $OUT_DIR/${base}_${tag}.jpg"
    return 0
  fi

  # Use a per-tag oneshot dir to avoid cross-talk.
  local one="$TMP/oneshot_${base}_${tag}"
  rm -rf "$one"; mkdir -p "$one/frames"

  # Create a very short MP4 by looping the still for 0.04s at $FPS.
  # Keep it simple to avoid zero-frame edge cases.
  if ! ffmpeg -y -loglevel error -stats \
      -loop 1 -t 0.04 -r "$FPS" -i "$src_path" \
      -vf "scale=${SCALE}:-2:flags=bicubic,format=yuv420p" \
      -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
      "$one/in_1f.mp4"; then
    echo "[warn] primary still->mp4 encode failed; retrying with fallback params"
    ffmpeg -y -loglevel warning \
      -loop 1 -t 0.04 -r "$FPS" -i "$src_path" \
      -vf "scale=${SCALE}:-2:flags=bicubic,format=yuv420p" \
      -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
      "$one/in_1f.mp4"
  fi

  [[ -s "$one/in_1f.mp4" ]] || { echo "[err] failed to create $one/in_1f.mp4 (no frames)"; ls -l "$one" || true; exit 1; }

  set +e
  python -u /app/pipeline.py \
    --model_type transformer \
    --model "$model_path" \
    --input_video "$one/in_1f.mp4" \
    --output_video "$one/out_1f.mp4" \
    --work_dir "$one" \
    --io_preset imagenet_255 \
    --image_ext jpg --jpeg_quality "$IMG_Q" \
    --max_frames 1 \
    --fps "$FPS" --scale "$SCALE"
  tr_status=$?
  set -e
  if [[ $tr_status -ne 0 ]]; then
    echo "[warn] transformer backend failed (status=$tr_status) for $base:$tag; skipping this style"
    return 1
  fi

  if [[ -f "$one/frames/styled_frame_0001.jpg" ]]; then
    cp -f "$one/frames/styled_frame_0001.jpg" "$out_img"
  elif [[ -f "$one/frames/styled_frame_001.jpg" ]]; then
    cp -f "$one/frames/styled_frame_001.jpg" "$out_img"
  else
    echo "[err] styled frame not found for $base:$tag"; exit 1
  fi

  cp -f "$out_img" "$OUT_DIR/${base}_${tag}.jpg"
  echo "[still] $base:$tag -> $OUT_DIR/${base}_${tag}.jpg"
}

#
# Build morph video for a single source image through all models
build_morph_for_image() {
  local src_img="$1" base
  base=$(basename "$src_img")
  base="${base%.*}"
  local work="$TMP/$base"
  rm -rf "$work"; mkdir -p "$work"/stills "$work"/clips

  # Select up to MAX_MODELS styles for this image, randomized with a per-image seed for reproducibility
  local SELECTED_MODELS=()
  if (( RANDOMIZE_MODELS )) && command -v shuf >/dev/null; then
    # Seed shuf in a reproducible way using the base name's checksum
    local seed
    seed=$(printf '%s' "${RANDOM_SEED:-$base}" | cksum | awk '{print $1}')
    mapfile -t SELECTED_MODELS < <(printf '%s\n' "${MODEL_ORDER[@]}" | shuf --random-source=<(yes "$seed") | head -n "$MAX_MODELS")
  else
    mapfile -t SELECTED_MODELS < <(printf '%s\n' "${MODEL_ORDER[@]}" | head -n "$MAX_MODELS")
  fi
  # Guard rails: if selection somehow comes back empty, fall back to the first MAX_MODELS items
  if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    SELECTED_MODELS=("${MODEL_ORDER[@]:0:$MAX_MODELS}")
  fi
  # If more than MAX_MODELS slipped in somehow, trim
  if [[ ${#SELECTED_MODELS[@]} -gt $MAX_MODELS ]]; then
    SELECTED_MODELS=("${SELECTED_MODELS[@]:0:$MAX_MODELS}")
  fi
  echo "[select] $base picked ${#SELECTED_MODELS[@]} style(s) out of ${#MODEL_ORDER[@]}: ${SELECTED_MODELS[*]}"

  # First build baseline stills only: original + each single model
  local BASE_SEQ=(orig "${SELECTED_MODELS[@]}")
  echo "[plan] $base baseline sequence (capped/randomized): ${BASE_SEQ[*]}"

  for tag in "${BASE_SEQ[@]}"; do
    echo "[stage] $base -> still:$tag"
    case "$tag" in
      orig)
        style_image "$src_img" "$tag" "orig" "" "$work"
        ;;
      magenta:*)
        style_image "$src_img" "$tag" "magenta" "" "$work"
        ;;
      candy|mosaic|udnie|rain_princess|composition_vii|la_muse|starry_night|the_scream|the_wave|candy_t7|mosaic_t7)
        model_path="${MODEL_PATHS[$tag]}"
        echo "[debug] $base using model '$tag' at: $model_path"
        style_image "$src_img" "$tag" "$tag" "$model_path" "$work"
        ;;
      *)
        echo "[warn] unknown model tag in baseline: '$tag'"; continue
        ;;
    esac
    echo "[stage][done] $base:$tag (attempted)"
    if [[ ! -f "$work/stills/${tag}.jpg" ]]; then
      echo "[warn] missing work still after baseline build (will exclude from blends): $work/stills/${tag}.jpg"
      continue
    fi
  done

  # Derive AVAILABLE models from actual stills on disk (guards against any earlier failure)
  local AVAILABLE=()
  for m in "${SELECTED_MODELS[@]}"; do
    if [[ -f "$work/stills/${m}.jpg" ]]; then
      AVAILABLE+=("$m")
    else
      echo "[warn] $base: model still missing, excluding from blends: $m ($work/stills/${m}.jpg)"
    fi
  done
  echo "[debug] $base available models for blending: ${AVAILABLE[*]}"

  # Build all unordered 2-model combinations (pairs) from AVAILABLE.
  # We keep an internal list with a safe delimiter ("::") to avoid issues with model keys that contain underscores.
  # Separately, we keep the display/filename tags that use underscores (e.g., candy_mosaic) for files and SEQ.
  local PAIR_INTERNAL=()
  local PAIR_STAGE_TAGS=()
  if [[ ${#AVAILABLE[@]} -ge 2 ]]; then
    for ((i=0; i<${#AVAILABLE[@]}-1; i++)); do
      for ((j=i+1; j<${#AVAILABLE[@]}; j++)); do
        # Skip near-duplicate pairs (e.g., candy vs candy_t7, mosaic vs mosaic_t7)
        if { [[ "${AVAILABLE[$i]}" == candy && "${AVAILABLE[$j]}" == candy_t7 ]] || [[ "${AVAILABLE[$i]}" == candy_t7 && "${AVAILABLE[$j]}" == candy ]]; } || \
           { [[ "${AVAILABLE[$i]}" == mosaic && "${AVAILABLE[$j]}" == mosaic_t7 ]] || [[ "${AVAILABLE[$i]}" == mosaic_t7 && "${AVAILABLE[$j]}" == mosaic ]]; }; then
          continue
        fi
        PAIR_INTERNAL+=("${AVAILABLE[$i]}::${AVAILABLE[$j]}")
        PAIR_STAGE_TAGS+=("${AVAILABLE[$i]}_${AVAILABLE[$j]}")
      done
    done
  fi
  # If requested, cap the number of combos to MAX_COMBOS (0 = no cap).
  local mc
  mc=${MAX_COMBOS:-0}
  if (( mc > 0 && ${#PAIR_INTERNAL[@]} > mc )); then
    # Prepare combined lines "internal|stage" so we can shuffle/slice while keeping them aligned.
    if command -v paste >/dev/null; then
      # Build a list of "internal|stage" lines
      mapfile -t _PAIR_LINES < <(paste -d '|' <(printf '%s\n' "${PAIR_INTERNAL[@]}") <(printf '%s\n' "${PAIR_STAGE_TAGS[@]}"))
    else
      # Fallback without paste
      _PAIR_LINES=()
      for ((pi=0; pi<${#PAIR_INTERNAL[@]}; pi++)); do
        _PAIR_LINES+=("${PAIR_INTERNAL[$pi]}|${PAIR_STAGE_TAGS[$pi]}")
      done
    fi
    # Choose a deterministic seed based on image base (or global RANDOM_SEED if set)
    local seed
    seed=$(printf '%s' "${RANDOM_SEED:-$base}" | cksum | awk '{print $1}')
    local -a _SEL=()
    if (( RANDOMIZE_COMBOS )) && command -v shuf >/dev/null; then
      mapfile -t _SEL < <(printf '%s\n' "${_PAIR_LINES[@]}" | shuf --random-source=<(yes "$seed") | head -n "${mc}")
    else
      mapfile -t _SEL < <(printf '%s\n' "${_PAIR_LINES[@]}" | head -n "${mc}")
    fi
    # Rebuild capped arrays from selection
    PAIR_INTERNAL=()
    PAIR_STAGE_TAGS=()
    for line in "${_SEL[@]}"; do
      local left="${line%%|*}"
      local right="${line#*|}"
      PAIR_INTERNAL+=("$left")
      PAIR_STAGE_TAGS+=("$right")
    done
    echo "[debug] $base capped combos to ${#PAIR_INTERNAL[@]} via MAX_COMBOS=${mc} (from total ${#_PAIR_LINES[@]})"
    unset _PAIR_LINES _SEL
  fi
  echo "[debug] $base built ${#PAIR_INTERNAL[@]} pair(s)"

  # Sequence: original → each model → every 2-model blend → (optional allfour) → original
  local SEQ=()
  if [[ "${INCLUDE_ALLFOUR:-0}" -eq 1 ]]; then
    SEQ=("orig" "${AVAILABLE[@]}" "${PAIR_STAGE_TAGS[@]}" "allfour" "orig")
  else
    SEQ=("orig" "${AVAILABLE[@]}" "${PAIR_STAGE_TAGS[@]}" "orig")
  fi
  echo "[plan] $base sequence: ${SEQ[*]}"

  # 1) Generate stills for each stage (skip allfour and pairs here; we build them after the models exist)
  for tag in "${SEQ[@]}"; do
    echo "[stage] $base -> still:$tag"
    case "$tag" in
      orig)
        style_image "$src_img" "$tag" "orig" "" "$work"
        ;;
      allfour)
        # Defer building the allfour composite until after the four model stills exist
        continue
        ;;
      *_*)
        # Defer pair blends (e.g., candy_mosaic) until after individual model stills exist
        continue
        ;;
      magenta:*)
        style_image "$src_img" "$tag" "magenta" "" "$work"
        ;;
      candy|mosaic|udnie|rain_princess|composition_vii|la_muse|starry_night|the_scream|the_wave|candy_t7|mosaic_t7)
        model_path="${MODEL_PATHS[$tag]}"
        echo "[debug] $base using model '$tag' at: $model_path"
        style_image "$src_img" "$tag" "$tag" "$model_path" "$work"
        ;;
      *)
        echo "[warn] unknown model tag: '$tag' (no MODEL_PATHS entry); skipping"
        continue
        ;;
    esac
    # verify per-tag still exists under work and got copied to OUT_DIR
    if [[ "$tag" != "allfour" ]]; then
      if [[ ! -f "$work/stills/${tag}.jpg" ]]; then
        echo "[warn] missing work still: $work/stills/${tag}.jpg (skipping this tag)"
        continue
      fi
      if [[ ! -f "$OUT_DIR/${base}_${tag}.jpg" ]]; then
        echo "[warn] expected OUT still not found yet: $OUT_DIR/${base}_${tag}.jpg"
      fi
    fi
  done

  # Generate all 2-model blended stills (true 50/50 average for each unordered pair)
  local PAIR_BUILT=()
  if [[ ${#PAIR_INTERNAL[@]} -gt 0 ]]; then
    for pair in "${PAIR_INTERNAL[@]}"; do
      # Robustly split the custom "::" delimiter without using IFS (which treats ":" as per-char)
      a="${pair%%::*}"      # left of first "::"
      b="${pair#*::}"       # right of first "::"
      # Guard against malformed entries (allow ':' inside a/b; delimiter is "::")
      if [[ -z "$a" || -z "$b" ]]; then
        echo "[warn] malformed pair tag: '$pair' (a='$a' b='$b')"; continue
      fi
      pair_tag="${a}_${b}"
      # Sanitize for filesystem safety (convert ':' and '/' to '-')
      pair_tag_safe="${pair_tag//[:\/]/-}"

      # Auto-rebuild missing dependency stills if possible
      if [[ ! -f "$work/stills/${a}.jpg" ]] && is_known_model "$a"; then
        echo "[rebuild] $base missing still for '$a' — regenerating"
        style_image "$src_img" "$a" "$a" "${MODEL_PATHS[$a]}" "$work"
      fi
      if [[ ! -f "$work/stills/${b}.jpg" ]] && is_known_model "$b"; then
        echo "[rebuild] $base missing still for '$b' — regenerating"
        style_image "$src_img" "$b" "$b" "${MODEL_PATHS[$b]}" "$work"
      fi

      if [[ ! -f "$work/stills/${a}.jpg" || ! -f "$work/stills/${b}.jpg" ]]; then
        echo "[warn] $base missing dependency for pair '$pair_tag': $work/stills/${a}.jpg or $work/stills/${b}.jpg; skipping"
        continue
      fi

      # Build the blended still if missing; otherwise reuse it.
      if [[ ! -f "$work/stills/${pair_tag_safe}.jpg" ]]; then
        echo "[blend] $base building pair '$pair_tag_safe' from: $a + $b (50% each)"
        # Build JPG directly using Pillow to avoid ffmpeg image2 quirks
        python - "$work/stills/${a}.jpg" "$work/stills/${b}.jpg" "$work/stills/${pair_tag_safe}.jpg" "$IMG_Q" <<'PY'
import sys
from PIL import Image

src_a, src_b, out_jpg, q = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

# Load and convert to RGB
A = Image.open(src_a).convert('RGB')
B = Image.open(src_b).convert('RGB')

# Match sizes (scale B to A using bicubic like scale2ref)
if B.size != A.size:
    B = B.resize(A.size, Image.BICUBIC)

# 50/50 blend
out = Image.blend(A, B, 0.5)

# Ensure even dimensions for downstream encoders
w, h = out.size
w2, h2 = w - (w % 2), h - (h % 2)
if (w2, h2) != (w, h):
    out = out.crop((0, 0, w2, h2))

# Save as high-quality JPEG
out.save(out_jpg, format='JPEG', quality=q, subsampling=0, optimize=True)
PY

        # Verify JPG exists
        if [[ ! -f "$work/stills/${pair_tag_safe}.jpg" ]]; then
          echo "[warn] $base failed to build pair still: $work/stills/${pair_tag_safe}.jpg; skipping"
          continue
        fi
      fi

      cp -f "$work/stills/${pair_tag_safe}.jpg" "$OUT_DIR/${base}_${pair_tag_safe}.jpg"
      echo "[still] $base:${pair_tag_safe} -> $OUT_DIR/${base}_${pair_tag_safe}.jpg"
      PAIR_BUILT+=("$pair_tag_safe")
    done
  fi

  # Generate allfour blended still (true 25% average of each model)
  if [[ "${INCLUDE_ALLFOUR:-0}" -eq 1 ]]; then
    if [[ ! -f "$work/stills/allfour.jpg" ]]; then
      for dep in candy mosaic udnie rain_princess; do
        if [[ ! -f "$work/stills/${dep}.jpg" ]]; then
          if is_known_model "$dep"; then
            echo "[rebuild] $base missing still for '$dep' — regenerating for allfour"
            style_image "$src_img" "$dep" "$dep" "${MODEL_PATHS[$dep]}" "$work"
          fi
          if [[ ! -f "$work/stills/${dep}.jpg" ]]; then
            echo "[err] $base missing dependency for allfour: $work/stills/${dep}.jpg"; exit 1
          fi
        fi
      done
      echo "[blend] $base building allfour from: candy + mosaic + udnie + rain_princess (25% each)"
      ffmpeg -y -hide_banner -loglevel error \
        -i "$work/stills/candy.jpg" \
        -i "$work/stills/mosaic.jpg" \
        -i "$work/stills/udnie.jpg" \
        -i "$work/stills/rain_princess.jpg" \
        -filter_complex "\
          [0:v][1:v]blend=all_expr='(A+B)/2'[avg2];\
          [avg2][2:v]blend=all_expr='(A*2+B)/3'[avg3];\
          [avg3][3:v]blend=all_expr='(A*3+B)/4',format=rgb24[out]" \
        -map "[out]" -frames:v 1 -update 1 -pix_fmt yuvj420p -f image2 "$work/stills/allfour.jpg"
      cp -f "$work/stills/allfour.jpg" "$OUT_DIR/${base}_allfour.jpg"
      echo "[still] $base:allfour -> $OUT_DIR/${base}_allfour.jpg"
    fi
    if [[ ! -f "$work/stills/allfour.jpg" ]]; then
      echo "[err] $base missing still for tag: allfour ($work/stills/allfour.jpg)"; exit 1
    fi
  fi

  # Build the final SEQ now that blends exist .
  local SEQ=()
  if [[ "${INCLUDE_ALLFOUR:-0}" -eq 1 && -f "$work/stills/allfour.jpg" ]]; then
    SEQ=(orig "${AVAILABLE[@]}" "${PAIR_BUILT[@]}" allfour orig)
  else
    SEQ=(orig "${AVAILABLE[@]}" "${PAIR_BUILT[@]}" orig)
  fi
  echo "[plan] $base final sequence: ${SEQ[*]}"

  # Verify all stills exist for this image .
  for tag in "${SEQ[@]}"; do
    if [[ ! -f "$work/stills/${tag}.jpg" ]]; then
      if [[ -f "$work/stills/${tag}.png" ]]; then
        echo "[fixup] $base found PNG for '$tag' — converting to JPG"
        if ! ffmpeg -y -hide_banner -loglevel error -i "$work/stills/${tag}.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -frames:v 1 -q:v "$IMG_Q" "$work/stills/${tag}.jpg"; then
          echo "[fixup] $base ffmpeg failed for PNG->JPG; trying Pillow"
          python - "$work/stills/${tag}.png" "$work/stills/${tag}.jpg" "$IMG_Q" <<'PY'
import sys
from PIL import Image
png, jpg, q = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(png).convert("RGB")
im.save(jpg, format="JPEG", quality=q, subsampling=0, optimize=True)
PY
        fi
        rm -f "$work/stills/${tag}.png" || true
      fi
    fi
    if [[ ! -f "$work/stills/${tag}.jpg" ]]; then
      echo "[warn] $base missing still for tag: $tag ($work/stills/${tag}.jpg); excluding from sequence"
      continue
    fi
  done

  # 2) Turn each still into HOLD+TRANS clips .
  local i=0
  declare -a HOLD_PER=()
  local -a CLIP_LIST=()
  local n=${#SEQ[@]}
  for tag in "${SEQ[@]}"; do
    i=$((i+1))
    # originals: first uses HOLD_ORIG_START, last uses HOLD_ORIG_END; models use HOLD_MODEL
    if [[ "$tag" == "orig" && $i -eq 1 ]]; then
      HOLD_THIS=$HOLD_ORIG_START
    elif [[ "$tag" == "orig" && $i -eq $n ]]; then
      HOLD_THIS=$HOLD_ORIG_END
    else
      HOLD_THIS=$HOLD_MODEL
    fi
    HOLD_PER[$i]=$HOLD_THIS
    if [[ ! -f "$work/stills/${tag}.jpg" ]]; then
      echo "[skip] $base:$tag no still on disk; skipping clip"
      continue
    fi
    # Scale to fixed height, keep aspect (width=-2), then format to yuv420p
    ffmpeg -y -loglevel error -stats \
      -loop 1 -t "$(awk -v h="$HOLD_THIS" -v t="$TRANS" 'BEGIN{ printf("%.6f", h+t) }')" -r "$FPS" -i "$work/stills/${tag}.jpg" \
      -vf "scale=${SCALE}:-2:flags=bicubic,format=yuv420p" \
      -c:v libx264 -pix_fmt yuv420p -crf "$CRF" -preset "$PRESET" "$work/clips/$i.mp4"
    echo "[clip] $base:$i ($tag) hold=${HOLD_THIS}s"
    CLIP_LIST+=("$work/clips/$i.mp4")
  done

  local n=${#CLIP_LIST[@]}
  if [[ $n -eq 0 ]]; then
    echo "[err] no clips were generated for $base"; exit 1
  fi
  echo "[chain] building morph for $base using $n clip(s)"
  echo "[info] TRANS=${TRANS}s | HOLD_ORIG_START=${HOLD_ORIG_START}s | HOLD_MODEL=${HOLD_MODEL}s | HOLD_ORIG_END=${HOLD_ORIG_END}s | transition=${TRANSITION}"

  local ACCUM="$work/accum.mp4"
  cp -f "${CLIP_LIST[0]}" "$ACCUM"

  for ((k=1; k<n; k++)); do
    local NEXT="${CLIP_LIST[$k]}"

    # Duration of the accumulated timeline so far (seconds, float)
    local ACCUM_S OFFSET_S FADE_DUR
    ACCUM_S=$(get_dur "$ACCUM")

    # Always use the requested TRANS as the fade length (no clamping to hold).
    FADE_DUR="$TRANS"

    # Start the fade FADE_DUR seconds before the end of the current timeline.
    OFFSET_S=$(awk -v a="$ACCUM_S" -v f="$FADE_DUR" 'BEGIN{ off=a-f; if(off<0) off=0; printf("%.6f", off) }')

    echo "[calc][$base] accum=${ACCUM_S}s fade=${FADE_DUR}s offset=${OFFSET_S}s"
    echo "[xfade][$base] k=$((k+1))  trans=${TRANSITION}  dur=${FADE_DUR}s  offset=${OFFSET_S}s"

    # Pad both inputs to guarantee the full crossfade window exists (prevents truncation when FADE_DUR > hold).
    ffmpeg -y -loglevel error -stats \
      -i "$ACCUM" -i "$NEXT" \
      -filter_complex "\
        [0:v]tpad=stop_mode=clone:stop_duration=${FADE_DUR}[a]; \
        [1:v]tpad=stop_mode=clone:stop_duration=${FADE_DUR}[b]; \
        [a][b]xfade=transition=${TRANSITION}:duration=${FADE_DUR}:offset=${OFFSET_S},format=yuv420p[v]" \
      -map "[v]" -r "$FPS" -pix_fmt yuv420p -colorspace bt709 -crf "$CRF" -preset "$PRESET" "$work/accum_next.mp4"

    mv -f "$work/accum_next.mp4" "$ACCUM"
  done

  if [[ ! -f "$ACCUM" ]]; then
    echo "[err] no accumulated video produced for $base"; exit 1
  fi

  local out_vid="$OUT_DIR/${base}_morph.mp4"
  mv -f "$ACCUM" "$out_vid"
  local dur
  dur=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$out_vid" || echo "unknown")
  printf '[done] %s (≈ %.3fs)\n' "$out_vid" "$dur"
  ls -lh "$OUT_DIR"/"${base}"_* | sed "s/^/[out][$base] /"
}

# --- Main: process every image ---
rm -rf "$TMP"; mkdir -p "$TMP"
for src in "${IMAGES[@]}"; do
  echo "===== [morph] $(basename "$src") ====="
  if [[ "${INCLUDE_ALLFOUR:-0}" -eq 1 ]]; then
    echo "[apply][REV=${REV}] pool: ${#MODEL_ORDER[@]} styles → up to ${MAX_MODELS} randomized per image (allfour enabled)"
  else
    echo "[apply][REV=${REV}] pool: ${#MODEL_ORDER[@]} styles → up to ${MAX_MODELS} randomized per image"
  fi
  build_morph_for_image "$src"
done

#     Summary
ls -lh "$OUT_DIR" | sed 's/^/[out] /'