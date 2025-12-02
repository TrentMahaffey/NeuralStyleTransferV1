#!/usr/bin/env bash
set -euo pipefail

# ---- Helpers ---------------------------------------------------------------
log() { printf "%s\n" "$*" ; }
die() { printf "[error] %s\n" "$*" >&2; exit 1; }

_resolve_weights() {
  # Accept both .pth and .pth.tar (either order)
  local p="$1"
  [[ -f "$p" ]] && { printf "%s" "$p"; return; }
  if [[ "$p" == *.tar ]]; then
    local alt="${p%.tar}"
  else
    local alt="${p}.tar"
  fi
  [[ -f "$alt" ]] && { printf "%s" "$alt"; return; }
  die "weights checkpoint not found: $p (or $alt)"
}

_has_variant() {
  local needle=",$1," ; local hay=",$2,"
  [[ "$hay" == *"$needle"* ]]
}

# ---- Config (env-driven with sane defaults) --------------------------------
INPUT_VIDEO=${INPUT_VIDEO:-/app/input_videos/input.mp4}
OUTPUT_VIDEO=${OUTPUT_VIDEO:-} # optional; per-variant names if empty
STYLE_MODEL=${STYLE_MODEL:-/app/models/pytorch/mosaic.pth}
STYLE_MODEL_TYPE=${STYLE_MODEL_TYPE:-transformer}
IO_PRESET=${IO_PRESET:-auto}
DEVICE=${DEVICE:-cpu}

# --- Per-label styling (optional) -------------------------------------------
STYLE_BY_LABEL=${STYLE_BY_LABEL:-0}  # 1 = enable multi-model per label
# Which semantic labels belong to each group
PERSON_LABELS=${PERSON_LABELS:-person}
VEHICLE_LABELS=${VEHICLE_LABELS:-bicycle,motorbike}
# Models for each group (paths). BASE_MODEL can be empty to reuse STYLE_MODEL.
BASE_MODEL=${BASE_MODEL:-}
BASE_MODEL_TYPE=${BASE_MODEL_TYPE:-transformer}
PERSON_MODEL=${PERSON_MODEL:-}
PERSON_MODEL_TYPE=${PERSON_MODEL_TYPE:-transformer}
VEHICLE_MODEL=${VEHICLE_MODEL:-}
VEHICLE_MODEL_TYPE=${VEHICLE_MODEL_TYPE:-transformer}
# If 1, skip BASE stylizing pass and use original frames as the base (keeps background unstyled)
BASE_AS_ORIGINAL=${BASE_AS_ORIGINAL:-0}
# Mask polarity (1=invert like our fg flows; 0=use masks as-is)
PERSON_MASK_INVERT=${PERSON_MASK_INVERT:-1}
VEHICLE_MASK_INVERT=${VEHICLE_MASK_INVERT:-1}

# Canvas / inference
FPS=${FPS:-24}
CANVAS_W=${CANVAS_W:-1920}
CANVAS_H=${CANVAS_H:-1080}
SCALE=${SCALE:-$(( CANVAS_W<CANVAS_H ? CANVAS_W : CANVAS_H ))}
INFER_RES=${INFER_RES:-1280}
FRAME_EXT=${FRAME_EXT:-png}
# Respect explicit user override of FILL_FRAME; default to 1 otherwise
if [[ -z "${FILL_FRAME+x}" ]]; then
  FILL_FRAME=1   # 1=fill+center-crop (no pillarbox), 0=fit+pad (may add bars)
  FILL_FRAME_USER_SET=0
else
  FILL_FRAME_USER_SET=1
fi

# Optional single env CANVAS=WxH and auto-orientation from input
AUTO_CANVAS=${AUTO_CANVAS:-1}
if [[ -n "${CANVAS:-}" ]]; then
  if [[ "$CANVAS" =~ ^([0-9]+)x([0-9]+)$ ]]; then
    CANVAS_W="${BASH_REMATCH[1]}"
    CANVAS_H="${BASH_REMATCH[2]}"
  else
    die "CANVAS must be formatted as WxH (e.g. 1920x1080). Got: $CANVAS"
  fi
fi

# Work root + folders (can be overridden)
IN_BASENAME="$(basename -- "$INPUT_VIDEO")"
IN_STEM="${IN_BASENAME%.*}"
log "[input] INPUT_VIDEO=$INPUT_VIDEO  AUTO_CANVAS=${AUTO_CANVAS}"

# If enabled, automatically flip canvas to match input video orientation.
# We only change orientation; absolute size stays as requested (defaults 1920x1080).
if [[ "${AUTO_CANVAS}" == "1" ]]; then
  if command -v ffprobe >/dev/null 2>&1; then
    # Robust width/height probe that works across ffprobe builds
    _PROBE_OUT=""
    _PROBE_RC=0
    set +e
    # Try simple CSV first (e.g., "1920,1080")
    _PROBE_OUT=$(ffprobe -v error -select_streams v:0 \
                          -show_entries stream=width,height \
                          -of csv=p=0 "$INPUT_VIDEO" 2>/dev/null)
    _PROBE_RC=$?
    # Fallback: two-line output (width\nheight)
    if [[ ${_PROBE_RC} -ne 0 || -z "${_PROBE_OUT}" ]]; then
      _PROBE_OUT=$(ffprobe -v error -select_streams v:0 \
                            -show_entries stream=width,height \
                            -of default=noprint_wrappers=1:nokey=1 "$INPUT_VIDEO" 2>/dev/null)
      _PROBE_RC=$?
      if [[ ${_PROBE_RC} -eq 0 && -n "${_PROBE_OUT}" ]]; then
        # Join lines into CSV-like form
        _PROBE_OUT=$(echo "${_PROBE_OUT}" | tr '\n' ',')
        _PROBE_OUT=${_PROBE_OUT%,}
      fi
    fi
    set -e

    if [[ ${_PROBE_RC} -eq 0 && -n "${_PROBE_OUT}" ]]; then
      # Parse width,height from comma-separated string
      IFS=',' read -r _W _H <<< "${_PROBE_OUT}"
      if [[ -n "${_W}" && -n "${_H}" ]]; then
        if [[ "${_H}" -gt "${_W}" ]]; then
          # Portrait input; ensure portrait canvas
          if [[ "${CANVAS_W}" -gt "${CANVAS_H}" ]]; then
            tmp="${CANVAS_W}"; CANVAS_W="${CANVAS_H}"; CANVAS_H="${tmp}"
            log "[auto-canvas] Portrait input ${_W}x${_H} → flipped canvas to ${CANVAS_W}x${CANVAS_H}"
          else
            log "[auto-canvas] Portrait input ${_W}x${_H} → canvas already portrait ${CANVAS_W}x${CANVAS_H}"
          fi
        else
          # Landscape input; ensure landscape canvas
          if [[ "${CANVAS_H}" -gt "${CANVAS_W}" ]]; then
            tmp="${CANVAS_W}"; CANVAS_W="${CANVAS_H}"; CANVAS_H="${tmp}"
            log "[auto-canvas] Landscape input ${_W}x${_H} → flipped canvas to ${CANVAS_W}x${CANVAS_H}"
          else
            log "[auto-canvas] Landscape input ${_W}x${_H} → canvas already landscape ${CANVAS_W}x${CANVAS_H}"
          fi
        fi
      else
        log "[auto-canvas][warn] could not parse dimensions (got '${_PROBE_OUT}') — keeping canvas ${CANVAS_W}x${CANVAS_H}"
      fi
    else
      log "[auto-canvas][warn] ffprobe returned no dimensions for: $INPUT_VIDEO — leaving canvas ${CANVAS_W}x${CANVAS_H}"
    fi
  else
    log "[auto-canvas][warn] ffprobe not available — leaving canvas ${CANVAS_W}x${CANVAS_H}"
  fi
fi

# Auto-pick fill vs pad policy based on input orientation (user prefers padding for horizontal)
# These can be overridden via env:
#   FILL_FRAME_FOR_LANDSCAPE (default 0 = fit+pad)
#   FILL_FRAME_FOR_PORTRAIT  (default 1 = fill+crop)
FILL_FRAME_FOR_LANDSCAPE=${FILL_FRAME_FOR_LANDSCAPE:-0}
FILL_FRAME_FOR_PORTRAIT=${FILL_FRAME_FOR_PORTRAIT:-1}

if [[ "${AUTO_CANVAS}" == "1" ]]; then
  # If we successfully probed width/height above, use that to pick policy.
  if [[ -n "${_W:-}" && -n "${_H:-}" ]]; then
    if [[ "${FILL_FRAME_USER_SET}" == "1" ]]; then
      log "[auto-canvas] FILL_FRAME explicitly set to ${FILL_FRAME}; keeping user value (0=fit+pad,1=fill+crop)"
    else
      if [[ "${_W}" -gt "${_H}" ]]; then
        # Horizontal input → prefer padding over cropping
        FILL_FRAME="${FILL_FRAME_FOR_LANDSCAPE}"
        log "[auto-canvas] Horizontal input: using FILL_FRAME=${FILL_FRAME} (0=fit+pad,1=fill+crop)"
      else
        # Vertical input → keep fill by default
        FILL_FRAME="${FILL_FRAME_FOR_PORTRAIT}"
        log "[auto-canvas] Vertical input: using FILL_FRAME=${FILL_FRAME} (0=fit+pad,1=fill+crop)"
      fi
    fi
  fi
fi

WORK_ROOT_DEFAULT="/app/_work/${IN_STEM}_cw${CANVAS_W}ch${CANVAS_H}_fps${FPS}"
WORK_ROOT=${WORK_ROOT:-$WORK_ROOT_DEFAULT}
FRAMES_DIR=${FRAMES_DIR:-$WORK_ROOT/frames}
MASKS_DIR=${MASKS_DIR:-$WORK_ROOT/masks}
OUT_DIR=${OUT_DIR:-/app/output}

# Variants and behavior
VARIANTS=${VARIANTS:-fg,bg}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
SKIP_MASKS=${SKIP_MASKS:-0}

# Mask generation
DEEPLAB_WEIGHTS=$(_resolve_weights "${DEEPLAB_WEIGHTS:-/app/models/deeplab/deeplab-resnet.pth.tar}")
BACKBONE=${BACKBONE:-resnet}
MASK_RES=${MASK_RES:-512}
MASK_EXPAND_PCT=${MASK_EXPAND_PCT:-3.0}
MASK_CONTRACT_PCT=${MASK_CONTRACT_PCT:-0.0}
MASK_FEATHER_PCT=${MASK_FEATHER_PCT:-3.0}
#MASK_ALIGN=${MASK_ALIGN:-auto}   # mask placement: center,left,right,top,bottom,auto
MASK_TARGET_LABELS=${MASK_TARGET_LABELS:-}   # e.g. "person,bicycle,motorbike"
MASK_DEBUG_OVERLAY=${MASK_DEBUG_OVERLAY:-0}
MASK_AUTOFIX=${MASK_AUTOFIX:-1}

# Style postproc / temporal (note: image mode ignores flow options)
BLEND=${BLEND:-1.0}
FLOW_EMA=${FLOW_EMA:-0}
FLOW_ALPHA=${FLOW_ALPHA:-0.85}
SMOOTHING=${SMOOTHING:-0}   # --smooth_lightness + --smooth_alpha 0.7

# ---- Prep dirs -------------------------------------------------------------
mkdir -p "$FRAMES_DIR" "$MASKS_DIR" "$OUT_DIR" "$WORK_ROOT" /app/_work/debug

log "[work] WORK_ROOT=$WORK_ROOT"
log "       FRAMES_DIR=$FRAMES_DIR"
log "       MASKS_DIR=$MASKS_DIR"
log "[cfg] CANVAS=${CANVAS_W}x${CANVAS_H} FPS=$FPS INFER_RES=$INFER_RES"
log "[cfg] MASK_RES=$MASK_RES EXPAND=${MASK_EXPAND_PCT}% FEATHER=${MASK_FEATHER_PCT}%"
log "[cfg] MODEL=$STYLE_MODEL TYPE=$STYLE_MODEL_TYPE IO_PRESET=$IO_PRESET VARIANTS=$VARIANTS"
log "[cfg] FILL_FRAME=$FILL_FRAME (1=fill+center-crop, 0=fit+pad) [landscape_pref=${FILL_FRAME_FOR_LANDSCAPE}, portrait_pref=${FILL_FRAME_FOR_PORTRAIT}]"
log "[cfg] FILL_FRAME_USER_SET=${FILL_FRAME_USER_SET} (1=user override honored)"
log "[cfg] STYLE_BY_LABEL=${STYLE_BY_LABEL} PERSON_LABELS='${PERSON_LABELS}' VEHICLE_LABELS='${VEHICLE_LABELS}'"

# ---- 1) Extract frames -----------------------------------------------------
if [[ "$SKIP_EXTRACT" != "1" ]]; then
  log "[1/3] Extract frames to canvas ${CANVAS_W}x${CANVAS_H} @ ${FPS}fps…"
  rm -rf "$FRAMES_DIR"
  mkdir -p "$FRAMES_DIR"
  if [[ "${FILL_FRAME}" == "1" ]]; then
    # Fill the canvas and center-crop (no pillarboxing/letterboxing)
    VF="scale=${CANVAS_W}:${CANVAS_H}:flags=lanczos:force_original_aspect_ratio=increase,crop=${CANVAS_W}:${CANVAS_H},fps=${FPS}"
  else
    # Fit inside canvas and pad (may introduce bars)
    VF="scale=${CANVAS_W}:${CANVAS_H}:flags=lanczos:force_original_aspect_ratio=decrease,pad=${CANVAS_W}:${CANVAS_H}:(ow-iw)/2:(oh-ih)/2:color=black,fps=${FPS}"
  fi
  ffmpeg -hide_banner -loglevel warning -nostats -y -i "$INPUT_VIDEO" -vf "$VF" \
         "$FRAMES_DIR/frame_%04d.${FRAME_EXT}"
else
  log "[1/3] SKIP_EXTRACT=1 — reusing frames in $FRAMES_DIR"
fi

# Quick sanity on frames
if ! ls -1 "$FRAMES_DIR"/frame_*."$FRAME_EXT" >/dev/null 2>&1; then
  die "No frames found in $FRAMES_DIR"
fi
FRAME_COUNT=$(ls -1 "$FRAMES_DIR"/frame_*."$FRAME_EXT" | wc -l | tr -d ' ')
log "[info] Frames ready: $FRAME_COUNT in $FRAMES_DIR"

# ---- 2) Generate masks -----------------------------------------------------
if [[ "$SKIP_MASKS" != "1" ]]; then
  if [[ "${STYLE_BY_LABEL}" == "1" ]]; then
    log "[2/3] Batch-generate masks by label…"
    # person masks
    MASKS_DIR_PERSON="${WORK_ROOT}/masks_person"
    rm -rf "$MASKS_DIR_PERSON"; mkdir -p "$MASKS_DIR_PERSON"
    ARGS_PERSON=(/app/sky_swap.py
                 --batch_frames "$FRAMES_DIR"
                 --batch_out_dir "$MASKS_DIR_PERSON"
                 --weights "$DEEPLAB_WEIGHTS"
                 --backbone "$BACKBONE"
                 --resolution "$MASK_RES"
                 --mask_expand_pct "$MASK_EXPAND_PCT"
                 --mask_feather_pct "$MASK_FEATHER_PCT"
                 --target_labels "$PERSON_LABELS"
                 --verbose)
    python "${ARGS_PERSON[@]}"

    # vehicle masks
    MASKS_DIR_VEHICLE="${WORK_ROOT}/masks_vehicle"
    rm -rf "$MASKS_DIR_VEHICLE"; mkdir -p "$MASKS_DIR_VEHICLE"
    ARGS_VEHICLE=(/app/sky_swap.py
                  --batch_frames "$FRAMES_DIR"
                  --batch_out_dir "$MASKS_DIR_VEHICLE"
                  --weights "$DEEPLAB_WEIGHTS"
                  --backbone "$BACKBONE"
                  --resolution "$MASK_RES"
                  --mask_expand_pct "$MASK_EXPAND_PCT"
                  --mask_feather_pct "$MASK_FEATHER_PCT"
                  --target_labels "$VEHICLE_LABELS"
                  --verbose)
    python "${ARGS_VEHICLE[@]}"

    # Keep MASKS_DIR pointing at the "person" set for sanity checks below
    MASKS_DIR="$MASKS_DIR_PERSON"
  else
    log "[2/3] Batch-generate masks (res=${MASK_RES}, expand=${MASK_EXPAND_PCT}%, feather=${MASK_FEATHER_PCT}%)…"
    rm -rf "$MASKS_DIR"
    mkdir -p "$MASKS_DIR"

    ARGS=(/app/sky_swap.py
          --batch_frames "$FRAMES_DIR"
          --batch_out_dir "$MASKS_DIR"
          --weights "$DEEPLAB_WEIGHTS"
          --backbone "$BACKBONE"
          --resolution "$MASK_RES"
          --mask_expand_pct "$MASK_EXPAND_PCT"
          --mask_feather_pct "$MASK_FEATHER_PCT"
          --verbose)

    if [[ -n "$MASK_TARGET_LABELS" ]]; then
      ARGS+=(--target_labels "$MASK_TARGET_LABELS")
    else
      ARGS+=(--scan_sky)
    fi

    python "${ARGS[@]}"
  fi
else
  log "[2/3] SKIP_MASKS=1 — reusing masks in $MASKS_DIR"
fi

# Quick sanity on masks
if ! ls -1 "$MASKS_DIR"/mask_*.png >/dev/null 2>&1; then
  log "[mask][WARN] no masks found under $MASKS_DIR — frames will be fully stylized."
fi

if [[ "${STYLE_BY_LABEL}" == "1" ]]; then
  if ! ls -1 "$WORK_ROOT"/masks_person/mask_*.png >/dev/null 2>&1; then
    die "STYLE_BY_LABEL=1 but no person masks in $WORK_ROOT/masks_person"
  fi
  if ! ls -1 "$WORK_ROOT"/masks_vehicle/mask_*.png >/dev/null 2>&1; then
    log "[mask][WARN] no vehicle masks in $WORK_ROOT/masks_vehicle — vehicle pass will be skipped"
    NO_VEHICLE_MASKS=1
  else
    NO_VEHICLE_MASKS=0
  fi
fi

_pick_io_preset_for_kind() {
  local kind="$1"
  case "$kind" in
    transformer) echo "imagenet_255" ;;
    reconet)     echo "imagenet_01"  ;;
    torch7)      echo "caffe_bgr"    ;;
    magenta)     echo "tanh"         ;;
    *)           echo "imagenet_255" ;;
  esac
}

# ---- 3) Style per-variant and assemble ------------------------------------
if [[ "${STYLE_BY_LABEL}" == "1" ]]; then
  # --- Three-pass composition: BASE -> PERSON overlay -> VEHICLE overlay ---
  # Resolve models and types (fall back when needed)
  _BASE_MODEL_PATH="${BASE_MODEL:-$STYLE_MODEL}"
  _BASE_MODEL_TYPE="${BASE_MODEL_TYPE}"

  if [[ "${BASE_AS_ORIGINAL}" == "1" ]]; then
    # We will use raw frames as the base; no model required
    _BASE_MODEL_PATH=""
  fi

  if [[ -z "${_BASE_MODEL_PATH}" && "${BASE_AS_ORIGINAL}" != "1" ]]; then
    die "STYLE_BY_LABEL=1 requires BASE_MODEL or STYLE_MODEL (or set BASE_AS_ORIGINAL=1 to keep background unstyled)"
  fi

  _PERSON_MODEL_PATH="${PERSON_MODEL}"
  _PERSON_MODEL_TYPE="${PERSON_MODEL_TYPE}"
  [[ -z "${_PERSON_MODEL_PATH}" ]] && die "STYLE_BY_LABEL=1 requires PERSON_MODEL"

  _VEHICLE_MODEL_PATH="${VEHICLE_MODEL:-}"
  _VEHICLE_MODEL_TYPE="${VEHICLE_MODEL_TYPE}"

  # Pick IO presets if global is auto
  _BASE_IO="${IO_PRESET}"
  [[ "${_BASE_IO}" == "auto" ]] && _BASE_IO="$(_pick_io_preset_for_kind "${_BASE_MODEL_TYPE}")"
  _PERSON_IO="${IO_PRESET}"
  [[ "${_PERSON_IO}" == "auto" ]] && _PERSON_IO="$(_pick_io_preset_for_kind "${_PERSON_MODEL_TYPE}")"
  _VEHICLE_IO="${IO_PRESET}"
  [[ -n "${_VEHICLE_MODEL_PATH}" && "${_VEHICLE_IO}" == "auto" ]] && _VEHICLE_IO="$(_pick_io_preset_for_kind "${_VEHICLE_MODEL_TYPE}")"

  # Pass 1: BASE over full frame (or reuse original frames)
  if [[ "${BASE_AS_ORIGINAL}" == "1" ]]; then
    BASE_DIR="$FRAMES_DIR"
    log "  >> BASE pass skipped (BASE_AS_ORIGINAL=1) — using original frames as background"
  else
    BASE_DIR="$WORK_ROOT/styled_base"
    rm -rf "$BASE_DIR"; mkdir -p "$BASE_DIR"
    CMD_BASE=(python /app/pipeline.py
              --input_dir "$FRAMES_DIR"
              --output_dir "$BASE_DIR"
              --image_ext "$FRAME_EXT"
              --model "$_BASE_MODEL_PATH"
              --model_type "$_BASE_MODEL_TYPE"
              --io_preset "$_BASE_IO"
              --device "$DEVICE"
              --fps "$FPS"
              --scale "$SCALE"
              --inference_res "$INFER_RES"
              --blend "$BLEND")
    log "  >> BASE command:"; printf "     %q " "${CMD_BASE[@]}"; echo
    env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" HOME="/root" "${CMD_BASE[@]}"
  fi

  # Pass 2: PERSON overlay on BASE
  PERSON_DIR="$WORK_ROOT/styled_pass_person"
  rm -rf "$PERSON_DIR"; mkdir -p "$PERSON_DIR"
  CMD_PERSON=(python /app/pipeline.py
              --input_dir "$BASE_DIR"
              --output_dir "$PERSON_DIR"
              --image_ext "$FRAME_EXT"
              --model "$_PERSON_MODEL_PATH"
              --model_type "$_PERSON_MODEL_TYPE"
              --io_preset "$_PERSON_IO"
              --device "$DEVICE"
              --fps "$FPS"
              --scale "$SCALE"
              --inference_res "$INFER_RES"
              --blend "$BLEND"
              --mask_dir "$WORK_ROOT/masks_person"
              --fit_mask_to input
              --composite_mode keep)
  # If BASE_AS_ORIGINAL=1, default to not inverting (apply style inside the person mask).
  # Otherwise, respect PERSON_MASK_INVERT as before.
  if [[ "${BASE_AS_ORIGINAL}" == "1" ]]; then
    :
  else
    [[ "${PERSON_MASK_INVERT}" == "1" ]] && CMD_PERSON+=(--mask_invert)
  fi
  [[ "$MASK_AUTOFIX" == "1" ]] && CMD_PERSON+=(--mask_autofix)
  log "  >> PERSON command:"; printf "     %q " "${CMD_PERSON[@]}"; echo
  env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" HOME="/root" "${CMD_PERSON[@]}"

  # Pass 3: VEHICLE overlay on PERSON result (if any vehicle masks)
  FINAL_DIR="$PERSON_DIR"
  if [[ -n "${_VEHICLE_MODEL_PATH}" && "${NO_VEHICLE_MASKS:-0}" != "1" ]]; then
    VEHICLE_DIR="$WORK_ROOT/styled_pass_vehicle"
    rm -rf "$VEHICLE_DIR"; mkdir -p "$VEHICLE_DIR"
    CMD_VEH=(python /app/pipeline.py
             --input_dir "$PERSON_DIR"
             --output_dir "$VEHICLE_DIR"
             --image_ext "$FRAME_EXT"
             --model "$_VEHICLE_MODEL_PATH"
             --model_type "$_VEHICLE_MODEL_TYPE"
             --io_preset "$_VEHICLE_IO"
             --device "$DEVICE"
             --fps "$FPS"
             --scale "$SCALE"
             --inference_res "$INFER_RES"
             --blend "$BLEND"
             --mask_dir "$WORK_ROOT/masks_vehicle"
             --fit_mask_to input
             --composite_mode keep)
    [[ "${VEHICLE_MASK_INVERT}" == "1" ]] && CMD_VEH+=(--mask_invert)
    [[ "$MASK_AUTOFIX" == "1" ]] && CMD_VEH+=(--mask_autofix)
    log "  >> VEHICLE command:"; printf "     %q " "${CMD_VEH[@]}"; echo
    env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" HOME="/root" "${CMD_VEH[@]}"
    FINAL_DIR="$VEHICLE_DIR"
  fi

  # Assemble final composite
  OUT_NAME="${IN_STEM}_bylabel_cw${CANVAS_W}ch${CANVAS_H}_fps${FPS}_ir${INFER_RES}_mr${MASK_RES}_exp${MASK_EXPAND_PCT}_feath${MASK_FEATHER_PCT}.mp4"
  [[ -n "$OUTPUT_VIDEO" ]] && OUT_NAME="$(basename -- "$OUTPUT_VIDEO")"
  log "  >> assembling ${OUT_NAME} from ${FINAL_DIR}/styled_frame_%04d.${FRAME_EXT} …"
  ffmpeg -hide_banner -loglevel warning -nostats -y \
         -framerate "$FPS" \
         -i "${FINAL_DIR}/styled_frame_%04d.${FRAME_EXT}" \
         -c:v libx264 -pix_fmt yuv420p "${OUT_DIR}/${OUT_NAME}"

else
  # --- Original sweep behavior (unchanged) ---
  for SPEC in "${MODELS_TO_RUN[@]}"; do
    MODEL_PATH="${SPEC%%|*}"
    MODEL_KIND="${SPEC##*|}"
    [[ -f "$MODEL_PATH" ]] || die "model not found: $MODEL_PATH"

    IO_PRESET_EFF="$IO_PRESET"
    if [[ "$IO_PRESET_EFF" == "auto" ]]; then
      IO_PRESET_EFF="$(_pick_io_preset_for_kind "${MODEL_KIND}")"
    fi

    TAG="$(basename "$MODEL_PATH")"
    TAG="${TAG%.pth}"
    TAG="${TAG%.t7}"

    IFS=',' read -r -a VARS <<< "$VARIANTS"
    for VAR in "${VARS[@]}"; do
      VAR_TRIM="${VAR//[[:space:]]/}"
      [[ -z "$VAR_TRIM" ]] && continue

      STYLED_DIR="$WORK_ROOT/styled_${VAR_TRIM}_${TAG}"
      rm -rf "$STYLED_DIR"; mkdir -p "$STYLED_DIR"

      log "  → ${TAG} (${VAR_TRIM})"
      log "     model_kind=${MODEL_KIND} io_preset=${IO_PRESET_EFF} (global=${IO_PRESET})"
      CMD=(python /app/pipeline.py
           --input_dir "$FRAMES_DIR"
           --output_dir "$STYLED_DIR"
           --image_ext "$FRAME_EXT"
           --model "$MODEL_PATH"
           --model_type "$MODEL_KIND"
           --io_preset "$IO_PRESET_EFF"
           --device "$DEVICE"
           --fps "$FPS"
           --scale "$SCALE"
           --inference_res "$INFER_RES"
           --blend "$BLEND")

      if ls -1 "$MASKS_DIR"/mask_*.png >/dev/null 2>&1; then
        CMD+=(--mask_dir "$MASKS_DIR" --fit_mask_to input --composite_mode keep)
        [[ "$VAR_TRIM" == "fg" ]] && CMD+=(--mask_invert)
        [[ "$MASK_AUTOFIX" == "1" ]] && CMD+=(--mask_autofix)
        [[ "$MASK_DEBUG_OVERLAY" == "1" ]] && CMD+=(--mask_debug_overlay)
      fi

      [[ "$FLOW_EMA" == "1" ]] && CMD+=(--flow_ema --flow_alpha "$FLOW_ALPHA")
      [[ "$SMOOTHING" == "1" ]] && CMD+=(--smooth_lightness --smooth_alpha 0.7)

      log "  >> pipeline.py command:"
      printf "     %q " "${CMD[@]}"; echo
      env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" HOME="/root" "${CMD[@]}"

      OUT_NAME="${IN_STEM}_${VAR_TRIM}_${TAG}_cw${CANVAS_W}ch${CANVAS_H}_fps${FPS}_ir${INFER_RES}_mr${MASK_RES}_exp${MASK_EXPAND_PCT}_feath${MASK_FEATHER_PCT}.mp4"
      [[ -n "$OUTPUT_VIDEO" ]] && OUT_NAME="$(basename -- "$OUTPUT_VIDEO")"
      log "  >> assembling ${OUT_NAME} from ${STYLED_DIR}/styled_frame_%04d.${FRAME_EXT} …"
      ffmpeg -hide_banner -loglevel warning -nostats -y \
             -framerate "$FPS" \
             -i "${STYLED_DIR}/styled_frame_%04d.${FRAME_EXT}" \
             -c:v libx264 -pix_fmt yuv420p "${OUT_DIR}/${OUT_NAME}"
    done
  done
fi
log "Done. Outputs in: $OUT_DIR"
ls -1 "$OUT_DIR"/*.mp4 2>/dev/null || true