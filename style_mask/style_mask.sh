#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# style_mask.sh
# Build FG/BG stylized stills (Magenta, Torch7, PyTorch) and
# assemble a crossfaded slideshow. Now supports flags:
#   -i INPUT   (image name or path; if bare name, assumed in /app/input)
#   -t IDS     (mask target IDs for DeepLab, e.g. "15" for person, "0" for background,
#               or comma-separated like "15,0")
#
# Examples:
#   ./style_mask.sh -i winter1.jpeg -t 15
#   ./style_mask.sh -i /app/input/frame_hq.jpeg -t 0
# ============================================================

IMG_ARG=""
TARGET_IDS_ARG="15"   # default = VOC person

usage() {
  cat <<EOF
Usage: $0 [-i INPUT_IMAGE] [-t TARGET_IDS]

  -i INPUT_IMAGE   Image to style. If it is a bare filename, it is resolved as /app/input/<name>.
                   You may also pass an absolute path like /app/input/foo.jpeg.
  -t TARGET_IDS    Comma-separated DeepLab class IDs to mask (default: 15 for person).
                   Common: 15=person, 0=background.
EOF
}

# ---- parse flags on host side ----
while getopts ":i:t:h" opt; do
  case "$opt" in
    i) IMG_ARG="$OPTARG" ;;
    t) TARGET_IDS_ARG="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Resolve image path: if no directory separators, assume /app/input/<name>
if [[ -z "${IMG_ARG}" ]]; then
  IMG_ARG="winter1.jpeg"     # keep old default behavior
fi
if [[ "$IMG_ARG" != /* ]]; then
  IMG_IN_DOCKER="/app/input/${IMG_ARG}"
else
  IMG_IN_DOCKER="$IMG_ARG"
fi

# Sanity echo
echo "[host] using IMG_IN=${IMG_IN_DOCKER}  TARGET_IDS=${TARGET_IDS_ARG}"

# ---- run the existing pipeline in the container,
#      reading IMG_IN and TARGET_IDS from env ----
docker compose run --rm \
  -e IMG_IN="${IMG_IN_DOCKER}" \
  -e TARGET_IDS="${TARGET_IDS_ARG}" \
  style bash -lc '
set -euo pipefail
shopt -s nullglob

# -------- settings --------
IMG_IN="${IMG_IN:-/app/input/winter1.jpeg}"       # from host -e
STEM="$(basename -- "${IMG_IN%.*}")"
WORK="/app/_work/${STEM}"
SLIDES="$WORK/slides"                             # all styled images go here
OUT="/app/output"
mkdir -p "$WORK" "$SLIDES" "$OUT"

# slideshow knobs
FPS=30
HOLD=1
FADE=2
TAIL=1
W=1920
H=1080

# -------- 0) bake input (remove EXIF, fix orientation) --------
ffmpeg -hide_banner -loglevel warning -y -i "$IMG_IN" \
  -vf "scale=iw:ih:flags=lanczos,setsar=1" -pix_fmt rgb24 \
  "$WORK/${STEM}_baked.png"

# -------- 1) DeepLab mask (configurable TARGET_IDS) --------
python /app/sky_swap.py \
  --image "$WORK/${STEM}_baked.png" \
  --weights /app/models/deeplab/deeplab-resnet.pth.tar \
  --backbone resnet \
  --resolution 1536 \
  --mask_expand_pct 0.5 \
  --mask_contract_pct 0.5 \
  --mask_feather_pct 0.0 \
  --target_ids "${TARGET_IDS:-15}" \
  --morph_close_ks 5 \
  --out_mask "$WORK/${STEM}_person_mask.png"

# -------- helpers --------
run_pair_magenta () {
  local STYLE_IMG="$1"         # /app/models/magenta_styles/*.jpg|png
  local TAG_RAW TAG
  TAG_RAW="$(basename -- "$STYLE_IMG")"
  TAG="${TAG_RAW%.*}"
  TAG="$(echo "$TAG" | tr "[:upper:]" "[:lower:]" | sed -E "s/[^a-z0-9]+/_/g")"
  echo "[magenta] style=$STYLE_IMG tag=$TAG"

  # FG (inside mask)
  python /app/pipeline.py \
    --input_image "$WORK/${STEM}_baked.png" \
    --output_image "$SLIDES/${STEM}_fg_magenta_${TAG}.png" \
    --model_type magenta \
    --io_preset tanh \
    --magenta_style "$STYLE_IMG" \
    --inference_res 1080 \
    --scale 720 \
    --mask "$WORK/${STEM}_person_mask.png" \
    --fit_mask_to input \
    --composite_mode keep \
    --mask_autofix

  # BG (invert mask)
  python /app/pipeline.py \
    --input_image "$WORK/${STEM}_baked.png" \
    --output_image "$SLIDES/${STEM}_bg_magenta_${TAG}.png" \
    --model_type magenta \
    --io_preset tanh \
    --magenta_style "$STYLE_IMG" \
    --inference_res 1080 \
    --scale 720 \
    --mask "$WORK/${STEM}_person_mask.png" \
    --fit_mask_to input \
    --mask_invert \
    --composite_mode keep \
    --mask_autofix
}

run_pair () {
  local MODEL_PATH="$1"        # .t7 or .pth
  local MODEL_TYPE="$2"        # torch7 | transformer
  local IO_PRESET="$3"         # caffe_bgr | imagenet_255
  local TAG="$4"

  # FG
  python /app/pipeline.py \
    --input_image "$WORK/${STEM}_baked.png" \
    --output_image "$SLIDES/${STEM}_fg_${TAG}.png" \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --io_preset "$IO_PRESET" \
    --inference_res 1080 \
    --scale 720 \
    --mask "$WORK/${STEM}_person_mask.png" \
    --fit_mask_to input \
    --composite_mode keep \
    --mask_autofix

  # BG
  python /app/pipeline.py \
    --input_image "$WORK/${STEM}_baked.png" \
    --output_image "$SLIDES/${STEM}_bg_${TAG}.png" \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --io_preset "$IO_PRESET" \
    --inference_res 1080 \
    --scale 720 \
    --mask "$WORK/${STEM}_person_mask.png" \
    --fit_mask_to input \
    --mask_invert \
    --composite_mode keep \
    --mask_autofix
}

# -------- 2) MAGENTA FIRST (style images) --------
for STYLE in /app/models/magenta_styles/*.{jpg,png}; do
  [ -f "$STYLE" ] || continue
  run_pair_magenta "$STYLE"
done

# -------- 3) Torch7 (.t7) --------
run_pair "/app/models/torch/composition_vii_eccv16.t7" torch7 caffe_bgr composition_vii_eccv16
run_pair "/app/models/torch/la_muse_eccv16.t7"         torch7 caffe_bgr la_muse_eccv16
run_pair "/app/models/torch/starry_night_eccv16.t7"    torch7 caffe_bgr starry_night_eccv16
run_pair "/app/models/torch/the_scream.t7"             torch7 caffe_bgr the_scream

# -------- 4) PyTorch (.pth) --------
run_pair "/app/models/pytorch/mosaic.pth"         transformer imagenet_255 mosaic
run_pair "/app/models/pytorch/rain_princess.pth"  transformer imagenet_255 rain_princess
run_pair "/app/models/pytorch/udnie.pth"          transformer imagenet_255 udnie

echo "✅ Styled PNGs saved to: $SLIDES"

# ===================== Slideshow build (from $SLIDES) =====================
OUT_MP4="$OUT/${STEM}_slideshow.mp4"
TMP="/app/_work/xfade_tmp_${STEM}"
rm -rf "$TMP" "$OUT_MP4"; mkdir -p "$TMP/clips"

# order slides by mtime (oldest -> newest)
mapfile -t imgs < <(
  find "$SLIDES" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
    -printf "%T@ %p\n" | sort -n | awk "{ \$1=\"\"; sub(/^ /,\"\"); print }"
)
[ "${#imgs[@]}" -ge 2 ] || { echo "[error] need at least 2 slides in $SLIDES"; exit 1; }

# per-slide clip length = HOLD + FADE + HOLD
LEN=$((HOLD+FADE+HOLD))

# 1) normalize each slide into a clip
i=0
for f in "${imgs[@]}"; do
  echo "[make] $(basename "$f") -> clip_$(printf "%03d" "$i").mp4"
  ffmpeg -hide_banner -loglevel warning -y \
    -loop 1 -t "$LEN" -i "$f" \
    -vf "scale=${W}:${H}:flags=lanczos:force_original_aspect_ratio=decrease,\
pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=black,fps=${FPS},format=yuv420p,setsar=1" \
    -an -c:v libx264 -pix_fmt yuv420p -r "$FPS" -movflags +faststart \
    "$TMP/clips/clip_$(printf "%03d" "$i").mp4"
  i=$((i+1))
done

# 2) sequential xfade (offset = duration(prev) - (HOLD + FADE))
prev="$TMP/clips/clip_000.mp4"
for n in $(seq -w 001 $(( ${#imgs[@]} - 1 ))); do
  next="$TMP/clips/clip_${n}.mp4"
  out="$TMP/xf_${n}.mp4"
  DPREV=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$prev")
  OFFSET=$(awk -v d="$DPREV" -v h="$HOLD" -v f="$FADE" "BEGIN{v=d-h-f; if(v<0)v=0; printf(\"%.6f\", v)}")
  echo "[xfade] $(basename "$prev") + $(basename "$next") @ offset=${OFFSET}s (fade=${FADE}s) -> $(basename "$out")"
  ffmpeg -hide_banner -loglevel warning -y \
    -i "$prev" -i "$next" \
    -filter_complex "[0:v][1:v]xfade=transition=fade:duration=${FADE}:offset=${OFFSET},format=yuv420p[v]" \
    -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -r "$FPS" -movflags +faststart "$out"
  prev="$out"
done

# 3) tail hold
ffmpeg -hide_banner -loglevel warning -y \
  -i "$prev" -filter_complex "tpad=stop_mode=clone:stop_duration=${TAIL}" \
  -an -c:v libx264 -pix_fmt yuv420p -r "$FPS" -movflags +faststart "$OUT_MP4"

# 4) duration report
printf "🎬 slideshow: %s\n" "$OUT_MP4"
printf "   slides=%d  hold=%ss  fade=%ss  tail=%ss  fps=%s\n" "${#imgs[@]}" "$HOLD" "$FADE" "$TAIL" "$FPS"
printf "   duration: "
ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$OUT_MP4" | awk "{printf(\"%.2f s\\n\", \$1)}"
'