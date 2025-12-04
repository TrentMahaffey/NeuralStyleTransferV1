#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# mask.sh - Generate DeepLab semantic segmentation masks
#
# Creates masks for different object classes (person, sky, etc.)
# that can be used with the style transfer pipeline.
#
# Usage:
#   ./mask.sh -i IMAGE [-t TARGET_IDS] [-o OUTPUT] [-r RESOLUTION]
#
# Examples:
#   ./mask.sh -i input.jpg -t 15           # person mask
#   ./mask.sh -i input.jpg -t 0            # background mask
#   ./mask.sh -i input.jpg -t 15,0         # person + background
#   ./mask.sh -i input.jpg -t sky          # sky detection (alias)
#
# DeepLab VOC Class IDs:
#   0  = background
#   1  = aeroplane
#   2  = bicycle
#   3  = bird
#   4  = boat
#   5  = bottle
#   6  = bus
#   7  = car
#   8  = cat
#   9  = chair
#   10 = cow
#   11 = dining table
#   12 = dog
#   13 = horse
#   14 = motorbike
#   15 = person
#   16 = potted plant
#   17 = sheep
#   18 = sofa
#   19 = train
#   20 = tv/monitor
# ============================================================

IMG_ARG=""
TARGET_IDS="15"
OUTPUT_ARG=""
RESOLUTION="1536"
FEATHER="0.0"
EXPAND="0.5"
CONTRACT="0.5"
MORPH_KS="5"
INVERT=""

usage() {
  cat <<EOF
Usage: $0 -i IMAGE [-t TARGET_IDS] [-o OUTPUT] [-r RESOLUTION] [options]

Required:
  -i IMAGE         Input image path

Options:
  -t TARGET_IDS    Comma-separated class IDs or alias (default: 15 for person)
                   Aliases: person=15, background=0, sky=0, vehicle=6,7,14
  -o OUTPUT        Output mask path (default: <input>_mask.png)
  -r RESOLUTION    Processing resolution (default: 1536)
  -f FEATHER       Feather percent 0.0-1.0 (default: 0.0)
  -e EXPAND        Expand mask percent (default: 0.5)
  -c CONTRACT      Contract mask percent (default: 0.5)
  -m MORPH_KS      Morphological close kernel size (default: 5)
  --invert         Invert the output mask
  -h               Show this help

Common class IDs:
  0=background, 6=bus, 7=car, 14=motorbike, 15=person
EOF
}

# Parse aliases
resolve_target() {
  local target="$1"
  case "$target" in
    person)     echo "15" ;;
    background) echo "0" ;;
    sky)        echo "0" ;;  # sky is typically class 0 in outdoor scenes
    vehicle)    echo "6,7,14" ;;
    animal)     echo "3,8,10,12,13,17" ;;
    furniture)  echo "9,11,18" ;;
    *)          echo "$target" ;;  # assume numeric
  esac
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) IMG_ARG="$2"; shift 2 ;;
    -t) TARGET_IDS="$(resolve_target "$2")"; shift 2 ;;
    -o) OUTPUT_ARG="$2"; shift 2 ;;
    -r) RESOLUTION="$2"; shift 2 ;;
    -f) FEATHER="$2"; shift 2 ;;
    -e) EXPAND="$2"; shift 2 ;;
    -c) CONTRACT="$2"; shift 2 ;;
    -m) MORPH_KS="$2"; shift 2 ;;
    --invert) INVERT="--invert_mask"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$IMG_ARG" ]]; then
  echo "[error] -i IMAGE is required" >&2
  usage
  exit 1
fi

# Resolve paths
if [[ "$IMG_ARG" != /* ]]; then
  IMG_IN="/app/input/${IMG_ARG}"
else
  IMG_IN="$IMG_ARG"
fi

STEM="$(basename -- "${IMG_IN%.*}")"

if [[ -z "$OUTPUT_ARG" ]]; then
  OUTPUT_ARG="/app/input/masks/${STEM}_mask.png"
fi

echo "[mask.sh] Input: $IMG_IN"
echo "[mask.sh] Target IDs: $TARGET_IDS"
echo "[mask.sh] Output: $OUTPUT_ARG"
echo "[mask.sh] Resolution: $RESOLUTION"

# Run in docker
docker compose run --rm \
  -e IMG_IN="$IMG_IN" \
  -e TARGET_IDS="$TARGET_IDS" \
  -e OUTPUT="$OUTPUT_ARG" \
  -e RESOLUTION="$RESOLUTION" \
  -e FEATHER="$FEATHER" \
  -e EXPAND="$EXPAND" \
  -e CONTRACT="$CONTRACT" \
  -e MORPH_KS="$MORPH_KS" \
  -e INVERT="$INVERT" \
  style bash -lc '
set -euo pipefail

IMG_IN="${IMG_IN}"
TARGET_IDS="${TARGET_IDS:-15}"
OUTPUT="${OUTPUT}"
RESOLUTION="${RESOLUTION:-1536}"
FEATHER="${FEATHER:-0.0}"
EXPAND="${EXPAND:-0.5}"
CONTRACT="${CONTRACT:-0.5}"
MORPH_KS="${MORPH_KS:-5}"
INVERT="${INVERT:-}"

STEM="$(basename -- "${IMG_IN%.*}")"
WORK="/app/_work/masks"
mkdir -p "$WORK" "$(dirname "$OUTPUT")"

# Check inputs
if [[ ! -f "$IMG_IN" ]]; then
  echo "[error] Input image not found: $IMG_IN"
  exit 1
fi

if [[ ! -f "/app/models/deeplab/deeplab-resnet.pth.tar" ]]; then
  echo "[error] DeepLab weights not found at /app/models/deeplab/deeplab-resnet.pth.tar"
  exit 1
fi

# Bake input (normalize orientation/format)
BAKED="$WORK/${STEM}_baked.png"
echo "[bake] Normalizing input -> $BAKED"
ffmpeg -hide_banner -loglevel warning -y -i "$IMG_IN" \
  -vf "scale=iw:ih:flags=lanczos,setsar=1" -pix_fmt rgb24 \
  "$BAKED"

# Run DeepLab segmentation
echo "[deeplab] Generating mask for target IDs: $TARGET_IDS"
python /app/sky_swap.py \
  --image "$BAKED" \
  --weights /app/models/deeplab/deeplab-resnet.pth.tar \
  --backbone resnet \
  --resolution "$RESOLUTION" \
  --mask_expand_pct "$EXPAND" \
  --mask_contract_pct "$CONTRACT" \
  --mask_feather_pct "$FEATHER" \
  --target_ids "$TARGET_IDS" \
  --morph_close_ks "$MORPH_KS" \
  --out_mask "$OUTPUT" \
  $INVERT

echo "✅ Mask saved to: $OUTPUT"

# Show mask info
echo "[info] Mask dimensions: $(identify -format "%wx%h" "$OUTPUT" 2>/dev/null || echo "unknown")"
'

echo ""
echo "Done! Mask saved to: $OUTPUT_ARG"
