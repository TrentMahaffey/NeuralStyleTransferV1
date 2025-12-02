docker compose run --rm style bash -lc '
set -euo pipefail
shopt -s nullglob

# ---------- knobs ----------
SLIDES_DIR=/app/_work/winter1_clipped/slides
OUT=/app/output/slideshow.mp4
FPS=30
HOLD=1          # seconds fully visible before each crossfade
TRANS=2         # seconds crossfade duration
TAIL=1          # seconds to hold the last frame
W=1920
H=1080
CRF=18          # quality for libx264
PRESET=slow     # encoding preset
# ---------------------------

# Check for bc availability
if ! command -v bc >/dev/null; then
  echo "[error] bc not found in the Docker environment"
  exit 1
fi
echo "[debug] bc version: $(bc --version 2>&1 | head -n 1)"

TMP="/app/_work/slideshow_tmp"
rm -rf "$TMP" "$OUT"; mkdir -p "$TMP/clips" "$TMP/fades"

# Gather images by mtime (oldest → newest)
mapfile -t IMGS < <(
  find "$SLIDES_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
    -printf "%T@ %p\n" | sort -n | cut -d" " -f2-
)
[ "${#IMGS[@]}" -ge 2 ] || { echo "[error] need at least 2 images in $SLIDES_DIR"; exit 1; }

# Per-slide duration
DURATION=$(( HOLD + TRANS ))

# Make CFR clip for each image with fade-in and fade-out
i=0
for f in "${IMGS[@]}"; do
  base=$(printf "clip_%03d.mp4" "$i")
  duration=$DURATION
  if [ $i -eq $(( ${#IMGS[@]} - 1 )) ]; then
    duration=$(( DURATION + TAIL )) # Add tail to last clip
  fi
  echo "[make] $(basename "$f") -> $base  (duration=${duration}s @ ${FPS}fps)"
  if [ $i -eq 0 ]; then
    # First clip: only fade-out
    ffmpeg -hide_banner -loglevel warning -y \
      -loop 1 -framerate ${FPS} -i "$f" \
      -vf "scale=${W}:${H}:flags=lanczos:force_original_aspect_ratio=decrease,\
pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:white,\
fps=fps=${FPS},format=rgb24,setsar=1,\
fade=type=out:duration=${TRANS}:start_time=${HOLD}" \
      -t "${duration}" \
      -an -c:v libx264 -pix_fmt yuv420p -crf ${CRF} -preset ${PRESET} -r ${FPS} \
      -movflags +faststart -video_track_timescale ${FPS} -fps_mode cfr \
      "$TMP/clips/$base"
  elif [ $i -eq $(( ${#IMGS[@]} - 1 )) ]; then
    # Last clip: only fade-in, include tail
    ffmpeg -hide_banner -loglevel warning -y \
      -loop 1 -framerate ${FPS} -i "$f" \
      -vf "scale=${W}:${H}:flags=lanczos:force_original_aspect_ratio=decrease,\
pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:white,\
fps=fps=${FPS},format=rgb24,setsar=1,\
fade=type=in:duration=${TRANS}:start_time=0" \
      -t "${duration}" \
      -an -c:v libx264 -pix_fmt yuv420p -crf ${CRF} -preset ${PRESET} -r ${FPS} \
      -movflags +faststart -video_track_timescale ${FPS} -fps_mode cfr \
      "$TMP/clips/$base"
  else
    # Middle clips: fade-in and fade-out
    ffmpeg -hide_banner -loglevel warning -y \
      -loop 1 -framerate ${FPS} -i "$f" \
      -vf "scale=${W}:${H}:flags=lanczos:force_original_aspect_ratio=decrease,\
pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:white,\
fps=fps=${FPS},format=rgb24,setsar=1,\
fade=type=in:duration=${TRANS}:start_time=0,\
fade=type=out:duration=${TRANS}:start_time=${HOLD}" \
      -t "${duration}" \
      -an -c:v libx264 -pix_fmt yuv420p -crf ${CRF} -preset ${PRESET} -r ${FPS} \
      -movflags +faststart -video_track_timescale ${FPS} -fps_mode cfr \
      "$TMP/clips/$base"
  fi
  i=$((i+1))
done

# Debug clip metadata and test fades
for clip in "$TMP"/clips/clip_*.mp4; do
  echo "[debug] Checking $clip"
  ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,r_frame_rate,time_base -of default=nk=1:noprint_wrappers=1 "$clip" | sed "s/^/[probe] /"
  echo "[debug] Testing fade effect for $(basename "$clip")"
  ffmpeg -hide_banner -loglevel warning -y -i "$clip" -c copy "$TMP/fades/test_$(basename "$clip")"
done

# Create concat list file
concat_list="$TMP/concat_list.txt"
: > "$concat_list"
clips=( "$TMP"/clips/clip_*.mp4 )

# Debug clips array
echo "[debug] Clips array contains ${#clips[@]} files:"
for clip in "${clips[@]}"; do
  echo "[debug] Clip: $clip"
done

# Generate concat list with full paths
for ((k=0; k<${#clips[@]}; k++)); do
  clip="${clips[$k]}"
  echo "[concat] Adding $(basename "$clip")"
  printf "file %s\n" "$clip" >> "$concat_list"
  if [ $k -lt $(( ${#clips[@]} - 1 )) ]; then
    printf "duration %d\n" $(( HOLD )) >> "$concat_list"
  fi
done

# Debug concat list
echo "[debug] Contents of $concat_list:"
cat "$concat_list"

# Concatenate clips
echo "[concat] Creating $OUT"
ffmpeg -hide_banner -loglevel warning -y \
  -f concat -safe 0 -i "$concat_list" \
  -c:v libx264 -pix_fmt yuv420p -crf ${CRF} -preset ${PRESET} -r ${FPS} \
  -movflags +faststart -video_track_timescale ${FPS} -fps_mode cfr \
  "$OUT"

# Sanity check
echo "[done] $OUT"
ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$OUT"
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,r_frame_rate,time_base -of default=nokey=1:noprint_wrappers=1 "$OUT" | sed "s/^/[probe] /"
'