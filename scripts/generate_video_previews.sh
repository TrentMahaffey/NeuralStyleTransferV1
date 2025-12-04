#!/usr/bin/env bash
set -euo pipefail

# Generate video previews for presets
# Usage:
#   ./generate_video_previews.sh              # Generate all (regions + morph)
#   ./generate_video_previews.sh --regions    # Region presets only
#   ./generate_video_previews.sh --morph      # Morph previews only
#   ./generate_video_previews.sh --morph 10   # Limit morph to 10 presets

cd "$(dirname "$0")"

MODE="${1:-all}"
LIMIT="${2:-0}"

case "$MODE" in
  --regions)
    echo "Generating Region preset video previews..."
    docker-compose run --rm web bash -lc "python /web/generate_video_samples.py --regions-only"
    ;;
  --morph)
    echo "Generating morph-style video previews (limit: ${LIMIT:-all})..."
    if [[ "$LIMIT" -gt 0 ]]; then
      docker-compose run --rm web bash -lc "python /web/generate_video_samples.py --morph-only --morph-limit $LIMIT"
    else
      docker-compose run --rm web bash -lc "python /web/generate_video_samples.py --morph-only"
    fi
    ;;
  --list)
    echo "Listing presets by type..."
    docker-compose run --rm web bash -lc "python /web/generate_video_samples.py --list"
    ;;
  *)
    echo "Generating ALL video previews (regions + morph)..."
    docker-compose run --rm web bash -lc "python /web/generate_video_samples.py"
    ;;
esac

# Refresh metadata to pick up new videos
echo ""
echo "Refreshing preset metadata..."
docker-compose run --rm web bash -lc "python -c \"
import sys
sys.path.insert(0, '/web')
from generate_preset_samples import save_preset_metadata
save_preset_metadata()
\""

echo ""
echo "Done! Video previews saved to static/preset_samples/"
