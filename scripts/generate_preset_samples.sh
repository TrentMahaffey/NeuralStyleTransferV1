#!/usr/bin/env bash
set -euo pipefail

# Generate preset sample images
# Usage:
#   ./generate_preset_samples.sh              # Generate all (skip existing)
#   ./generate_preset_samples.sh --force      # Force regenerate all
#   ./generate_preset_samples.sh --list       # List all presets
#   ./generate_preset_samples.sh --retries 5  # Set max retries (default: 3)

cd "$(dirname "$0")"

FORCE=""
LIST=""
RETRIES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f)
      FORCE="--force"
      shift
      ;;
    --list|-l)
      LIST="--list"
      shift
      ;;
    --retries|-r)
      RETRIES="--max-retries $2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./generate_preset_samples.sh [--force] [--list] [--retries N]"
      exit 1
      ;;
  esac
done

if [[ -n "$LIST" ]]; then
  echo "Listing all presets..."
  docker-compose run --rm web bash -lc "python /web/generate_preset_samples.py --list"
else
  echo "Generating preset sample images..."
  echo "Options: $FORCE $RETRIES"
  docker-compose run --rm web bash -lc "python /web/generate_preset_samples.py $FORCE $RETRIES"
fi

echo ""
echo "Done! Samples saved to static/preset_samples/"
