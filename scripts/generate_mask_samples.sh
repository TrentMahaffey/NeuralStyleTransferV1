#!/usr/bin/env bash
set -euo pipefail

# Generate DeepLab semantic mask sample images
# Usage:
#   ./generate_mask_samples.sh              # Generate all (skip existing)
#   ./generate_mask_samples.sh --force      # Force regenerate all
#   ./generate_mask_samples.sh --list       # List expected input photos

cd "$(dirname "$0")"

ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f)
      ARGS="$ARGS --force"
      shift
      ;;
    --list|-l)
      ARGS="$ARGS --list"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./generate_mask_samples.sh [--force] [--list]"
      exit 1
      ;;
  esac
done

echo "DeepLab Mask Sample Generator"
echo "=============================="
echo ""

if [[ "$ARGS" == *"--list"* ]]; then
  docker-compose run --rm web bash -lc "python /web/generate_mask_samples.py --list"
else
  echo "Input photos go in: ../NeuralStyleTransferV1/input/mask_samples/"
  echo "Output saved to: static/mask_samples/"
  echo ""
  docker-compose run --rm web bash -lc "python /web/generate_mask_samples.py$ARGS"
fi

echo ""
echo "Done!"
