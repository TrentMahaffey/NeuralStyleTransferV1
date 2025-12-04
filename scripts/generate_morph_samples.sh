#!/bin/bash
# Generate animated GIF samples for Morph presets
# Usage: ./generate_morph_samples.sh [--force]

cd "$(dirname "$0")"
docker-compose run --rm web bash -lc "python /web/generate_morph_samples.py $*"
