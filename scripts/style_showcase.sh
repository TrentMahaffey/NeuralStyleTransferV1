#!/bin/bash
# Style Showcase - Generate morph videos showcasing neural style transfers
#
# Usage:
#   ./style_showcase.sh                     # Process all images in input/
#   ./style_showcase.sh --motion ken_burns  # With Ken Burns effect
#   ./style_showcase.sh --motion random     # Random motion per clip
#   ./style_showcase.sh --styles candy,mosaic,starry_night
#   ./style_showcase.sh --list              # List available styles
#
# Motion options: none, zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, ken_burns, random
#
# Environment variables (can also be passed as args):
#   IN_DIR=/app/input          Input directory
#   OUT_DIR=/app/output        Output directory
#   SCALE=720                  Output height
#   FPS=24                     Frames per second
#   HOLD_MODEL=1.5             Seconds to hold each style
#   TRANS=1.0                  Transition duration
#   TRANSITION=fade            FFmpeg xfade type
#   MAX_MODELS=10              Max styles per image
#   MOTION=none                Motion effect

cd "$(dirname "$0")"
docker-compose run --rm style bash -lc "python /app/web/style_showcase.py $*"
