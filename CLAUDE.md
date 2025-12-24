# Neural Style Transfer Video Pipeline

## Project Overview
Multi-model neural style transfer pipeline for video processing with region-based blending, supporting up to 8 simultaneous style models (A-H) with per-region scales and rotation.

## Quick Start

```bash
# Run via docker-compose
docker-compose run --rm style bash -lc 'python3 /app/pipeline.py --input_video /app/input_videos/in.mp4 --output_video /app/output/styled.mp4 --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 --scale 720'
```

## Architecture

### Core Files
- `pipeline.py` - Main video processing pipeline with multi-model support
- `region_blend.py` - Spatial region blending (voronoi, fractal, spiral, etc.)
- `run_videos.py` - Thin adapter between drive_videos.py and pipeline.py
- `drive_videos.py` - Video batch processing driver
- `docker-compose.yml` - Container configuration

### Model Backends
| Type | Extension | Flag | Directory |
|------|-----------|------|-----------|
| PyTorch Transformer | `.pth` | `--model_type transformer` | `models/pytorch/` |
| Torch7 (OpenCV DNN) | `.t7` | `--model_type torch7` | `models/torch/` |
| Magenta TF-Hub | N/A | `--model_type magenta` | `models/magenta_styles/` |
| ReCoNet | `.pth` | `--model_type reconet` | `models/reconet/` |

### Available Models
**PyTorch** (`models/pytorch/`): candy, mosaic, rain_princess, udnie
**Torch7** (`models/torch/`): the_scream, composition_vii_eccv16, la_muse_eccv16, starry_night_eccv16
**Magenta Styles** (`models/magenta_styles/`): starry_night, gptstyle4, gpt_style3, canyon, atoms, dunes2, frame, gpt_style2, mountain_geo, style_gpt, style_rainforest

## Multi-Model Region Blending

### Model Slots (A-H)
```bash
--model /path/to/model.pth --model_type transformer --io_preset raw_255      # Model A
--model_b /path/to/model.pth --model_b_type transformer --io_preset_b raw_255 # Model B
--model_c ... --model_c_type ... --io_preset_c ...                            # Model C
# ... through model_h
```

For magenta models:
```bash
--model_b magenta --model_b_type magenta --magenta_style_b /app/models/magenta_styles/starry_night.jpg
```

### Region Modes
`--region_mode`: grid, diagonal, voronoi, fractal, radial, waves, spiral, concentric, random

### Region Blend Spec
`--region_blend_spec "A:0.5+B:0.5|C|D:0.7+E:0.3|O"`
- Pipe `|` separates regions
- Plus `+` blends models within a region with weights
- `O` = original (unstyled)

### Per-Region Scales
`--region_scales "0.5|0.75|1.0|1.0"`
- Multiplier of base `--scale` for each region
- Reduces computation for some regions

### Variable Region Sizes
`--region_sizes "1,1,1,0.2"` (voronoi mode only)

Control relative sizes of each region - make some bigger, others smaller.

**Format**: Comma or pipe-separated relative weights
- Higher number = larger region
- Weights are normalized internally

**Examples**:
```bash
--region_sizes "1,1,1,0.2"      # Last region ~5x smaller (~5% of frame)
--region_sizes "1,1,1,1,0.1"    # 5 regions, last one tiny (~3% of frame)
--region_sizes "2,1,1,1"        # First region 2x larger than others
--region_sizes "1,1,1,0.05"     # Last region very small (~1.5% of frame)
```

### Region Rotation
`--region_rotate 2` - Rotate regions 2 degrees per frame (spinning effect)

### Region Morphing (Organic Animation)
`--region_morph "mode"` or `--region_morph "speed,amplitude,frequency,mode"`

Makes region boundaries animate organically like blobs or tentacles weaving in and out.

**Modes**:
| Mode | Effect |
|------|--------|
| `blob` | Organic, smooth morphing boundaries |
| `tentacle` | Elongated, reaching distortions |
| `wave` | Sinusoidal flowing motion |
| `pulse` | Radial pulsing from center |

**Parameters** (all optional, use defaults):
- **speed**: Animation speed multiplier (1.0 = normal)
- **amplitude**: How far boundaries move (0.15 = 15% of frame size)
- **frequency**: Detail level (3.0 = moderate, higher = more tentacles)

**Examples**:
```bash
--region_morph "blob"                    # Organic blobs with defaults
--region_morph "tentacle"                # Tentacle-like stretching
--region_morph "1.5,0.2,4.0,tentacle"   # Faster, larger, more detailed tentacles
--region_morph "0.5,0.1,2.0,wave"       # Slow, subtle waves
```

### Optimized Mode
`--region_optimize` - Only style pixels needed for each region (crop-based)
`--region_padding 48` - Padding around crops for convolution context

### Animated Blend Weights
Smoothly oscillate blend weights between models within each region using harmonic waveforms.

**Global animation (all regions same)**:
```bash
--blend_animate "72,sine"              # 72-frame sine cycle (3 sec at 24fps)
--blend_animate "72,sine,0,0.3,0.7"    # Smoother: opacity stays between 30-70%
```

**Per-region animation**:
```bash
--blend_animate_regions "48,sine|72,triangle|static|60,square"
```

**Format**: `period,waveform,phase,min,max`
- **period**: Frames per cycle (24=1sec, 72=3sec, 120=5sec at 24fps)
- **waveform**: `sine`, `triangle`, `sawtooth`, `sawtooth_down`, `square`
- **phase**: Starting phase offset in degrees (0-360)
- **min,max**: Opacity range (default 0.0,1.0 - use 0.3,0.7 for smoother transitions)

**Opacity range effects**:
| Range | Effect |
|-------|--------|
| `0.0,1.0` | Full swing - one model can fully dominate (jarring) |
| `0.3,0.7` | Gentle - always 30-70% blend, never fully one style |
| `0.4,0.6` | Subtle - mostly even blend with slight variation |

### Animated Scale (Resolution Pulsing)
Oscillate region resolution over time, creating pulsing detail effects.

**Global animation (all regions same)**:
```bash
--scale_animate "60,sine"              # 60-frame sine cycle, 0.5-1.0 scale
--scale_animate "60,sine,0,0.3,0.8"    # Custom scale range (30%-80% resolution)
```

**Per-region animation**:
```bash
--scale_animate_regions "60,sine|30,triangle|static|90,sawtooth"
```

**Format**: `period,waveform,phase,min,max`
- **period**: Frames per cycle
- **waveform**: `sine`, `triangle`, `sawtooth`, `sawtooth_down`, `square`
- **phase**: Phase offset in degrees (0-360)
- **min,max**: Scale range (default 0.5,1.0 - resolution oscillates between 50%-100%)

**Scale range effects**:
| Range | Effect |
|-------|--------|
| `0.5,1.0` | Standard pulsing - half to full resolution |
| `0.3,0.7` | Always reduced - softer look, faster processing |
| `0.8,1.0` | Subtle - slight detail variation |

## Common Commands

### Simple single-model styling
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/styled.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth \
  --model_type transformer --io_preset raw_255
'
```

### 4-region voronoi with different models
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/region4.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b /app/models/pytorch/mosaic.pth --model_b_type transformer --io_preset_b raw_255 \
  --model_c /app/models/pytorch/udnie.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d /app/models/torch/the_scream.t7 --model_d_type torch7 \
  --region_mode voronoi --region_count 4 --region_feather 25 \
  --region_blend_spec "A|B|C|D" \
  --region_optimize --region_padding 48
'
```

### 8-model with magenta, per-region scales, and rotation
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/region_8model_spin.mp4 \
  --scale 1080 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b magenta --model_b_type magenta --magenta_style_b /app/models/magenta_styles/starry_night.jpg \
  --model_c /app/models/pytorch/mosaic.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d magenta --model_d_type magenta --magenta_style_d /app/models/magenta_styles/gptstyle4.jpg \
  --model_e /app/models/pytorch/udnie.pth --model_e_type transformer --io_preset_e raw_255 \
  --model_f magenta --model_f_type magenta --magenta_style_f /app/models/magenta_styles/gpt_style3.jpg \
  --model_g /app/models/torch/the_scream.t7 --model_g_type torch7 \
  --model_h magenta --model_h_type magenta --magenta_style_h /app/models/magenta_styles/canyon.jpg \
  --magenta_tile 512 --magenta_overlap 64 \
  --region_mode voronoi --region_count 6 --region_feather 30 \
  --region_blend_spec "A:0.5+B:0.5|C:0.5+D:0.5|E:0.5+F:0.5|G:0.5+H:0.5|A:0.5+D:0.5|O" \
  --region_scales "0.5|0.6|0.75|0.5|0.6|1.0" \
  --region_rotate 2 \
  --region_optimize --region_padding 48
'
```

### Animated blends with spinning regions
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/harmonic_spin.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b /app/models/pytorch/mosaic.pth --model_b_type transformer --io_preset_b raw_255 \
  --model_c /app/models/pytorch/udnie.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d /app/models/torch/the_scream.t7 --model_d_type torch7 \
  --region_mode voronoi --region_count 4 --region_feather 25 \
  --region_blend_spec "A:0.5+B:0.5|C:0.5+D:0.5|A:0.5+C:0.5|B:0.5+D:0.5" \
  --blend_animate "72,sine,0,0.3,0.7" \
  --region_rotate 2 \
  --region_optimize --region_padding 48
'
```

### Per-region animation with different waveforms
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/multi_harmonic.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b /app/models/pytorch/mosaic.pth --model_b_type transformer --io_preset_b raw_255 \
  --model_c /app/models/pytorch/udnie.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d /app/models/torch/the_scream.t7 --model_d_type torch7 \
  --region_mode voronoi --region_count 4 --region_feather 25 \
  --region_blend_spec "A:0.5+B:0.5|C:0.5+D:0.5|A:0.5+C:0.5|B:0.5+D:0.5" \
  --blend_animate_regions "48,sine,0,0.3,0.7|72,triangle,0,0.3,0.7|static|60,sawtooth,0,0.3,0.7" \
  --region_rotate 2 \
  --region_optimize --region_padding 48
'
```

### Organic tentacle morphing with rotation
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/tentacle_morph.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b /app/models/pytorch/mosaic.pth --model_b_type transformer --io_preset_b raw_255 \
  --model_c /app/models/pytorch/udnie.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d /app/models/torch/the_scream.t7 --model_d_type torch7 \
  --region_mode voronoi --region_count 4 --region_feather 30 \
  --region_blend_spec "A|B|C|D" \
  --region_morph "1.5,0.15,3.0,tentacle" \
  --region_rotate 1 \
  --region_optimize --region_padding 48
'
```

### Blob morphing with animated blends (full psychedelic)
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/psychedelic.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b /app/models/pytorch/mosaic.pth --model_b_type transformer --io_preset_b raw_255 \
  --model_c /app/models/pytorch/udnie.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d /app/models/torch/the_scream.t7 --model_d_type torch7 \
  --region_mode spiral --region_count 4 --region_feather 35 \
  --region_blend_spec "A:0.5+B:0.5|C:0.5+D:0.5|A:0.5+C:0.5|B:0.5+D:0.5" \
  --blend_animate "72,sine,0,0.3,0.7" \
  --region_morph "blob" \
  --region_rotate 2 \
  --region_optimize --region_padding 48
'
```

### Pulsing scale with all animations combined
```bash
docker-compose run --rm style bash -lc '
python3 /app/pipeline.py \
  --input_video /app/input_videos/in.mp4 \
  --output_video /app/output/full_animated.mp4 \
  --scale 720 --fps 24 --blend 0.9 \
  --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 \
  --model_b /app/models/pytorch/mosaic.pth --model_b_type transformer --io_preset_b raw_255 \
  --model_c /app/models/pytorch/udnie.pth --model_c_type transformer --io_preset_c raw_255 \
  --model_d /app/models/torch/the_scream.t7 --model_d_type torch7 \
  --region_mode voronoi --region_count 4 --region_feather 30 \
  --region_blend_spec "A:0.5+B:0.5|C:0.5+D:0.5|A:0.5+C:0.5|B:0.5+D:0.5" \
  --blend_animate "72,sine,0,0.3,0.7" \
  --scale_animate "60,triangle,0,0.5,1.0" \
  --region_morph "tentacle" \
  --region_rotate 1 \
  --region_optimize --region_padding 48
'
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--scale` | Output resolution (longest edge) | 720 |
| `--fps` | Output framerate | 24 |
| `--blend` | Style/original blend (0-1) | 0.9 |
| `--max_frames` | Limit frames for testing | None |
| `--region_count` | Number of regions | 4 |
| `--region_feather` | Edge softness in pixels | 20 |
| `--region_seed` | Fixed seed for stable regions | 42 (optimized mode) |
| `--region_sizes` | Relative region sizes (voronoi only) | Equal |
| `--region_rotate` | Degrees to rotate regions per frame | 0 |
| `--region_morph` | Organic morphing mode (blob, tentacle, wave, pulse) | None |
| `--region_scales` | Per-region resolution multipliers | 1.0 |
| `--blend_animate` | Global blend animation spec | None |
| `--blend_animate_regions` | Per-region blend animation specs | None |
| `--scale_animate` | Global scale animation spec | None |
| `--scale_animate_regions` | Per-region scale animation specs | None |
| `--io_preset` | Input/output normalization | raw_255 |

### IO Presets
- `raw_255` - Input/output in 0-255 range (most .pth models)
- `imagenet_255` - ImageNet normalization, 0-255 scale
- `imagenet_01` - ImageNet normalization, 0-1 scale
- `tanh` - Output in -1 to 1 range

## File Locations
- Input videos: `input_videos/`
- Output videos: `output/`
- Work directory: `_work/frames/` (temporary frame storage)
- Models: `models/{pytorch,torch,magenta_styles,reconet}/`

## Web UI

The web interface has been moved to a separate repository:
- **Repo**: [NeuralStyleWeb](https://github.com/TrentMahaffey/NeuralStyleWeb)
- **Location**: `../NeuralStyleWeb/` (sibling folder)

To run the web UI:
```bash
cd ../NeuralStyleWeb
docker-compose up web
# Access at http://localhost:5001
```

The web UI's docker-compose mounts both repos:
- `/app` → This pipeline repo
- `/web` → NeuralStyleWeb repo

## Troubleshooting

### Regions flashing/changing
Add `--region_seed 42` or use `--region_optimize` (auto-fixes seed)

### Out of memory
- Reduce `--scale`
- Use `--region_scales` to lower resolution on some regions
- Use `--region_optimize` for crop-based processing

### Magenta model errors
Ensure `--magenta_style_X` is set for each magenta model slot

### Video artifacts (vertical lines, jitter)
The optical flow algorithm has been improved to reduce artifacts:
- Pre-blur applied to grayscale images before flow calculation
- Larger window size (21 vs 15) and more pyramid levels (6 vs 5)
- Post-processing flow field smoothing to reduce vertical line artifacts
- Cubic easing for smooth pan/zoom motion

## New Server Setup

### Prerequisites
- Docker and docker-compose
- Python 3.x with OpenCV, NumPy, PIL
- ffmpeg for video conversion
- Sufficient disk space for models (~2GB) and outputs

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/TrentMahaffey/NeuralStyleTransferV1.git
cd NeuralStyleTransferV1

# Build Docker container
docker-compose build style

# Test the pipeline
docker-compose run --rm style bash -lc "python3 /app/pipeline.py --help"
```

### Model Setup
Models are stored in `models/` directory with subdirectories:
- `models/pytorch/` - PyTorch transformer models (.pth)
- `models/torch/` - Torch7 models for OpenCV DNN (.t7)
- `models/magenta_styles/` - Style images for Magenta TF-Hub
- `models/deeplab/` - DeepLab semantic segmentation weights
- `models/reconet/` - ReCoNet video stylization models

DeepLab weights can be downloaded from the DeepLab-ResNet repository.

### Directory Structure
```
NeuralStyleTransferV1/
├── pipeline.py              # Main video processing pipeline
├── region_blend.py          # Region blending utilities
├── sky_swap.py              # DeepLab semantic segmentation
├── docker-compose.yml       # Docker configuration
├── CLAUDE.md                # This documentation
├── scripts/                 # Utility scripts
│   ├── morph_v2.py         # Self-style morph pipeline
│   ├── style_morph.py      # Cross-image weight flow morph
│   ├── batch_selfstyle_all_images.py
│   └── optical_flow_*.py   # Video slideshow generators
├── input_videos/            # Input video files
├── input/                   # Input images
│   └── self_style_samples/ # Images for self-style processing
├── output/                  # Generated outputs
├── models/                  # Model weights
└── _work/                   # Temporary frame storage
```

### Common Tasks

**Run MorphV2 self-style pipeline:**
```bash
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --auto --vertical"
```

**Generate batch self-style images:**
```bash
docker-compose run --rm style bash -lc "python /app/scripts/batch_selfstyle_all_images.py"
```

**Style a video with single model:**
```bash
docker-compose run --rm style bash -lc "python3 /app/pipeline.py --input_video /app/input_videos/in.mp4 --output_video /app/output/out.mp4 --model /app/models/pytorch/candy.pth --model_type transformer --io_preset raw_255 --scale 720"
```

## MorphV2 - Self-Style Morph Pipeline

`scripts/morph_v2.py` - Automated self-style morph video generation from a single image.

### Overview
Given a single input image, MorphV2:
1. Runs DeepLab semantic segmentation to detect regions (person, car, dog, etc.)
2. Automatically selects the best region using a scoring algorithm
3. Extracts the region as a tight crop (style source)
4. Runs Magenta arbitrary style transfer with multiple tile/overlap configurations
5. Generates optical flow morph video from the styled outputs

### Quick Start

```bash
# Automatic mode (recommended) - detects best region automatically
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --auto"

# Vertical video for mobile
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --auto --vertical"

# With Ken Burns pan/zoom effect
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --auto --vertical --pan_zoom 2.0 --pan_direction horizontal"

# Analyze image to see detected regions (no processing)
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --analyze"

# Manual target selection
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --target_label person"

# Skip mask, use whole image as style
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --skip_mask"
```

### Ken Burns Pan/Zoom Effect

Add cinematic camera movement to morph videos with `--pan_zoom` and `--pan_direction`.

```bash
# Horizontal pan (left to right) at 2x zoom
--pan_zoom 2.0 --pan_direction horizontal

# Vertical pan (top to bottom) at 1.5x zoom
--pan_zoom 1.5 --pan_direction vertical

# Diagonal pan (top-left to bottom-right)
--pan_zoom 2.0 --pan_direction diagonal

# Reverse diagonal (top-right to bottom-left)
--pan_zoom 2.0 --pan_direction diagonal_reverse
```

**How it works:**
- Images are loaded at `pan_zoom` times the target resolution
- Optical flow morphing is performed at this larger size
- A viewport is extracted from each frame, panning across as the video progresses
- Creates smooth camera movement through the styled artwork

### PyTorch Pre-Styling

Add neural style transfer before Magenta processing for richer effects:

```bash
# Random PyTorch model applied first
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --auto --pytorch_style"

# Specific PyTorch model
docker-compose run --rm style bash -lc "python /app/scripts/morph_v2.py --image /app/input/photo.jpg --auto --pytorch_style --pytorch_model candy"
```

Creates blend variants (0%, 25%, 50%, 75%, 100% PyTorch style) each processed through all 7 Magenta tile configurations.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--image` | Input image path (required) | - |
| `--auto` | Auto-detect best region | - |
| `--target_label` | Manual region selection (person, car, dog, etc.) | - |
| `--skip_mask` | Use whole image as style source | - |
| `--analyze` | Show detected regions without processing | - |
| `--vertical` | Output vertical video (720x1280) | horizontal |
| `--scale` | Output resolution for styled images | 1440 |
| `--blend` | Style blend ratio | 0.95 |
| `--fps` | Video framerate | 24 |
| `--morph_seconds` | Seconds per morph transition | 2.0 |
| `--hold_frames` | Frames to hold on final image | 24 |
| `--zoom` | Static zoom factor | 1.5 |
| `--pan_zoom` | Ken Burns zoom level (e.g., 2.0) | None |
| `--pan_direction` | Pan direction: horizontal, vertical, diagonal, diagonal_reverse | horizontal |
| `--pytorch_style` | Pre-style with PyTorch neural style | - |
| `--pytorch_model` | Specific PyTorch model (candy, mosaic, rain_princess, udnie) | random |
| `--force` | Regenerate existing outputs | - |

### Output Structure

```
output/morphv2/<image_name>/
├── styled/                    # Styled images
│   ├── <name>_tile128_overlap16.jpg
│   ├── <name>_tile160_overlap20.jpg
│   ├── ...
│   └── <name>_tile512_overlap64.jpg
├── work/                      # Intermediate files
│   ├── mask.png              # Semantic mask
│   ├── style_crop.jpg        # Extracted region
│   └── style_crop_pytorch.jpg # PyTorch styled (if enabled)
├── style_source.jpg          # Copy of style source
└── <name>_morph.mp4          # Final morph video
```

### Region Scoring

Auto mode scores regions based on:
- **Coverage**: Sweet spot 5-40% of image (not too small, not too large)
- **Aspect ratio**: Prefers square-ish regions
- **Position**: Slight preference for centered regions
- **Semantic preference**: person, animals > vehicles > furniture

### Available Labels

```
background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair,
cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep,
sofa, train, tvmonitor
```

Use `--list_labels` to see all available semantic labels with their IDs.

## Multi-Model Video Pipeline

Full pipeline for styling videos with multiple models (udnie, mosaic, tenharmsel) blended together with Gaussian pulses for a dynamic, evolving style effect.

### Pipeline Overview

1. **Extract frames** from video at 8fps
2. **Style frames** with multiple models and weight variants
3. **Create walk files** for weight interpolation
4. **Compose video** with crossfades and Gaussian pulsing between styles

### Quick Start

```bash
# Full pipeline: extract + style + compose
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/style_video_pipeline.py \
  --video /project/input.mp4 \
  --output_dir /project/output \
  --output_name my_video \
  --styled_start 17 --styled_end 160

# Then compose the video
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/multi_model_video.py \
  --orig_dir /project/output/my_video/frames \
  --styled_udnie /project/output/my_video/styled_udnie \
  --styled_mosaic /project/output/my_video/styled_mosaic \
  --styled_tenharmsel /project/output/my_video/styled_tenharmsel \
  --output /project/output/my_video.mp4 \
  --styled_start 17 --styled_end 160
```

### Scripts

#### `scripts/style_video_pipeline.py`

Full styling pipeline - extracts frames and styles with all models.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--video` | Input video path | required |
| `--output_dir` | Output directory | required |
| `--output_name` | Name for output folder/video | required |
| `--styled_start` | First frame to style | 17 (skip 2s) |
| `--styled_end` | Last frame to style | 160 (20s) |
| `--fps_extract` | Frame extraction rate | 8 |
| `--size` | Output resolution | 1080 |
| `--no_tenharmsel` | Skip tenharmsel styling | false |

**Frame math at 8fps:**
- 2 seconds = 16 frames (original intro)
- 20 seconds = 160 frames
- `--styled_start 17` skips 2s for fade-in from original

#### `scripts/multi_model_video.py`

Composes styled frames into video with crossfades and Gaussian pulsing.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--orig_dir` | Directory with original frames | required |
| `--styled_udnie` | Udnie styled frames | required |
| `--styled_mosaic` | Mosaic styled frames | optional |
| `--styled_tenharmsel` | Tenharmsel styled frames | optional |
| `--output` | Output video path | required |
| `--styled_start` | First styled frame | 17 |
| `--styled_end` | Last styled frame | 160 |
| `--fps` | Output framerate | 24 |
| `--size` | Output resolution | 1080 |
| `--orig_blend` | Original image blend (0-1) | 0.4 |
| `--crossfade_seconds` | Fade duration | 2.0 |
| `--hold_frames` | Frames per source (8fps→24fps = 3) | 3 |

**Blending behavior:**
- Udnie is the base layer with weight ramping
- Mosaic pulses in/out with 4 Gaussian pulses (max 50% blend)
- Tenharmsel pulses in/out offset by 1/8 cycle (max 50% blend)
- Creates dynamic interweaving of all three styles

#### `scripts/style_all_weights.py`

Batch style frames with all weight variants for a single model.

```bash
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/style_all_weights.py \
  --style udnie_strong \
  --frame_start 17 --frame_end 160 \
  --input_dir /project/frames \
  --output_dir /project/styled_udnie
```

**Available style presets:**
| Style | Weights |
|-------|---------|
| `udnie` | style1e10 - style9e10 (9 weights) |
| `udnie_strong` | style5e10 - style9e10 (5 weights, stronger effect) |
| `candy` | style1e10 - style9e10 (9 weights) |
| `mosaic` | style5e10 only (1 weight) |
| `tenharmsel` | style1e10 - style9e10 (9 weights) |
| `tenharmsel_strong` | style5e10 - style9e10 (5 weights) |

### Weight Ladder

Models have weight variants from `style1e10` (subtle) to `style9e10` (strong):
- `style1e10` - Very subtle, mostly original
- `style5e10` - Medium strength
- `style9e10` - Full style effect

The video compositor ramps through weights over time for evolving intensity.

### Output Structure

```
output/my_video/
├── frames/                    # Extracted original frames
│   ├── frame_0001_original.jpg
│   ├── frame_0002_original.jpg
│   └── ...
├── styled_udnie/              # Udnie styled frames
│   ├── frame_0017_original.jpg
│   ├── frame_0017_udnie_style5e10.jpg
│   ├── frame_0017_udnie_style6e10.jpg
│   └── ...
├── styled_mosaic/             # Mosaic styled frames
│   └── ...
├── styled_tenharmsel/         # Tenharmsel styled frames
│   └── ...
└── walk_*.json                # Weight walk files
```

### Walk Files

Walk files control weight interpolation over time:
```json
{
  "walk": [0, 0, 1, 1, 2, 2, 3, ...],
  "weights": ["udnie_style5e10", "udnie_style6e10", ...],
  "frame_start": 17,
  "frame_end": 160
}
```

### Example: Style 20s of a Portrait Video

```bash
# Mount your project directory
docker compose run --rm -v /home/user/my_project:/project style bash

# Inside container:
# 1. Run full pipeline
python3 /app/scripts/style_video_pipeline.py \
  --video /project/videos/sunset.mp4 \
  --output_dir /project/output \
  --output_name sunset_styled \
  --styled_start 17 --styled_end 160

# 2. Compose video
python3 /app/scripts/multi_model_video.py \
  --orig_dir /project/output/sunset_styled/frames \
  --styled_udnie /project/output/sunset_styled/styled_udnie \
  --styled_mosaic /project/output/sunset_styled/styled_mosaic \
  --styled_tenharmsel /project/output/sunset_styled/styled_tenharmsel \
  --output /project/output/sunset_styled.mp4 \
  --styled_start 17 --styled_end 160 \
  --orig_blend 0.4 --crossfade_seconds 2.0
```

### Tips

- **Skip tenharmsel** for faster processing: `--no_tenharmsel`
- **Adjust blend**: Higher `--orig_blend` keeps more original detail (0.4 = 40% original)
- **Longer crossfades**: Increase `--crossfade_seconds` for smoother transitions
- **Portrait videos**: Script auto-detects and handles 9:16 aspect ratio
- **Resume**: Existing styled frames are skipped, so you can resume interrupted runs

## Style Morph - Cross-Image Weight Flow

`scripts/style_morph.py` - Creates smooth morph videos from pre-styled images with flowing weight transitions across multiple style families.

### Overview

Given a directory of pre-styled images (with weight variants), Style Morph:
1. Loads all available weight ladder images for each source image
2. Interpolates smoothly through weight ladders using slow sine waves
3. Blends multiple style families (candy, udnie, mosaic, rain_princess, tenharmsel) simultaneously
4. Creates crossfade transitions between source images
5. Applies subtle color filters for visual polish

### Quick Start

```bash
# Basic usage - square output
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/style_morph.py \
  --styled_dir /project/styled_images \
  --output /project/output/morph.mp4

# Portrait mode (keeps aspect ratio)
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/style_morph.py \
  --styled_dir /project/styled_images \
  --output /project/output/morph.mp4 \
  --portrait

# Specific style families only
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/style_morph.py \
  --styled_dir /project/styled_images \
  --output /project/output/morph.mp4 \
  --families tenharmsel udnie

# Higher original blend for more detail
docker compose run --rm -v /path/to/project:/project style \
  python3 /app/scripts/style_morph.py \
  --styled_dir /project/styled_images \
  --output /project/output/morph.mp4 \
  --orig_blend 0.15
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--styled_dir` | Directory with pre-styled images | required |
| `--output` | Output video path | required |
| `--frame_time` | Seconds per source image | 4.0 |
| `--size` | Output resolution | 1080 |
| `--fps` | Output framerate | 24 |
| `--no_skip_first` | Include first image (normally skipped) | skip first |
| `--portrait` | Portrait mode (maintain aspect ratio) | square crop |
| `--orig_blend` | Original image blend amount (0-1) | 0.08 |
| `--families` | Style families to use | all 5 |

### Weight Ladders

Each style family has a weight progression from subtle to intense:

| Family | Weights |
|--------|---------|
| candy | candy → candy_style1e9 → ... → candy_style1e12 |
| udnie | udnie → udnie_style1e9 → ... → udnie_style1e12 |
| mosaic | mosaic → mosaic_style1e9 → ... → mosaic_style1e12 |
| rain_princess | rain_princess → rain_princess_style1e9 → ... → rain_princess_style1e12 |
| tenharmsel | tenharmsel_style1e9 → 2e9 → ... → 9e9 → 1e10 → ... → 1e12 (28 weights) |

### Input Requirements

Styled images directory should contain:
```
styled_dir/
├── IMG_001_original.jpg
├── IMG_001_candy.jpg
├── IMG_001_candy_style1e9.jpg
├── IMG_001_candy_style5e9.jpg
├── ...
├── IMG_001_tenharmsel_style1e9.jpg
├── ...
├── IMG_002_original.jpg
├── IMG_002_candy.jpg
└── ...
```

Use `lib/style_images.py` from NeuralStyleGoats or the batch styling scripts to generate these.

### How It Works

1. **Weight Trajectories**: Each style family's ladder position (0-1) drifts slowly using sine waves with different frequencies/phases
2. **Style Blending**: All active styles are blended together with weights that also drift over time
3. **Interpolation**: Within each ladder, adjacent weights are smoothstep-blended based on position
4. **Crossfades**: 1.5-second crossfades between source images
5. **Filters**: Random subtle filters (saturation, vibrance, warmth) applied per image

### Example: Full Workflow

```bash
# 1. First, style your source images with all weights
# (Using NeuralStyleGoats or batch scripts)

# 2. Then create the morph video
docker compose run --rm -v /home/user/project:/project style \
  python3 /app/scripts/style_morph.py \
  --styled_dir /project/output/tavian_full \
  --output /project/output/videos/tavian_morph.mp4 \
  --frame_time 4.0 \
  --orig_blend 0.08 \
  --families candy udnie mosaic tenharmsel

# Output: ~1 minute video with smooth flowing style transitions
```

### Tips

- **More original detail**: Increase `--orig_blend` to 0.15-0.25
- **Single family**: Use `--families tenharmsel` for cleaner look
- **Longer transitions**: Increase `--frame_time` to 6.0 or 8.0
- **Portrait photos**: Use `--portrait` to avoid square cropping
- **Skip problematic first image**: First image is skipped by default (often different lighting)
