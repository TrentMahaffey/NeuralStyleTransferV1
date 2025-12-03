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
