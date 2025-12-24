# Full Weight Ladder Recipe

Applies all 69 style weight variants across 5 model families to a folder of images. Creates a complete progression from subtle to intense styling for each model family.

## Effect Description

For each input image, generates 70 outputs:
- 1 original copy
- 69 styled versions across 5 model families at various weight intensities

Weight progression: `1e9` (barely visible) -> `5e9` -> `1e10` -> `5e10` -> `1e11` -> `5e11` -> `1e12` (maximum intensity)

## Quick Start

```bash
python3 /home/trent-mahaffey/Dev/NeuralStyleGoats/lib/style_images.py \
  --input_dir /path/to/your/images \
  --output_dir /path/to/styled/output
```

## Example Output

Created from: `TavianBlueSlide/` (25 images)
Output: `output/tavian_full/` (1,750 styled images)

## Model Families

| Family | Weights | Description |
|--------|---------|-------------|
| candy | 8 | Bright, colorful candy-like effect |
| mosaic | 8 | Cubist mosaic patterns |
| udnie | 8 | Abstract expressionist style |
| rain_princess | 8 | Soft, impressionist rain effect |
| tenharmsel | 37 | Fine-grained artistic style (most weights) |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | TavianBlueSlide | Directory with .jpg/.JPG images |
| `--output_dir` | output/tavian_full | Output directory |
| `--scale` | 1080 | Output resolution |

Fixed parameters:
- `inference_res`: 720
- `blend`: 0.9
- `io_preset`: auto
- `device`: cuda

## Output Naming

```
<image_stem>_original.jpg
<image_stem>_candy.jpg
<image_stem>_candy_style1e9.jpg
<image_stem>_candy_style5e9.jpg
<image_stem>_candy_style1e10.jpg
...
<image_stem>_tenharmsel_style9e11.jpg
<image_stem>_tenharmsel_style1e12.jpg
```

## Weight Intensity Guide

| Weight | Intensity | Use Case |
|--------|-----------|----------|
| base (no suffix) | ~50% | Standard style transfer |
| style1e9 | ~10% | Subtle hint of style |
| style5e9 | ~30% | Light styling |
| style1e10 | ~50% | Medium styling |
| style5e10 | ~70% | Strong styling |
| style1e11 | ~85% | Very strong styling |
| style5e11 | ~95% | Near maximum |
| style1e12 | 100% | Maximum intensity |

## Use Cases

### Creating Weight Morph Videos

Use the full weight ladder with `style_morph.py`:
```bash
python3 scripts/style_morph.py \
  --styled_dir /path/to/styled/output \
  --output /path/to/morph.mp4
```

### Multi-Model Video Pipeline

Use with `multi_model_video.py` for Gaussian-pulsed style blending:
```bash
python3 scripts/multi_model_video.py \
  --styled_udnie /path/to/styled_udnie \
  --styled_mosaic /path/to/styled_mosaic \
  --output video.mp4
```

## Tips

- ~5 minutes per image on GPU
- Resumable: skips existing outputs
- 25 images = ~2 hours processing time
- Creates `style_images_run.json` log file
- Use lower `--scale` (720) for faster testing
