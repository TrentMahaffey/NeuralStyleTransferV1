# Blob Face Morph Recipe

Detects faces in an image, applies multiple neural style transfers, and creates a video that zooms out from each face while organically morphing between styles using gaussian blob blending.

## Effect Description

- Starts zoomed in on detected face
- Multiple blob regions show different styled versions simultaneously
- Blobs animate and flow organically with soft gaussian boundaries
- Camera slowly zooms out while styles morph
- Ends with blend back to original image
- Multi-face images: crossfades between faces

## Quick Start

```bash
docker-compose run --rm style bash -lc "python /app/scripts/morph_faces.py \
  --image /app/input/YOUR_IMAGE.jpg \
  --blob \
  --vertical"
```

## Example Output

Created from: `input/PhotoOfMe.jpg`
Output: `examples/PhotoOfMe_faces_blob.mp4`

## Parameters

### Essential
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image` | required | Input image path |
| `--blob` | flag | Enable blob mode (required for this recipe) |
| `--vertical` | flag | Vertical video (720x1280) |

### Blob Animation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_blobs` | 4 | Number of blob regions (4-8 recommended) |
| `--blob_frequency` | 2.5 | Detail level (higher = more complex shapes) |
| `--blob_speed` | 1.0 | Animation speed (0.5 = slower, 2.0 = faster) |
| `--blob_feather` | 0.3 | Boundary softness (higher = softer edges) |

### Zoom & Timing
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min_zoom` | 1.0 | Final zoom level (1.0 = full frame) |
| `--max_zoom` | 4.0 | Starting zoom level |
| `--morph_time` | 0.5 | Seconds per style transition |
| `--fps` | 24 | Output framerate |

### Styling
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scale` | 1440 | Styled image resolution |
| `--blend` | 0.95 | Style intensity (0-1) |
| `--face_padding` | 0.6 | Extra padding around face crop |

## Variations

### Dreamy Slow Morph
```bash
docker-compose run --rm style bash -lc "python /app/scripts/morph_faces.py \
  --image /app/input/photo.jpg \
  --blob --vertical \
  --blob_speed 0.3 \
  --morph_time 1.5 \
  --blob_feather 0.5"
```

### Intense Fast Animation
```bash
docker-compose run --rm style bash -lc "python /app/scripts/morph_faces.py \
  --image /app/input/photo.jpg \
  --blob --vertical \
  --num_blobs 6 \
  --blob_speed 2.0 \
  --blob_frequency 4.0"
```

### Dramatic Zoom Range
```bash
docker-compose run --rm style bash -lc "python /app/scripts/morph_faces.py \
  --image /app/input/photo.jpg \
  --blob --vertical \
  --min_zoom 0.8 \
  --max_zoom 8.0"
```

## Tips

- Face must be at least 3% of image area to be detected
- Well-lit, front-facing portraits work best
- Higher `--blob_feather` creates softer, more organic transitions
- Lower `--blob_speed` for meditative, slow-moving effect
- Multiple faces are processed sequentially with crossfades
