# Neural Style Transfer Recipes

Proven configurations for generating styled media. Each recipe includes:
- **recipe.json** - Machine-readable parameters (copy these to reproduce)
- **README.md** - Human-readable documentation with usage examples
- **examples/** - Sample outputs demonstrating the recipe

## Available Recipes

| Recipe | Description | Input | Output |
|--------|-------------|-------|--------|
| [blob_face_morph](blob_face_morph/) | Face detection + blob-blended style morphing | Single image | Vertical video |
| [full_weight_ladder](full_weight_ladder/) | All 69 style weights across 5 model families | Image folder | Styled images |

## Quick Start

1. Pick a recipe folder
2. Read the README.md for usage
3. Copy the command from recipe.json, replacing input/output paths
4. Run via docker-compose

## Adding New Recipes

When you create something worth saving:

```bash
# Scripts now auto-generate run logs
# Copy the *_run.json to recipes/<name>/recipe.json
# Add a README.md explaining the effect
# Symlink or copy example outputs
```
