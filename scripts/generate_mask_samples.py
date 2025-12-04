#!/usr/bin/env python3
"""
Generate DeepLab semantic mask sample previews for the web UI.

Creates before/after/mask comparison images showing how semantic masking works
for different object classes, with various styles and invert options.

Usage:
    docker-compose run --rm web bash -lc "python /web/generate_mask_samples.py"
    docker-compose run --rm web bash -lc "python /web/generate_mask_samples.py --force"
    docker-compose run --rm web bash -lc "python /web/generate_mask_samples.py --list"

Input photos should be placed in: input/mask_samples/
Naming convention: <class>_<description>.jpg (e.g., person_beach.jpg, car_street.jpg)

Output saved to: static/mask_samples/
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from PIL import Image
import argparse

# Paths
PIPELINE_DIR = Path(os.environ.get("PIPELINE_DIR", "/app"))
WEB_DIR = Path(os.environ.get("WEB_DIR", "/web"))

PIPELINE = PIPELINE_DIR / "pipeline.py"
SKY_SWAP = PIPELINE_DIR / "sky_swap.py"
DEEPLAB_WEIGHTS = PIPELINE_DIR / "models" / "deeplab" / "deeplab-resnet.pth.tar"
WORK_DIR = PIPELINE_DIR / "_work" / "mask_samples"

# Input/Output directories
INPUT_PHOTOS_DIR = PIPELINE_DIR / "input" / "mask_samples"
OUTPUT_DIR = WEB_DIR / "static" / "mask_samples"

# VOC21 class labels
VOC21_CLASSES = {
    "background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
    "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
    "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
    "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
    "sofa": 18, "train": 19, "tvmonitor": 20,
}

# Sample configurations - each defines a mask sample to generate
# Based on available input photos with multiple objects each
PRESETS_DIR = WEB_DIR / "presets"

MASK_SAMPLES = [
    # bike-dog-person.png - bicycle(2), dog(12), person(15)
    {
        "input_prefix": "bike-dog-person",
        "target_ids": "15",
        "invert": False,
        "style": {"model": "/app/models/pytorch/candy.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "person_styled",
        "description": "Style the person only",
        "preset_name": "Mask: Style Person",
        "tags": ["person", "semantic", "mask"]
    },
    {
        "input_prefix": "bike-dog-person",
        "target_ids": "12",
        "invert": True,
        "style": {"model": "/app/models/pytorch/mosaic.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "dog_protected",
        "description": "Keep dog original, style everything else",
        "preset_name": "Mask: Protect Dog",
        "tags": ["dog", "invert", "mosaic", "mask"]
    },
    {
        "input_prefix": "bike-dog-person",
        "target_ids": "2",
        "invert": False,
        "style": {"model": "/app/models/pytorch/udnie.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "bicycle_styled",
        "description": "Style the bicycle only",
        "preset_name": "Mask: Style Bicycle",
        "tags": ["bicycle", "udnie", "mask"]
    },

    # cat-sheep-plant.png - cat(8), sheep(17), pottedplant(16)
    {
        "input_prefix": "cat-sheep-plant",
        "target_ids": "8",
        "invert": False,
        "style": {"model": "/app/models/pytorch/rain_princess.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "cat_styled",
        "description": "Style the cat only",
        "preset_name": "Mask: Style Cat",
        "tags": ["cat", "rain_princess", "mask"]
    },
    {
        "input_prefix": "cat-sheep-plant",
        "target_ids": "17",
        "invert": True,
        "style": {"model": "/app/models/torch/starry_night_eccv16.t7", "type": "torch7", "io": "caffe_bgr"},
        "name": "sheep_protected",
        "description": "Keep sheep original, style background",
        "preset_name": "Mask: Protect Sheep",
        "tags": ["sheep", "invert", "starry_night", "mask"]
    },
    {
        "input_prefix": "cat-sheep-plant",
        "target_ids": "8,17",
        "invert": False,
        "style": {"model": "/app/models/torch/la_muse_eccv16.t7", "type": "torch7", "io": "caffe_bgr"},
        "name": "animals_styled",
        "description": "Style cat and sheep together",
        "preset_name": "Mask: Style Animals",
        "tags": ["cat", "sheep", "animals", "la_muse", "mask"]
    },

    # chair-tv-train.png - chair(9), tvmonitor(20), train(19)
    {
        "input_prefix": "chair-tv-train",
        "target_ids": "20",
        "invert": False,
        "style": {"model": "/app/models/pytorch/candy.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "tv_styled",
        "description": "Style the TV monitor only",
        "preset_name": "Mask: Style TV",
        "tags": ["tvmonitor", "candy", "mask"]
    },
    {
        "input_prefix": "chair-tv-train",
        "target_ids": "19",
        "invert": True,
        "style": {"model": "/app/models/torch/composition_vii_eccv16.t7", "type": "torch7", "io": "caffe_bgr"},
        "name": "train_protected",
        "description": "Keep train original, artistic background",
        "preset_name": "Mask: Protect Train",
        "tags": ["train", "invert", "composition_vii", "mask"]
    },

    # cow-cat-chair.png - cow(10), cat(8), chair(9)
    {
        "input_prefix": "cow-cat-chair",
        "target_ids": "10",
        "invert": False,
        "style": {"model": "/app/models/pytorch/mosaic.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "cow_styled",
        "description": "Style the cow only",
        "preset_name": "Mask: Style Cow",
        "tags": ["cow", "mosaic", "mask"]
    },
    {
        "input_prefix": "cow-cat-chair",
        "target_ids": "10,8",
        "invert": True,
        "style": {"model": "/app/models/pytorch/udnie.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "animals_protected",
        "description": "Keep cow and cat original, style rest",
        "preset_name": "Mask: Protect Cow & Cat",
        "tags": ["cow", "cat", "animals", "invert", "udnie", "mask"]
    },

    # horse-boat-person.png - horse(13), boat(4), person(15)
    {
        "input_prefix": "horse-boat-person",
        "target_ids": "13",
        "invert": False,
        "style": {"model": "/app/models/torch/the_wave_eccv16.t7", "type": "torch7", "io": "caffe_bgr"},
        "name": "horse_styled",
        "description": "Style the horse only",
        "preset_name": "Mask: Style Horse",
        "tags": ["horse", "the_wave", "mask"]
    },
    {
        "input_prefix": "horse-boat-person",
        "target_ids": "4",
        "invert": False,
        "style": {"model": "/app/models/pytorch/rain_princess.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "boat_styled",
        "description": "Style the boat only",
        "preset_name": "Mask: Style Boat",
        "tags": ["boat", "rain_princess", "mask"]
    },
    {
        "input_prefix": "horse-boat-person",
        "target_ids": "15",
        "invert": True,
        "style": {"model": "/app/models/torch/starry_night_eccv16.t7", "type": "torch7", "io": "caffe_bgr"},
        "name": "person_protected",
        "description": "Keep person realistic, style surroundings",
        "preset_name": "Mask: Protect Person",
        "tags": ["person", "invert", "starry_night", "mask"]
    },

    # plant-dog-table.png - pottedplant(16), dog(12), diningtable(11)
    {
        "input_prefix": "plant-dog-table",
        "target_ids": "12",
        "invert": False,
        "style": {"model": "/app/models/pytorch/candy.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "dog_styled",
        "description": "Style the dog only",
        "preset_name": "Mask: Style Dog",
        "tags": ["dog", "candy", "mask"]
    },
    {
        "input_prefix": "plant-dog-table",
        "target_ids": "16",
        "invert": False,
        "style": {"model": "/app/models/torch/la_muse_eccv16.t7", "type": "torch7", "io": "caffe_bgr"},
        "name": "plant_styled",
        "description": "Style the potted plant only",
        "preset_name": "Mask: Style Plant",
        "tags": ["pottedplant", "la_muse", "mask"]
    },
    {
        "input_prefix": "plant-dog-table",
        "target_ids": "11",
        "invert": True,
        "style": {"model": "/app/models/pytorch/mosaic.pth", "type": "transformer", "io": "imagenet_255"},
        "name": "table_protected",
        "description": "Keep table original, style everything else",
        "preset_name": "Mask: Protect Table",
        "tags": ["diningtable", "invert", "mosaic", "mask"]
    },
]


def find_input_photo(prefix: str) -> Path | None:
    """Find an input photo matching the given prefix."""
    if not INPUT_PHOTOS_DIR.exists():
        return None

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        matches = list(INPUT_PHOTOS_DIR.glob(f"{prefix}*{ext[1:]}"))
        if matches:
            return matches[0]
    return None


def generate_deeplab_mask(input_image: Path, target_ids: str, output_mask: Path,
                          resolution: int = 1024, feather_pct: float = 0.5) -> bool:
    """Generate a DeepLab semantic mask."""
    if not DEEPLAB_WEIGHTS.exists():
        print(f"  [SKIP] DeepLab weights not found: {DEEPLAB_WEIGHTS}")
        return False

    cmd = [
        "python3", str(SKY_SWAP),
        "--image", str(input_image),
        "--weights", str(DEEPLAB_WEIGHTS),
        "--backbone", "resnet",
        "--resolution", str(resolution),
        "--target_ids", target_ids,
        "--mask_feather_pct", str(feather_pct),
        "--out_mask", str(output_mask)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_mask.exists()
    except Exception as e:
        print(f"  [ERROR] DeepLab failed: {e}")
        return False


def apply_style(input_image: Path, output_image: Path, mask: Path,
                style_config: dict, invert: bool, work_dir: Path) -> bool:
    """Apply style to image with mask."""
    cmd = [
        "python3", str(PIPELINE),
        "--input_image", str(input_image),
        "--output_image", str(output_image),
        "--work_dir", str(work_dir),
        "--mask", str(mask),
        "--fit_mask_to", "input",
        "--composite_mode", "keep",
        "--mask_autofix",
        "--inference_res", "720",
        "--scale", "720",
    ]

    if invert:
        cmd.append("--mask_invert")

    # Add style-specific options
    if style_config.get("type") == "magenta":
        cmd += ["--model_type", "magenta", "--io_preset", "tanh"]
        if style_config.get("style_image"):
            cmd += ["--magenta_style", style_config["style_image"]]
    else:
        cmd += [
            "--model", style_config["model"],
            "--model_type", style_config["type"],
            "--io_preset", style_config["io"]
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0 and output_image.exists()
    except Exception as e:
        print(f"  [ERROR] Pipeline failed: {e}")
        return False


def create_preset_json(config: dict, styled_image: str) -> bool:
    """Create a preset JSON file for the mask sample."""
    name = config["name"]
    preset_name = config.get("preset_name", f"Mask: {name}")

    # Determine model_type from style config
    style = config["style"]
    if style["type"] == "transformer":
        model_type = "transformer"
    elif style["type"] == "torch7":
        model_type = "torch7"
    else:
        model_type = style["type"]

    # Parse target_ids into list of integers
    target_ids_str = config["target_ids"]
    target_ids = [int(x.strip()) for x in target_ids_str.split(",")]

    preset = {
        "name": preset_name,
        "description": config["description"],
        "category": "Semantic Masking",
        "tags": config.get("tags", ["mask", "semantic"]),
        "sample_image": f"mask_samples/{styled_image}",
        "params": {
            "model": style["model"],
            "model_type": model_type,
            "io_preset": style["io"],
            "blend": 1.0,
            "deeplab_target_ids": target_ids
        }
    }

    # Add invert if True
    if config.get("invert", False):
        preset["params"]["deeplab_invert"] = True

    # Create filename from name
    preset_filename = f"mask_{name}.json"
    preset_path = PRESETS_DIR / preset_filename

    try:
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)
        print(f"  [OK] Preset saved: {preset_filename}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create preset: {e}")
        return False


def create_comparison_image(original: Path, mask: Path, styled: Path,
                            output: Path, invert: bool = False):
    """Create a side-by-side comparison: original | mask | styled."""
    try:
        orig_img = Image.open(original).convert('RGB')
        mask_img = Image.open(mask).convert('L')
        styled_img = Image.open(styled).convert('RGB')

        # Resize all to match
        target_size = (400, 300)
        orig_img = orig_img.resize(target_size, Image.LANCZOS)
        mask_img = mask_img.resize(target_size, Image.LANCZOS)
        styled_img = styled_img.resize(target_size, Image.LANCZOS)

        # If inverted, show the inverted mask for clarity
        if invert:
            from PIL import ImageOps
            mask_img = ImageOps.invert(mask_img)

        # Convert mask to RGB for display
        mask_rgb = Image.merge('RGB', (mask_img, mask_img, mask_img))

        # Create combined image with labels area
        combined = Image.new('RGB', (target_size[0] * 3 + 20, target_size[1]), (26, 26, 46))
        combined.paste(orig_img, (0, 0))
        combined.paste(mask_rgb, (target_size[0] + 10, 0))
        combined.paste(styled_img, (target_size[0] * 2 + 20, 0))

        combined.save(output, 'JPEG', quality=90)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create comparison: {e}")
        return False


def generate_sample(config: dict, force: bool = False) -> bool:
    """Generate a single mask sample."""
    name = config["name"]
    print(f"\n[{name}] {config['description']}")

    # Check if output already exists
    comparison_output = OUTPUT_DIR / f"{name}_comparison.jpg"
    styled_output_file = f"{name}_styled.jpg"
    if comparison_output.exists() and not force:
        # Still create preset JSON for existing samples (use styled image, not comparison)
        create_preset_json(config, styled_output_file)
        print(f"  [SKIP] Already exists (preset updated)")
        return True

    # Find input photo
    input_photo = find_input_photo(config["input_prefix"])
    if not input_photo:
        print(f"  [SKIP] No input photo found with prefix: {config['input_prefix']}")
        print(f"         Add photos to: {INPUT_PHOTOS_DIR}/")
        return False

    print(f"  Input: {input_photo.name}")

    # Create work directory
    sample_work = WORK_DIR / name
    sample_work.mkdir(parents=True, exist_ok=True)

    # Generate mask
    mask_path = sample_work / "mask.png"
    print(f"  Generating mask for classes: {config['target_ids']}...")
    if not generate_deeplab_mask(input_photo, config["target_ids"], mask_path):
        print(f"  [FAIL] Could not generate mask")
        return False

    # Apply style
    styled_path = sample_work / "styled.jpg"
    invert = config.get("invert", False)
    print(f"  Applying style (invert={invert})...")
    if not apply_style(input_photo, styled_path, mask_path, config["style"], invert, sample_work):
        print(f"  [FAIL] Could not apply style")
        return False

    # Create comparison image
    print(f"  Creating comparison...")
    if not create_comparison_image(input_photo, mask_path, styled_path, comparison_output, invert):
        print(f"  [FAIL] Could not create comparison")
        return False

    # Also save individual outputs
    styled_output = OUTPUT_DIR / f"{name}_styled.jpg"
    mask_output = OUTPUT_DIR / f"{name}_mask.png"

    Image.open(styled_path).save(styled_output, 'JPEG', quality=90)
    Image.open(mask_path).save(mask_output, 'PNG')

    # Create preset JSON file for this mask sample (use styled image, not comparison)
    create_preset_json(config, f"{name}_styled.jpg")

    print(f"  [OK] Saved to {comparison_output.name}")
    return True


def save_metadata():
    """Save metadata about generated samples."""
    metadata = {
        "samples": [],
        "classes": VOC21_CLASSES
    }

    for config in MASK_SAMPLES:
        name = config["name"]
        comparison = OUTPUT_DIR / f"{name}_comparison.jpg"
        if comparison.exists():
            metadata["samples"].append({
                "name": name,
                "description": config["description"],
                "target_ids": config["target_ids"],
                "invert": config.get("invert", False),
                "comparison": f"{name}_comparison.jpg",
                "styled": f"{name}_styled.jpg",
                "mask": f"{name}_mask.png"
            })

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")


def list_expected_inputs():
    """List expected input photos."""
    print("\n" + "="*60)
    print("EXPECTED INPUT PHOTOS")
    print("="*60)
    print(f"\nPlace photos in: {INPUT_PHOTOS_DIR}/\n")

    prefixes = set(config["input_prefix"] for config in MASK_SAMPLES)

    for prefix in sorted(prefixes):
        samples = [c for c in MASK_SAMPLES if c["input_prefix"] == prefix]
        print(f"  {prefix}_*.jpg")
        for s in samples:
            classes = s["target_ids"]
            class_names = [k for k, v in VOC21_CLASSES.items() if str(v) in classes.split(",")]
            print(f"    -> {s['name']}: {', '.join(class_names)} ({'inverted' if s.get('invert') else 'masked'})")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate DeepLab mask samples")
    parser.add_argument("--force", "-f", action="store_true", help="Force regenerate all")
    parser.add_argument("--list", "-l", action="store_true", help="List expected input photos")
    args = parser.parse_args()

    if args.list:
        list_expected_inputs()
        return

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GENERATING DEEPLAB MASK SAMPLES")
    print("="*60)
    print(f"Input photos: {INPUT_PHOTOS_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    # Check for input photos
    if not any(INPUT_PHOTOS_DIR.glob("*.*")):
        print(f"\n[WARNING] No input photos found!")
        list_expected_inputs()
        return

    success = 0
    failed = 0
    skipped = 0

    for config in MASK_SAMPLES:
        result = generate_sample(config, force=args.force)
        if result:
            success += 1
        elif find_input_photo(config["input_prefix"]):
            failed += 1
        else:
            skipped += 1

    save_metadata()

    print("\n" + "="*60)
    print(f"COMPLETE: {success} generated, {failed} failed, {skipped} skipped (no input)")
    print("="*60)


if __name__ == "__main__":
    main()
