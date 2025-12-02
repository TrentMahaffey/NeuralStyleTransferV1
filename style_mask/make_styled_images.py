
import os
import glob
import subprocess
import shutil
import argparse
import sys

# Configuration
INPUT_DIR = "/app/input"
WORK_DIR = "/app/_work"
OUTPUT_DIR = "/app/output"
MAGENTA_STYLES_DIR = "/app/models/magenta_styles"
DEEPLAB_WEIGHTS = "/app/models/deeplab/deeplab-resnet.pth.tar"

def run_command(cmd, check=True):
    """Run a shell command and return its output."""
    print(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result.stdout, result.stderr

def check_tools():
    """Check if required tools are available."""
    for tool in ["ffmpeg", "python"]:
        try:
            run_command([tool, "--version"])
        except subprocess.CalledProcessError:
            print(f"[error] {tool} not found")
            sys.exit(1)
    if not os.path.exists("/app/pipeline.py"):
        print("[error] /app/pipeline.py missing")
        sys.exit(1)
    if not os.path.exists("/app/sky_swap.py"):
        print("[error] /app/sky_swap.py missing")
        sys.exit(1)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Style and mask an image using Magenta, Torch7, and PyTorch models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-image",
        default="winter1.jpeg",
        help="Image to style. If a bare filename, resolved as /app/input/<name>.\n"
             "You may also pass an absolute path like /app/input/foo.jpeg."
    )
    parser.add_argument(
        "-t", "--target-ids",
        default="15",
        help="Comma-separated DeepLab class IDs to mask (default: 15 for person).\n"
             "Common: 15=person, 0=background."
    )
    args = parser.parse_args()
    # Resolve input image path
    if not args.input_image.startswith("/"):
        args.input_image = os.path.join(INPUT_DIR, args.input_image)
    return args.input_image, args.target_ids

def main():
    # Check tools
    check_tools()

    # Parse arguments
    input_image, target_ids = parse_args()
    print(f"[info] Using input image: {input_image}, target IDs: {target_ids}")

    # Setup directories
    stem = os.path.splitext(os.path.basename(input_image))[0]
    work = os.path.join(WORK_DIR, stem)
    slides = os.path.join(work, "slides")
    os.makedirs(work, exist_ok=True)
    os.makedirs(slides, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 0: Bake input image (remove EXIF, fix orientation)
    baked_image = os.path.join(work, f"{stem}_baked.png")
    print(f"[bake] Baking {input_image} -> {baked_image}")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_image,
        "-vf", "scale=iw:ih:flags=lanczos,setsar=1",
        "-pix_fmt", "rgb24",
        baked_image
    ]
    run_command(cmd)

    # Step 1: Generate DeepLab mask
    mask_output = os.path.join(work, f"{stem}_person_mask.png")
    print(f"[deeplab] Generating mask -> {mask_output}")
    cmd = [
        "python", "/app/sky_swap.py",
        "--image", baked_image,
        "--weights", DEEPLAB_WEIGHTS,
        "--backbone", "resnet",
        "--resolution", "1536",
        "--mask_expand_pct", "0.5",
        "--mask_contract_pct", "0.5",
        "--mask_feather_pct", "0.0",
        "--target_ids", target_ids,
        "--morph_close_ks", "5",
        "--out_mask", mask_output
    ]
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[error] DeepLab mask generation failed: {e.stderr}")
        sys.exit(1)

    # Step 2: Magenta styles
    for style_path in glob.glob(os.path.join(MAGENTA_STYLES_DIR, "*.jpg")) + glob.glob(os.path.join(MAGENTA_STYLES_DIR, "*.png")):
        if not os.path.isfile(style_path):
            continue
        style_name = os.path.splitext(os.path.basename(style_path))[0].lower().replace("[^a-z0-9]+", "_")
        print(f"[magenta] Styling with {style_path}, tag={style_name}")

        # FG (inside mask)
        fg_output = os.path.join(slides, f"{stem}_fg_magenta_{style_name}.png")
        cmd = [
            "python", "/app/pipeline.py",
            "--input_image", baked_image,
            "--output_image", fg_output,
            "--model_type", "magenta",
            "--io_preset", "tanh",
            "--magenta_style", style_path,
            "--inference_res", "1080",
            "--scale", "720",
            "--mask", mask_output,
            "--fit_mask_to", "input",
            "--composite_mode", "keep",
            "--mask_autofix"
        ]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warning] Magenta FG styling failed for {style_name}: {e.stderr}")
            continue

        # BG (invert mask)
        bg_output = os.path.join(slides, f"{stem}_bg_magenta_{style_name}.png")
        cmd = [
            "python", "/app/pipeline.py",
            "--input_image", baked_image,
            "--output_image", bg_output,
            "--model_type", "magenta",
            "--io_preset", "tanh",
            "--magenta_style", style_path,
            "--inference_res", "1080",
            "--scale", "720",
            "--mask", mask_output,
            "--fit_mask_to", "input",
            "--mask_invert",
            "--composite_mode", "keep",
            "--mask_autofix"
        ]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warning] Magenta BG styling failed for {style_name}: {e.stderr}")
            continue

    # Step 3: Torch7 models
    torch7_models = [
        ("/app/models/torch/composition_vii_eccv16.t7", "composition_vii_eccv16"),
        ("/app/models/torch/la_muse_eccv16.t7", "la_muse_eccv16"),
        ("/app/models/torch/starry_night_eccv16.t7", "starry_night_eccv16"),
        ("/app/models/torch/the_scream.t7", "the_scream")
    ]
    for model_path, tag in torch7_models:
        print(f"[torch7] Styling with {model_path}, tag={tag}")

        # FG
        fg_output = os.path.join(slides, f"{stem}_fg_{tag}.png")
        cmd = [
            "python", "/app/pipeline.py",
            "--input_image", baked_image,
            "--output_image", fg_output,
            "--model", model_path,
            "--model_type", "torch7",
            "--io_preset", "caffe_bgr",
            "--inference_res", "1080",
            "--scale", "720",
            "--mask", mask_output,
            "--fit_mask_to", "input",
            "--composite_mode", "keep",
            "--mask_autofix"
        ]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warning] Torch7 FG styling failed for {tag}: {e.stderr}")
            continue

        # BG
        bg_output = os.path.join(slides, f"{stem}_bg_{tag}.png")
        cmd = [
            "python", "/app/pipeline.py",
            "--input_image", baked_image,
            "--output_image", bg_output,
            "--model", model_path,
            "--model_type", "torch7",
            "--io_preset", "caffe_bgr",
            "--inference_res", "1080",
            "--scale", "720",
            "--mask", mask_output,
            "--fit_mask_to", "input",
            "--mask_invert",
            "--composite_mode", "keep",
            "--mask_autofix"
        ]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warning] Torch7 BG styling failed for {tag}: {e.stderr}")
            continue

    # Step 4: PyTorch models
    pytorch_models = [
        ("/app/models/pytorch/mosaic.pth", "mosaic"),
        ("/app/models/pytorch/rain_princess.pth", "rain_princess"),
        ("/app/models/pytorch/udnie.pth", "udnie")
    ]
    for model_path, tag in pytorch_models:
        print(f"[pytorch] Styling with {model_path}, tag={tag}")

        # FG
        fg_output = os.path.join(slides, f"{stem}_fg_{tag}.png")
        cmd = [
            "python", "/app/pipeline.py",
            "--input_image", baked_image,
            "--output_image", fg_output,
            "--model", model_path,
            "--model_type", "transformer",
            "--io_preset", "imagenet_255",
            "--inference_res", "1080",
            "--scale", "720",
            "--mask", mask_output,
            "--fit_mask_to", "input",
            "--composite_mode", "keep",
            "--mask_autofix"
        ]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warning] PyTorch FG styling failed for {tag}: {e.stderr}")
            continue

        # BG
        bg_output = os.path.join(slides, f"{stem}_bg_{tag}.png")
        cmd = [
            "python", "/app/pipeline.py",
            "--input_image", baked_image,
            "--output_image", bg_output,
            "--model", model_path,
            "--model_type", "transformer",
            "--io_preset", "imagenet_255",
            "--inference_res", "1080",
            "--scale", "720",
            "--mask", mask_output,
            "--fit_mask_to", "input",
            "--mask_invert",
            "--composite_mode", "keep",
            "--mask_autofix"
        ]
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[warning] PyTorch BG styling failed for {tag}: {e.stderr}")
            continue

    print(f"✅ Styled PNGs saved to: {slides}")

if __name__ == "__main__":
    main()