import os
import glob
import subprocess
import shutil
import json

# Configuration
SLIDES_DIR = "/app/_work/winter1/slides"
OUT = "/app/output/slideshow.mp4"
FPS = 30
HOLD = 1  # seconds fully visible before each crossfade
TRANS = 2  # seconds crossfade duration
TAIL = 1  # seconds to hold the last frame
CRF = 18  # quality for libx264
PRESET = "slow"  # encoding preset
TMP = "/app/_work/slideshow_tmp"

def run_command(cmd, check=True):
    """Run a shell command and return its output."""
    print(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result.stdout, result.stderr

def check_ffmpeg():
    """Check if ffmpeg and ffprobe are available."""
    try:
        run_command(["ffmpeg", "-version"])
        run_command(["ffprobe", "-version"])
    except subprocess.CalledProcessError:
        print("[error] ffmpeg or ffprobe not found")
        exit(1)
    print("[debug] ffmpeg version: ", run_command(["ffmpeg", "-version"])[0].split("\n")[0])

def get_image_dimensions(image_path):
    """Get the width and height of an image using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", image_path
    ]
    stdout, _ = run_command(cmd)
    data = json.loads(stdout)
    return data["streams"][0]["width"], data["streams"][0]["height"]

def get_duration(file_path):
    """Get the duration of a video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1", file_path
    ]
    stdout, _ = run_command(cmd, check=False)
    try:
        return float(stdout.strip())
    except ValueError:
        return 0.0

def determine_video_resolution(images):
    """Determine the video resolution based on the first image's aspect ratio."""
    width, height = get_image_dimensions(images[0])
    print(f"[info] First image dimensions: {width}x{height}")
    if width > height:
        # Landscape: Use 1920x1080 or scale proportionally
        target_width = 1920
        target_height = int(target_width * height / width)
        if target_height % 2 != 0:
            target_height -= 1  # Ensure even dimensions for libx264
    else:
        # Portrait: Use 1080x1920 or scale proportionally
        target_width = 1080
        target_height = int(target_width * height / width)
        if target_height % 2 != 0:
            target_height -= 1  # Ensure even dimensions
    print(f"[info] Video resolution: {target_width}x{target_height}")
    return target_width, target_height

def main():
    # Check ffmpeg
    check_ffmpeg()

    # Clean and create temporary directory
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    if os.path.exists(OUT):
        os.remove(OUT)
    os.makedirs(os.path.join(TMP, "clips"), exist_ok=True)

    # Gather images by mtime (oldest → newest)
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(glob.glob(os.path.join(SLIDES_DIR, ext)))
    images.sort(key=lambda x: os.path.getmtime(x))
    if len(images) < 2:
        print(f"[error] need at least 2 images in {SLIDES_DIR}")
        exit(1)
    print(f"[info] Found {len(images)} images: {[os.path.basename(f) for f in images]}")

    # Determine video resolution based on the first image
    W, H = determine_video_resolution(images)

    # Verify all images have compatible dimensions
    for img in images[1:]:
        w, h = get_image_dimensions(img)
        if (w > h) != (W > H):
            print(f"[warning] Image {os.path.basename(img)} has different orientation ({w}x{h}) than first image")
            # Optionally, you could exit or adjust here; for now, we'll proceed

    # Make CFR clip for each image
    clips = []
    for i, f in enumerate(images):
        base = f"clip_{i:03d}.mp4"
        clip_path = os.path.join(TMP, "clips", base)
        duration = HOLD + TRANS
        if i == len(images) - 1:
            duration += TAIL  # Add tail to last clip
        print(f"[make] {os.path.basename(f)} -> {base} (duration={duration}s @ {FPS}fps)")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-loop", "1", "-framerate", str(FPS), "-i", f,
            "-vf", f"scale={W}:{H}:flags=lanczos:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,fps=fps={FPS},format=yuv420p,setsar=1",
            "-t", str(duration),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF), "-preset", PRESET, "-r", str(FPS),
            "-movflags", "+faststart", "-video_track_timescale", str(FPS), "-fps_mode", "cfr",
            clip_path
        ]
        run_command(cmd)
        clips.append(clip_path)

    # Debug clip metadata
    for clip in clips:
        print(f"[debug] Checking {clip}")
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate,time_base",
            "-of", "default=nokey=1:noprint_wrappers=1", clip
        ]
        stdout, _ = run_command(cmd)
        for line in stdout.splitlines():
            print(f"[probe] {line}")

    # Chain clips with xfade
    print(f"[chain] Building slideshow with {len(clips)} clips")
    accum = os.path.join(TMP, "accum.mp4")
    shutil.copy(clips[0], accum)

    for k in range(1, len(clips)):
        next_clip = clips[k]
        # Calculate duration of accumulated video
        accum_dur = get_duration(accum)
        print(f"[debug] Accumulated duration: {accum_dur}s")
        # Start transition TRANS seconds before the end of accum
        offset = max(accum_dur - TRANS, 0)
        print(f"[xfade] {os.path.basename(clips[k-1])} + {os.path.basename(next_clip)}  trans=fade dur={TRANS}s offset={offset:.6f}s")
        output = os.path.join(TMP, "accum_next.mp4")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", accum, "-i", next_clip,
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={TRANS}[a];[1:v]tpad=stop_mode=clone:stop_duration={TRANS}[b];[a][b]xfade=transition=fade:duration={TRANS}:offset={offset:.6f},format=yuv420p[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF), "-preset", PRESET, "-r", str(FPS),
            "-movflags", "+faststart", "-video_track_timescale", str(FPS), "-fps_mode", "cfr",
            output
        ]
        run_command(cmd)
        shutil.move(output, accum)

    # Move final accumulated video to output
    shutil.move(accum, OUT)

    # Sanity check
    print(f"[done] {OUT}")
    duration = get_duration(OUT)
    print(f"[done] Duration: {duration:.6f}s")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,time_base",
        "-of", "default=nokey=1:noprint_wrappers=1", OUT
    ]
    stdout, _ = run_command(cmd)
    for line in stdout.splitlines():
        print(f"[probe] {line}")

if __name__ == "__main__":
    main()