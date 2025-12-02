#!/usr/bin/env python3
import argparse, torch, cv2, numpy as np, os
from PIL import Image, ImageOps
from torch.serialization import add_safe_globals
add_safe_globals([np.core.multiarray.scalar])

# Simple color palette for debug visualization
_DEF_PALETTE = np.array([
    [  0,   0,   0],[128, 64,128],[244, 35,232],[ 70, 70, 70],[102,102,156],
    [190,153,153],[153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],
    [152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],
    [  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32],
    [255,255,255]
], dtype=np.uint8)

def _colorize_pred(pred: np.ndarray) -> Image.Image:
    h,w = pred.shape
    K = _DEF_PALETTE.shape[0]
    idx = np.clip(pred, 0, K-1)
    rgb = _DEF_PALETTE[idx]
    return Image.fromarray(rgb, mode='RGB')

def _apply_transpose(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'none' or not mode:
        return arr
    if mode == 'rot90':
        return np.rot90(arr, 1)
    if mode == 'rot270':
        return np.rot90(arr, 3)
    if mode == 'flip_h':
        return np.ascontiguousarray(np.flip(arr, axis=1))
    if mode == 'flip_v':
        return np.ascontiguousarray(np.flip(arr, axis=0))
    return arr

def _pct_to_px_val(pct: float, base: int) -> int:
    try:
        return int(round(max(0.0, float(pct)) * 0.01 * base))
    except Exception:
        return 0

# --- Block any implicit backbone downloads (compat across repo variants) ---
try:
    from torch.utils import model_zoo
except Exception:
    model_zoo = None
try:
    import torchvision
except Exception:
    torchvision = None
import torch.hub as _torch_hub

def _no_download(*args, **kwargs):
    print("[note] Skipping backbone weight download (forced offline).")
    return {}

# Common download entry points used by jfzhang95 repo and forks.
try:
    if torchvision is not None and hasattr(torchvision.models, "resnet") and hasattr(torchvision.models.resnet, "load_state_dict_from_url"):
        torchvision.models.resnet.load_state_dict_from_url = _no_download
except Exception:
    pass
try:
    if model_zoo is not None and hasattr(model_zoo, "load_url"):
        model_zoo.load_url = _no_download
except Exception:
    pass
try:
    if hasattr(_torch_hub, "load_state_dict_from_url"):
        _torch_hub.load_state_dict_from_url = _no_download
except Exception:
    pass
# --------------------------------------------------------------------------

# ---- import DeepLab from local modeling/ (jfzhang95 repo layout) ----
try:
    from modeling.deeplab import DeepLab
except Exception as e:
    print("[error] Could not import DeepLab. Make sure modeling/ is present from jfzhang95 repo.")
    raise


CITYSCAPES_SKY_ID_DEFAULT = 10  # common sky trainId in Cityscapes configs

# ----- Label maps for common datasets -----
VOC21_LABELS = {
    "background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
    "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
    "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
    "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
    "sofa": 18, "train": 19, "tvmonitor": 20,
}

CITYSCAPES19_LABELS = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4,
    "pole": 5, "traffic light": 6, "traffic sign": 7, "vegetation": 8,
    "terrain": 9, "sky": 10, "person": 11, "rider": 12, "car": 13,
    "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18,
}

def canonicalize_label_name(s: str) -> str:
    # normalize user input to keys above
    return s.strip().lower().replace("_", " ").replace("-", " ")

def lookup_label_ids(label_names, used_nc: int):
    """Map a list of human-readable label names to class ids for the
    detected dataset (VOC=21 classes, Cityscapes=19 classes)."""
    if used_nc == 21:
        table = VOC21_LABELS
    elif used_nc == 19:
        table = CITYSCAPES19_LABELS
    else:
        # fall back: try VOC first, then Cityscapes; keep those that exist
        table = {**VOC21_LABELS, **CITYSCAPES19_LABELS}
    ids = []
    for name in label_names:
        key = canonicalize_label_name(name)
        if key in table:
            ids.append(int(table[key]))
        else:
            print(f"[warn] unknown label '{name}' for used_nc={used_nc}; skipping")
    return sorted(set(ids))

def _require_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[error] {label} not found: {path}")

def _detect_num_classes_from_state(state):
    cand = []
    for k, v in state.items():
        try:
            if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[2] == 1 and v.shape[3] == 1:
                K = int(v.shape[0])
                if 2 <= K <= 256:
                    cand.append(K)
        except Exception:
            continue
    for pref in (19, 21, 150, 80):
        if pref in cand:
            return pref
    return max(cand) if cand else None

def load_deeplab(weights_path, backbone="resnet", num_classes=None, device="cpu"):
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    detected_nc = _detect_num_classes_from_state(state)
    if num_classes is None and detected_nc is not None:
        num_classes = detected_nc
    if num_classes is None:
        num_classes = 19

    print(f"[info] using num_classes={num_classes} (detected={detected_nc}) backbone={backbone}")

    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        output_stride=16,
        sync_bn=False,
        freeze_bn=False
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("  unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    model.eval().to(device)
    return model, int(num_classes)

def preprocess_pil(pil_im):
    im = np.array(pil_im.convert("RGB"), dtype=np.float32) / 255.0
    im = (im - (0.485,0.456,0.406)) / (0.229,0.224,0.225)
    x = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()
    return x

def infer_mask(model, pil_im, sky_id=CITYSCAPES_SKY_ID_DEFAULT, device="cpu",
               expand_px=0, contract_px=0, feather_px=3, target_ids=None,
               return_pred: bool=False):
    x = preprocess_pil(pil_im).to(device)
    with torch.no_grad():
        out = model(x)
        logits = out.get("out", next(iter(out.values()))) if isinstance(out, dict) else out
        logits = torch.nn.functional.interpolate(logits, size=pil_im.size[::-1], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    raw_pred = pred.copy()

    if target_ids is None or len(target_ids) == 0:
        target_ids = [int(sky_id)]

    sky = np.zeros_like(pred, dtype=np.uint8)
    for cid in target_ids:
        sky |= (pred == int(cid)).astype(np.uint8)
    sky = (sky * 255).astype(np.uint8)

    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    if int(expand_px) > 0:
        k = int(expand_px) * 2 + 1
        sky = cv2.dilate(sky, np.ones((k, k), np.uint8), iterations=1)
    if int(contract_px) > 0:
        k = int(contract_px) * 2 + 1
        sky = cv2.erode(sky, np.ones((k, k), np.uint8), iterations=1)

    if int(feather_px) > 0:
        sigma = float(feather_px) * 0.5
        sky = cv2.GaussianBlur(sky, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if return_pred:
        return sky, raw_pred
    return sky

def guess_sky_id(model, pil_im, num_classes, top_frac=0.4, device="cpu"):
    x = preprocess_pil(pil_im).to(device)
    with torch.no_grad():
        out = model(x)
        logits = out.get("out", next(iter(out.values()))) if isinstance(out, dict) else out
        logits = torch.nn.functional.interpolate(logits, size=pil_im.size[::-1], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int32)

    h, w = pred.shape
    top_h = max(1, int(h * float(top_frac)))
    scores = []
    for cid in range(int(num_classes)):
        full = (pred == cid).sum() / float(h * w)
        top = (pred[:top_h, :] == cid).sum() / float(top_h * w)
        scores.append((top, full, cid))
    scores.sort(reverse=True)
    best_top, best_full, best_cid = scores[0]
    print(f"[info] scan_sky: best_id={best_cid} top={best_top:.3f} full={best_full:.3f}")
    return int(best_cid), float(best_top), float(best_full)

from typing import Tuple
def _resize_plate_preserve_ar(plate_pil: Image.Image, target_size: Tuple[int, int], mode: str = "crop") -> Image.Image:
    W, H = target_size
    if mode == "crop":
        return ImageOps.fit(plate_pil, (W, H), method=Image.LANCZOS, bleed=0.0, centering=(0.5, 0.5))
    elif mode == "pad":
        contained = ImageOps.contain(plate_pil, (W, H), method=Image.LANCZOS)
        canvas = Image.new("RGB", (W, H))
        try:
            edge_color = contained.getpixel((0, 0))
        except Exception:
            edge_color = (0, 0, 0)
        canvas.paste(edge_color, [0, 0, W, H])
        x = (W - contained.width) // 2
        y = (H - contained.height) // 2
        canvas.paste(contained, (x, y))
        return canvas
    else:
        return plate_pil.resize((W, H), Image.LANCZOS)

def composite(base_pil, plate_pil, mask_u8, fit_mode: str = "crop"):
    base = np.array(base_pil.convert("RGB"))
    plate_resized = _resize_plate_preserve_ar(plate_pil.convert("RGB"), (base.shape[1], base.shape[0]), mode=fit_mode)
    plate = np.array(plate_resized)
    alpha = (mask_u8.astype(np.float32)/255.0)[...,None]
    comp = (alpha*plate + (1.0-alpha)*base).astype(np.uint8)
    return Image.fromarray(comp)

from pathlib import Path

def batch_masks_from_frames(frames_dir: str, out_dir: str, model, sky_id: int, device: str,
                            expand_pct: float = 0.0, contract_pct: float = 0.0, feather_pct: float = 0.0,
                            expand_px: int = 0, contract_px: int = 0, feather_px: int = 3,
                            resolution: int = 256, verbose: bool = False,
                            target_ids=None, debug_pred: bool=False, debug_overlay: bool=False,
                            transpose: str='none', morph_close_ks: int=5):
    fdir = Path(frames_dir)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    frames = sorted([p for p in fdir.glob('frame_*.png')] + [p for p in fdir.glob('frame_*.jpg')] + [p for p in fdir.glob('frame_*.jpeg')])
    if verbose:
        print(f"[batch] frames_dir={fdir}  out_dir={odir}  found={len(frames)}")
    if not frames:
        raise FileNotFoundError(f"[batch][error] No frames like frame_*.png/.jpg in {fdir}")

    n_ok = 0
    for fp in frames:
        num = fp.stem.split('_')[-1]
        dst = odir / f"mask_{num}.png"
        try:
            img = Image.open(fp).convert('RGB')
            orig_w, orig_h = img.size
            # Optional downscale by resolution (like single-image path)
            if resolution and resolution > 0:
                w, h = img.size
                scale = float(resolution) / max(w, h)
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    if verbose:
                        print(f"[batch][{num}] resized to {new_w}x{new_h} (max side {resolution})")

            # Compute effective px from pct (based on current working height)
            w2, h2 = img.size
            e_px = _pct_to_px_val(expand_pct, h2) if expand_pct and expand_pct > 0 else int(expand_px)
            c_px = _pct_to_px_val(contract_pct, h2) if contract_pct and contract_pct > 0 else int(contract_px)
            f_px = _pct_to_px_val(feather_pct, h2) if feather_pct and feather_pct > 0 else int(feather_px)

            if verbose:
                print(f"[batch][{num}] expand_px={e_px} (pct={expand_pct}), contract_px={c_px} (pct={contract_pct}), feather_px={f_px} (pct={feather_pct})")

            if debug_pred or debug_overlay:
                m, pred = infer_mask(model, img, sky_id=sky_id, device=device,
                                     expand_px=e_px, contract_px=c_px, feather_px=f_px,
                                     target_ids=target_ids, return_pred=True)
            else:
                m = infer_mask(model, img, sky_id=sky_id, device=device,
                               expand_px=e_px, contract_px=c_px, feather_px=f_px,
                               target_ids=target_ids, return_pred=False)
            # If we resized the working image, upsample mask back to original frame size
            if m.shape[1] != orig_w or m.shape[0] != orig_h:
                try:
                    import cv2 as _cv
                    m = _cv.resize(m, (orig_w, orig_h), interpolation=_cv.INTER_LINEAR)
                    if debug_pred or debug_overlay:
                        pred = _cv.resize(pred, (orig_w, orig_h), interpolation=_cv.INTER_NEAREST)
                except Exception:
                    from PIL import Image as _PILImage
                    _tmp = _PILImage.fromarray(m)
                    _tmp = _tmp.resize((orig_w, orig_h), _PILImage.BILINEAR)
                    m = np.array(_tmp, dtype=np.uint8)
                    if debug_pred or debug_overlay:
                        _tmp_pred = _PILImage.fromarray(pred)
                        _tmp_pred = _tmp_pred.resize((orig_w, orig_h), _PILImage.NEAREST)
                        pred = np.array(_tmp_pred, dtype=np.uint8)
            if transpose and transpose != 'none':
                m = _apply_transpose(m, transpose)
                if debug_pred or debug_overlay:
                    pred = _apply_transpose(pred, transpose)
            # Optional debug visualizations
            if debug_pred:
                _colorize_pred(pred).resize((orig_w, orig_h), Image.NEAREST).save(odir / f"pred_{num}.png")
            if debug_overlay:
                base = Image.open(fp).convert('RGB')
                if base.size != (orig_w, orig_h):
                    base = base.resize((orig_w, orig_h), Image.LANCZOS)
                overlay = np.array(base, dtype=np.uint8)
                alpha = (m.astype(np.float32) / 255.0)[:, :, None]
                red = np.zeros_like(overlay)
                red[..., 0] = 255
                mixed = (alpha * red + (1 - alpha) * overlay).astype(np.uint8)
                overlay_img = Image.fromarray(mixed)
                try:
                    if overlay_img is not None:
                        overlay_path = odir / f"overlay_{num}.jpg"
                        overlay_img.save(overlay_path, quality=92)
                        if verbose:
                            print(f"[mask][debug] wrote {overlay_path}")
                except Exception as _e:
                    if verbose:
                        print(f"[mask][debug][WARN] could not save overlay: {_e}")
            Image.fromarray(m).save(dst)
            n_ok += 1
        except Exception as ex:
            print(f"[batch][warn] failed {fp.name}: {ex}")
    print(f"[batch] wrote {n_ok}/{len(frames)} masks to {odir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Print extra diagnostics (file checks, sizes, chosen sky id).")
    ap.add_argument("--image", required=False, help="Input image/frame (required unless --batch_frames is used)")
    ap.add_argument("--weights", required=True, help="deeplab-resnet.pth.tar or deeplab-drn.pth.tar")
    ap.add_argument("--backbone", choices=["resnet","drn"], default="resnet")
    ap.add_argument("--sky_id", type=int, default=CITYSCAPES_SKY_ID_DEFAULT)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--scan_sky", action="store_true")
    ap.add_argument("--scan_top_frac", type=float, default=0.4)
    ap.add_argument("--plate", help="Optional sky plate image to composite")
    ap.add_argument("--plate_fit", choices=["crop", "pad", "stretch"], default="crop")
    ap.add_argument("--out_mask", default="sky_mask.png")
    ap.add_argument("--out_image", default="sky_swapped.jpg")
    ap.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--mask_expand", type=int, default=0)
    ap.add_argument("--mask_contract", type=int, default=0)
    ap.add_argument("--mask_feather", type=int, default=3)
    ap.add_argument("--mask_expand_pct", type=float, default=0.0)
    ap.add_argument("--mask_contract_pct", type=float, default=0.0)
    ap.add_argument("--mask_feather_pct", type=float, default=0.0)
    # Batch mode: generate masks for a frames directory (frame_####.png/.jpg)
    ap.add_argument("--batch_frames", type=str, default=None, help="Directory containing frame_####.png/.jpg to mask in batch.")
    ap.add_argument("--batch_out_dir", type=str, default=None, help="Output directory to write mask_####.png (defaults to <frames_dir>/../masks).")
    ap.add_argument("--target_labels", type=str, default=None,
                    help="Comma-separated list of labels to segment (e.g. 'person,bicycle,motorbike'). Overrides --sky_id.")
    ap.add_argument("--target_ids", type=str, default=None,
                    help="Comma-separated list of class ids to segment (dataset-dependent). Overrides --sky_id.")
    ap.add_argument("--debug_pred", action="store_true", help="Save a colorized class prediction image (pred_####.png) next to masks in batch mode.")
    ap.add_argument("--debug_overlay", action="store_true", help="Save an RGB overlay showing the mask in red (overlay_####.jpg) next to masks in batch mode.")
    ap.add_argument("--transpose", choices=["none","rot90","rot270","flip_h","flip_v"], default="none", help="Optional transform applied to each mask (and debug outputs) before saving, to compensate for orientation issues.")
    ap.add_argument("--morph_close_ks", type=int, default=5, help="Kernel size for morphology close (0 disables).")
    args = ap.parse_args()

    # Require either single-image mode or batch mode .
    if not args.batch_frames and not args.image:
        ap.error("either --image or --batch_frames must be provided")

    # --- preflight checks ---
    if args.image:
        _require_file(args.image, "input image")
    _require_file(args.weights, "weights checkpoint")
    if args.plate:
        _require_file(args.plate, "sky plate")
    if args.verbose:
        print(f"[preflight] image={args.image}")
        print(f"[preflight] weights={args.weights}")
        if args.plate:
            print(f"[preflight] plate={args.plate}")
        print(f"[preflight] backbone={args.backbone} device={args.device}")

    # --- DRN compatibility shim: ensure deletes of fc.* won't KeyError when offline ---
    if args.backbone == "drn":
        try:
            import torch.utils.model_zoo as _mz
            def _drn_compat_load_url(*_a, **_k):
                # Return stub with dummy fc.* entries so downstream 'del' calls succeed,
                # then loaders see an empty dict and effectively skip loading.
                print("[note] DRN compat: stubbing model_zoo.load_url with dummy fc keys.")
                import torch as _t
                return {}
            _mz.load_url = _drn_compat_load_url
        except Exception:
            pass
    # ------------------------------------------------------------------------------
    dev = torch.device(args.device)
    model, used_nc = load_deeplab(args.weights, backbone=args.backbone, num_classes=args.num_classes, device=dev)

    # --- Batch mode path ---
    if args.batch_frames:
        frames_dir = args.batch_frames
        out_dir = args.batch_out_dir or str(Path(frames_dir).parent / 'masks')
        sky_id = None
        if args.scan_sky:
            # Use the first frame to guess sky_id
            try:
                first = sorted(list(Path(frames_dir).glob('frame_*.png')) + list(Path(frames_dir).glob('frame_*.jpg')))[0]
                probe = Image.open(first).convert('RGB')
                if args.resolution and args.resolution > 0:
                    w, h = probe.size
                    sc = float(args.resolution) / max(w, h)
                    if sc < 1.0:
                        probe = probe.resize((int(w*sc), int(h*sc)), Image.LANCZOS)
                gid, _, _ = guess_sky_id(model, probe, used_nc, top_frac=args.scan_top_frac, device=dev)
                sky_id = gid
                if args.verbose:
                    print(f"[batch] scan_sky chose sky_id={sky_id}")
            except Exception as _e:
                if args.verbose:
                    print(f"[batch][warn] scan_sky failed: {_e}; falling back to args.sky_id")
        if sky_id is None:
            sky_id = args.sky_id
            if args.verbose:
                print(f"[batch] using sky_id={sky_id} (args)")

        # Determine target class ids (by names or raw ids); fall back to sky_id
        selected_ids = None
        if args.target_labels:
            names = [s for s in args.target_labels.split(',') if s.strip()]
            selected_ids = lookup_label_ids(names, used_nc)
            if args.verbose:
                print(f"[targets] using labels={names} -> ids={selected_ids}")
        elif args.target_ids:
            try:
                selected_ids = sorted(set(int(x) for x in args.target_ids.split(',') if x.strip()))
                if args.verbose:
                    print(f"[targets] using explicit ids={selected_ids}")
            except Exception as _e:
                print(f"[warn] could not parse --target_ids: {_e}")
        if not selected_ids:
            selected_ids = [int(sky_id)]
            if args.verbose:
                print(f"[targets] defaulting to sky_id={selected_ids}")

        batch_masks_from_frames(
            frames_dir=frames_dir,
            out_dir=out_dir,
            model=model,
            sky_id=int(sky_id),
            device=args.device,
            expand_pct=float(args.mask_expand_pct or 0.0),
            contract_pct=float(args.mask_contract_pct or 0.0),
            feather_pct=float(args.mask_feather_pct or 0.0),
            expand_px=int(args.mask_expand or 0),
            contract_px=int(args.mask_contract or 0),
            feather_px=int(args.mask_feather or 0),
            resolution=int(args.resolution or 0),
            verbose=bool(args.verbose),
            target_ids=selected_ids,
            debug_pred=bool(args.debug_pred),
            debug_overlay=bool(args.debug_overlay),
            transpose=args.transpose,
            morph_close_ks=int(args.morph_close_ks),
        )
        return

    src = Image.open(args.image).convert("RGB")
    if args.resolution is not None and args.resolution > 0:
        w, h = src.size
        scale = float(args.resolution) / max(w, h)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            src = src.resize((new_w, new_h), Image.LANCZOS)

    if args.verbose:
        print(f"[preprocess] working size: {src.size[0]}x{src.size[1]} (after optional downscale to max side {args.resolution})")

    w, h = src.size
    def _pct_to_px(pct, base):
        try:
            return int(round(max(0.0, float(pct)) * 0.01 * base))
        except Exception:
            return 0
    expand_px_eff = _pct_to_px_val(args.mask_expand_pct, h) if args.mask_expand_pct and args.mask_expand_pct > 0 else int(args.mask_expand)
    contract_px_eff = _pct_to_px_val(args.mask_contract_pct, h) if args.mask_contract_pct and args.mask_contract_pct > 0 else int(args.mask_contract)
    feather_px_eff = _pct_to_px_val(args.mask_feather_pct, h) if args.mask_feather_pct and args.mask_feather_pct > 0 else int(args.mask_feather)

    if args.verbose:
        print(f"[mask cfg] expand_px={expand_px_eff} (pct={args.mask_expand_pct}), contract_px={contract_px_eff} (pct={args.mask_contract_pct}), feather_px={feather_px_eff} (pct={args.mask_feather_pct})")

    if args.scan_sky:
        guessed_id, _, _ = guess_sky_id(model, src, used_nc, top_frac=args.scan_top_frac, device=dev)
        sky_id = guessed_id
        if args.verbose:
            print(f"[diagnostic] using sky_id={sky_id} (auto-scanned)")
    else:
        sky_id = args.sky_id
        if args.verbose:
            print(f"[diagnostic] using sky_id={sky_id} (from args)")

    if used_nc == 21 and not args.scan_sky and args.sky_id == CITYSCAPES_SKY_ID_DEFAULT:
        print("[warn] Checkpoint looks VOC-like (21 classes). Default Cityscapes sky_id=10 may be incorrect. Use --scan_sky or set --sky_id explicitly.")

    # Select targets for single-image mode
    selected_ids = None
    if args.target_labels:
        names = [s for s in args.target_labels.split(',') if s.strip()]
        selected_ids = lookup_label_ids(names, used_nc)
        if args.verbose:
            print(f"[targets] using labels={names} -> ids={selected_ids}")
    elif args.target_ids:
        try:
            selected_ids = sorted(set(int(x) for x in args.target_ids.split(',') if x.strip()))
            if args.verbose:
                print(f"[targets] using explicit ids={selected_ids}")
        except Exception as _e:
            print(f"[warn] could not parse --target_ids: {_e}")
    if not selected_ids:
        selected_ids = [int(sky_id)]
        if args.verbose:
            print(f"[targets] defaulting to sky_id={selected_ids}")

    mask = infer_mask(model, src, sky_id=sky_id, device=dev,
                      expand_px=expand_px_eff, contract_px=contract_px_eff, feather_px=feather_px_eff,
                      target_ids=selected_ids)
    if args.transpose and args.transpose != 'none':
        mask = _apply_transpose(mask, args.transpose)
    Image.fromarray(mask).save(args.out_mask)
    print(f"[ok] wrote mask → {args.out_mask}" + (f" ({os.path.abspath(args.out_mask)})" if args.verbose else ""))

    if args.debug_overlay:
        base = src if src.size == mask.shape[::-1] else src.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        overlay = np.array(base, dtype=np.uint8)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        red = np.zeros_like(overlay); red[...,0] = 255
        mixed = (alpha * red + (1 - alpha) * overlay).astype(np.uint8)
        Image.fromarray(mixed).save("overlay_single.jpg", quality=92)
        print("[ok] wrote overlay → overlay_single.jpg")

    if args.plate:
        plate = Image.open(args.plate).convert("RGB")
        out = composite(src, plate, mask, fit_mode=args.plate_fit)
        out.save(args.out_image, quality=95)
        print(f"[ok] wrote composite → {args.out_image}" + (f" ({os.path.abspath(args.out_image)})" if args.verbose else ""))
    else:
        print("[note] no --plate provided; skipping composite")

if __name__ == "__main__":
    main()