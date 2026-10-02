"""Paint synthetic defects into GOOD images with (optionally LoRA-tuned) SD inpainting.

For each sample:
  1. pick a good image and a SIZE x SIZE window on it (inside the ROI if given)
  2. make a defect mask (real-shape / stroke / blob, see src/mask_gen.py)
  3. inpaint the window inside the mask, blend it back with a feathered mask
     (pixels outside the mask stay exactly the original)
  4. refine the label: keep only pixels that actually changed
     (|generated - original| > threshold inside the mask); drop samples where
     almost nothing changed
  5. save full-res image + mask PNG + YOLO label + metadata

Example (after training):
    python -m src.diffusion.inpaint_generate \
        --good-dir data/metal_part/good --lora outputs/lora_scratch \
        --prompt "a photo of sks scratch" --mask-mode mixed \
        --real-mask-dir data/metal_part/crops/scratch/masks \
        --num 200 --out outputs/gen_scratch --class-id 0

Without --lora this is the zero-shot baseline (see scripts/zero_shot_inpaint.py).
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.common import dilate, list_images, read_mask, read_rgb, write_mask, write_rgb
from src.export import mask_to_boxes, write_yolo
from src.mask_gen import MaskGenerator


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative-prompt", default="blurry, cartoon, drawing, text, watermark")
    ap.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--lora", help="Folder with pytorch_lora_weights.safetensors from train_lora")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--roi-dir", help="Optional per-good-image masks (same stem): white = defects allowed here")
    ap.add_argument("--min-roi-overlap", type=float, default=0.98)
    ap.add_argument("--mask-mode", default="mixed", choices=["mixed", "real", "stroke", "blob"])
    ap.add_argument("--real-mask-dir", help="Crop masks from prepare_crops (<crops>/masks) for real-shape masks")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--size", type=int, default=512, help="Window size; keep equal to the training crop size")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--mask-dilate", type=int, default=6, help="Inpaint region = mask grown by N px")
    ap.add_argument("--feather", type=int, default=5, help="Blur radius for blending back")
    ap.add_argument("--diff-threshold", type=float, default=12.0, help="0-255 change needed to count as defect")
    ap.add_argument("--min-changed-ratio", type=float, default=0.2,
                    help="Discard if changed px < ratio * planned mask px")
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="Run in float32 (CPU or debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=3, help="Generation attempts per sample")
    return ap.parse_args(argv)


def pick_window(img_h, img_w, size, rng):
    side = min(size, img_h, img_w)
    return rng.randint(0, img_w - side), rng.randint(0, img_h - side), side


def refine_mask(orig, gen, planned, threshold, min_area=16):
    """Pixels that really changed, inside the planned (dilated) region."""
    diff = np.abs(gen.astype(np.int16) - orig.astype(np.int16)).max(axis=2).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    m = np.where((diff > threshold) & (planned > 0), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            m[labels == i] = 0
    return m


def blend(orig, gen, region, feather):
    """Only pixels inside the (feathered) region come from the generated image."""
    a = region.astype(np.float32) / 255.0
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather)
    a = a[..., None]
    return (gen.astype(np.float32) * a + orig.astype(np.float32) * (1 - a)).round().astype(np.uint8)


def load_pipe(args):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (args.fp32 or device == "cpu") else torch.float16
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False).to(device)
    if args.lora:
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def main(argv=None):
    args = parse_args(argv)
    import torch
    from PIL import Image

    rng = random.Random(args.seed)
    good = list_images(args.good_dir)
    if not good:
        raise SystemExit(f"No images in {args.good_dir}")
    masks = MaskGenerator(args.size, args.mask_mode, args.real_mask_dir, seed=args.seed)
    pipe, device = load_pipe(args)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = open(out / "metadata.jsonl", "a")
    done = discarded = roi_misses = 0
    while done < args.num:
        if discarded > 10 * args.num + 50:
            print("Too many discarded samples - check prompt / LoRA / --diff-threshold. Stopping.")
            break
        if roi_misses > 2000 * args.num + 2000:
            print("Could not place masks inside the ROI - check --roi-dir masks or lower --min-roi-overlap. Stopping.")
            break
        gpath = rng.choice(good)
        img = read_rgb(gpath)
        roi = read_mask(Path(args.roi_dir) / (gpath.stem + ".png")) if args.roi_dir else None
        h, w = img.shape[:2]

        x0, y0, side = pick_window(h, w, args.size, rng)
        planned, mode = masks()
        planned_win = cv2.resize(planned, (side, side), interpolation=cv2.INTER_NEAREST) if side != args.size else planned
        if roi is not None:
            roi_win = roi[y0:y0 + side, x0:x0 + side]
            inside = ((planned_win > 0) & (roi_win > 0)).sum() / max(1, (planned_win > 0).sum())
            if inside < args.min_roi_overlap:
                roi_misses += 1
                continue  # resample location/mask; not counted as a discard

        win = img[y0:y0 + side, x0:x0 + side]
        win_s = cv2.resize(win, (args.size, args.size), interpolation=cv2.INTER_AREA) if side != args.size else win
        region = dilate(planned, args.mask_dilate)

        ok = False
        for _ in range(args.max_tries):
            res = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt, image=Image.fromarray(win_s),
                       mask_image=Image.fromarray(region), height=args.size, width=args.size,
                       num_inference_steps=args.steps, guidance_scale=args.guidance, strength=args.strength,
                       generator=gen).images[0]
            res = np.asarray(res.convert("RGB"))
            if side != args.size:
                res = cv2.resize(res, (side, side), interpolation=cv2.INTER_AREA)
            reg_win = cv2.resize(region, (side, side), interpolation=cv2.INTER_NEAREST) if side != args.size else region
            comp = blend(win, res, reg_win, args.feather)
            label = refine_mask(win, comp, dilate(reg_win, 2 * args.feather), args.diff_threshold)
            if (label > 0).sum() >= args.min_changed_ratio * max(1, (planned_win > 0).sum()):
                ok = True
                break
        if not ok:
            discarded += 1
            continue

        full = img.copy()
        full[y0:y0 + side, x0:x0 + side] = comp
        full_mask = np.zeros((h, w), np.uint8)
        full_mask[y0:y0 + side, x0:x0 + side] = label
        name = f"{gpath.stem}_syn{done:05d}"
        write_rgb(out / "images" / f"{name}.png", full)
        write_mask(out / "masks" / f"{name}.png", full_mask)
        boxes = mask_to_boxes(full_mask)
        write_yolo(out / "labels" / f"{name}.txt", boxes, w, h, args.class_id)
        meta.write(json.dumps({"image": f"images/{name}.png", "mask": f"masks/{name}.png", "source": gpath.name,
                               "window": [x0, y0, side], "mask_mode": mode, "prompt": args.prompt,
                               "lora": args.lora, "boxes": boxes}) + "\n")
        meta.flush()
        done += 1
        print(f"[{done}/{args.num}] {name} ({mode}, {len(boxes)} box(es))")
    meta.close()
    print(f"Done: {done} saved, {discarded} discarded -> {out}")


if __name__ == "__main__":
    main()
