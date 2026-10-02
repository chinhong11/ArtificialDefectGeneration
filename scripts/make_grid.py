"""Contact sheet of images, optionally with mask contours (red) and boxes (green).

Use it to review annotation masks, training crops and generated samples.

Examples:
    # review masks made by labelme_to_masks / sam_box_to_mask
    python scripts/make_grid.py --images data/metal_part/defect/scratch \
        --masks data/metal_part/masks/scratch --overlay --out outputs/check_masks.png

    # real vs synthetic side by side: run twice and compare
    python scripts/make_grid.py --images outputs/gen_scratch/images \
        --masks outputs/gen_scratch/masks --overlay --boxes --out outputs/gen_grid.png
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import find_mask_for, list_images, read_mask, read_rgb, write_rgb  # noqa: E402
from src.export import mask_to_boxes  # noqa: E402


def tile(img, mask, overlay, boxes, size):
    img = img.copy()
    if mask is not None and overlay:
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 0, 0), max(1, img.shape[1] // 300))
    if mask is not None and boxes:
        for x0, y0, x1, y1 in mask_to_boxes(mask):
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), max(1, img.shape[1] // 300))
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 32, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks")
    ap.add_argument("--overlay", action="store_true", help="Draw mask contours")
    ap.add_argument("--boxes", action="store_true", help="Draw boxes derived from masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = list_images(args.images)
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.n]
    if not paths:
        sys.exit(f"No images in {args.images}")
    tiles = []
    for p in paths:
        mp = find_mask_for(p, args.masks) if args.masks else None
        tiles.append(tile(read_rgb(p), read_mask(mp) if mp else None, args.overlay, args.boxes, args.tile))
    while len(tiles) % args.cols:
        tiles.append(np.full_like(tiles[0], 32))
    rows = [np.concatenate(tiles[i:i + args.cols], axis=1) for i in range(0, len(tiles), args.cols)]
    write_rgb(args.out, np.concatenate(rows, axis=0))
    print(f"Grid of {len(paths)} images -> {args.out}")


if __name__ == "__main__":
    main()
