"""Automatic part area (ROI) on good images - no clicking needed.

Assumes a plain, fairly uniform background (e.g. backlit or a flat table): the
background is estimated with a large median filter, pixels that differ from it
are the part (edges, texture, print), gaps are closed, holes filled, and the
result shrunk a little so the noisy halo at the border is dropped. Missing a bit
of the part only means fewer places to paint defects; including background
would put defects in the air - so it errs on the small side.

CLI (writes <out>/<stem>.png, white = part):
    python -m src.roi --images data/p/good --out data/p/work/roi_auto --sensitivity 6
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.common import list_images, write_mask


def auto_part_mask(img, sensitivity=6.0, close_frac=0.03, shrink_frac=0.012, min_area_frac=0.01):
    """img: RGB/BGR/gray uint8. sensitivity: grey-level difference from background (lower = more)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = g.astype(np.float32)
    h, w = g.shape
    small = cv2.resize(g, (max(8, w // 4), max(8, h // 4))).astype(np.uint8)
    k = max(3, (min(small.shape) // 6) | 1)
    bg = cv2.resize(cv2.medianBlur(small, k), (w, h)).astype(np.float32)
    m = ((np.abs(cv2.GaussianBlur(g, (5, 5), 0) - bg) > sensitivity) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop sensor noise
    ck = max(3, int(min(h, w) * close_frac) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # border so (0,0) is background
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)       # fill holes
    m = m | cv2.bitwise_not(ff[1:-1, 1:-1])
    sk = int(min(h, w) * shrink_frac) | 1
    if sk > 1:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sk, sk)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > min_area_frac * h * w:
            out[lab == i] = 255
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensitivity", type=float, default=6.0)
    args = ap.parse_args()
    for p in list_images(args.images):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        m = auto_part_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), args.sensitivity)
        write_mask(Path(args.out) / f"{p.stem}.png", m)
        print(f"{p.name}: part = {100 * (m > 0).mean():.0f}% of image")


if __name__ == "__main__":
    main()
