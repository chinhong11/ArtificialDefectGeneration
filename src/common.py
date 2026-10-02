"""Shared image / mask I/O helpers."""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def read_mask(path):
    """Read a mask as uint8 {0, 255}. Any non-zero pixel counts as defect."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return np.where(m > 0, 255, 0).astype(np.uint8)


def write_mask(path, mask):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.where(mask > 0, 255, 0).astype(np.uint8))


def find_mask_for(image_path, mask_dir):
    """Mask has the same stem as the image (any image extension)."""
    for ext in (".png", *IMAGE_EXTS):
        p = Path(mask_dir) / (Path(image_path).stem + ext)
        if p.exists():
            return p
    return None


def mask_bbox(mask):
    """(x0, y0, x1, y1) of non-zero pixels, x1/y1 exclusive, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)
