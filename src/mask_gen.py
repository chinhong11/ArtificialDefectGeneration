"""Masks that say WHERE and in WHAT SHAPE a new defect is painted.

Modes:
  real   - a real defect mask crop (from prepare_crops) randomly flipped,
           rotated, scaled and elastically deformed. Most realistic shapes.
  stroke - thin curved lines (scratches, cracks).
  blob   - irregular blobs (stains, dents, spots, contamination).

All functions return a SIZE x SIZE uint8 mask {0, 255}.
"""
import random

import cv2
import numpy as np

from src.common import list_images, mask_bbox, read_mask


def stroke_mask(size, rng, n_strokes=(1, 2), thickness=(2, 6), length=(0.1, 0.45)):
    m = np.zeros((size, size), np.uint8)
    for _ in range(rng.randint(*n_strokes)):
        total = rng.uniform(*length) * size
        n_seg = rng.randint(4, 10)
        step = total / n_seg
        x, y = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(x, y)]
        for _ in range(n_seg):
            ang += rng.gauss(0, 0.25)
            x = float(np.clip(x + step * np.cos(ang), 0.1 * size, 0.9 * size))
            y = float(np.clip(y + step * np.sin(ang), 0.1 * size, 0.9 * size))
            pts.append((x, y))
        cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 255,
                      thickness=rng.randint(*thickness), lineType=cv2.LINE_AA)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def blob_mask(size, rng, radius=(0.04, 0.15), roughness=0.5):
    """Ellipse distorted by smoothed noise -> irregular natural-looking blob."""
    r = rng.uniform(*radius) * size
    cx, cy = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size
    ax = r * rng.uniform(0.6, 1.4)
    ay = r * rng.uniform(0.6, 1.4)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = np.deg2rad(rng.uniform(0, 180))
    dx, dy = xx - cx, yy - cy
    u = (dx * np.cos(t) + dy * np.sin(t)) / ax
    v = (-dx * np.sin(t) + dy * np.cos(t)) / ay
    dist = np.sqrt(u ** 2 + v ** 2)
    noise = np.random.default_rng(rng.randrange(2 ** 32)).standard_normal((size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.0, r / 3))
    noise /= noise.std() + 1e-6
    return np.where(dist + roughness * 0.3 * noise < 1.0, 255, 0).astype(np.uint8)


def deform_real(mask, rng, scale=(0.7, 1.3), elastic=0.0):
    """Random flip / rotate / scale about the defect centre (+ optional elastic)."""
    size = mask.shape[0]
    m = mask.copy()
    if rng.random() < 0.5:
        m = m[:, ::-1]
    if rng.random() < 0.5:
        m = m[::-1, :]
    bb = mask_bbox(m)
    if bb is None:
        return m
    c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
    M = cv2.getRotationMatrix2D(c, rng.uniform(0, 360), rng.uniform(*scale))
    # move defect centre towards a random point in the middle of the crop
    M[0, 2] += rng.uniform(0.35, 0.65) * size - c[0]
    M[1, 2] += rng.uniform(0.35, 0.65) * size - c[1]
    m = cv2.warpAffine(np.ascontiguousarray(m), M, (size, size), flags=cv2.INTER_NEAREST)
    if elastic > 0:
        g = np.random.default_rng(rng.randrange(2 ** 32))
        dx = cv2.GaussianBlur(g.standard_normal((size, size)).astype(np.float32), (0, 0), 8) * elastic
        dy = cv2.GaussianBlur(g.standard_normal((size, size)).astype(np.float32), (0, 0), 8) * elastic
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        m = cv2.remap(m, xx + dx, yy + dy, cv2.INTER_NEAREST)
    return np.where(m > 0, 255, 0).astype(np.uint8)


class MaskGenerator:
    def __init__(self, size, mode="mixed", real_mask_dir=None, seed=0):
        self.size, self.mode = size, mode
        self.rng = random.Random(seed)
        self.real = []
        if real_mask_dir:
            for p in list_images(real_mask_dir):
                m = read_mask(p)
                if m.shape != (size, size):
                    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                if mask_bbox(m) is not None:
                    self.real.append(m)
        if mode == "real" and not self.real:
            raise ValueError("mode='real' needs --real-mask-dir with at least one non-empty mask")

    def __call__(self):
        mode = self.mode
        if mode == "mixed":
            choices = (["real"] * 2 if self.real else []) + ["stroke", "blob"]
            mode = self.rng.choice(choices)
        for _ in range(20):
            if mode == "real":
                m = deform_real(self.rng.choice(self.real), self.rng, elastic=self.rng.uniform(0, 4))
            elif mode == "stroke":
                m = stroke_mask(self.size, self.rng)
            elif mode == "blob":
                m = blob_mask(self.size, self.rng)
            else:
                raise ValueError(f"unknown mask mode: {mode}")
            if (m > 0).sum() >= 16:
                return m, mode
        raise RuntimeError("could not generate a non-empty mask")
