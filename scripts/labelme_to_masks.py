"""Convert labelme JSON annotations to binary mask PNGs (one folder per label).

Supported shape types: polygon, rectangle, circle, linestrip/line (drawn with
--line-width), point (small disc).

Example:
    python scripts/labelme_to_masks.py \
        --json-dir data/metal_part/defect_raw \
        --out-dir data/metal_part/masks

Writes   data/metal_part/masks/<label>/<image_stem>.png
Optionally copies the images to --copy-images-to/<label>/ so the folder layout
matches what src/data/prepare_crops.py expects.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import write_mask  # noqa: E402


def shapes_to_masks(data, line_width=5, point_radius=3, labels=None):
    """Return {label: uint8 mask} for one labelme JSON dict."""
    h, w = int(data["imageHeight"]), int(data["imageWidth"])
    masks = {}
    for shape in data.get("shapes", []):
        label = shape["label"].strip()
        if labels and label not in labels:
            continue
        pts = np.asarray(shape["points"], dtype=np.float64)
        m = masks.setdefault(label, np.zeros((h, w), np.uint8))
        kind = shape.get("shape_type") or "polygon"
        ipts = np.round(pts).astype(np.int32)
        if kind == "polygon":
            cv2.fillPoly(m, [ipts], 255)
        elif kind == "rectangle":
            (x0, y0), (x1, y1) = ipts[0], ipts[1]
            cv2.rectangle(m, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), 255, -1)
        elif kind == "circle":
            r = int(round(np.linalg.norm(pts[1] - pts[0])))
            cv2.circle(m, tuple(ipts[0]), r, 255, -1)
        elif kind in ("linestrip", "line"):
            cv2.polylines(m, [ipts], False, 255, thickness=line_width)
        elif kind == "point":
            cv2.circle(m, tuple(ipts[0]), point_radius, 255, -1)
        else:
            print(f"  skip unsupported shape type: {kind}")
    return masks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True, help="Folder with labelme .json files")
    ap.add_argument("--out-dir", required=True, help="Output mask root (one subfolder per label)")
    ap.add_argument("--labels", nargs="*", help="Only convert these labels")
    ap.add_argument("--line-width", type=int, default=5, help="Thickness for line/linestrip shapes")
    ap.add_argument("--copy-images-to", help="Also copy source images to <this>/<label>/")
    args = ap.parse_args()

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    if not json_files:
        sys.exit(f"No .json files in {args.json_dir}")
    counts = {}
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        masks = shapes_to_masks(data, args.line_width, labels=set(args.labels or []))
        img_path = jf.parent / data.get("imagePath", jf.stem + ".png")
        for label, m in masks.items():
            write_mask(Path(args.out_dir) / label / f"{jf.stem}.png", m)
            counts[label] = counts.get(label, 0) + 1
            if args.copy_images_to:
                if img_path.exists():
                    dst = Path(args.copy_images_to) / label / (jf.stem + img_path.suffix)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dst)
                else:
                    print(f"  image not found for {jf.name}: {img_path}")
    for label, n in sorted(counts.items()):
        print(f"{label}: {n} masks")


if __name__ == "__main__":
    main()
