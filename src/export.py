"""Mask -> labels: YOLO boxes, COCO JSON (boxes + polygons).

One generated mask gives both detection and segmentation labels.

CLI (convert a folder of images + masks, e.g. generator output):
    python -m src.export --images outputs/scratch/images --masks outputs/scratch/masks \
        --class-id 0 --class-name scratch --yolo-dir outputs/scratch/labels \
        --coco outputs/scratch/coco.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.common import find_mask_for, list_images, read_mask


def mask_to_boxes(mask, min_area=16):
    """One (x0, y0, x1, y1) box per connected component (x1/y1 exclusive)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def yolo_lines(boxes, img_w, img_h, class_id):
    lines = []
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}")
    return lines


def write_yolo(path, boxes, img_w, img_h, class_id):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(yolo_lines(boxes, img_w, img_h, class_id)) + ("\n" if boxes else ""))


def mask_to_coco_annotations(mask, min_area=16):
    """[(bbox_xywh, polygon_list, area)] per connected component."""
    out = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]
        if not polys:
            continue
        x, y, w, h = (int(v) for v in stats[i, :4])
        out.append(([x, y, w, h], polys, area))
    return out


class CocoBuilder:
    def __init__(self, categories):
        """categories: {class_id: name}"""
        self.data = {"images": [], "annotations": [],
                     "categories": [{"id": int(k), "name": v} for k, v in sorted(categories.items())]}

    def add(self, file_name, mask, class_id, min_area=16):
        h, w = mask.shape[:2]
        img_id = len(self.data["images"]) + 1
        self.data["images"].append({"id": img_id, "file_name": file_name, "width": w, "height": h})
        for bbox, polys, area in mask_to_coco_annotations(mask, min_area):
            self.data["annotations"].append({
                "id": len(self.data["annotations"]) + 1, "image_id": img_id, "category_id": int(class_id),
                "bbox": bbox, "segmentation": polys, "area": area, "iscrowd": 0})

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--class-name", default="defect")
    ap.add_argument("--yolo-dir")
    ap.add_argument("--coco")
    ap.add_argument("--min-area", type=int, default=16)
    args = ap.parse_args()
    if not (args.yolo_dir or args.coco):
        ap.error("give --yolo-dir and/or --coco")

    coco = CocoBuilder({args.class_id: args.class_name}) if args.coco else None
    n = 0
    for img in list_images(args.images):
        mp = find_mask_for(img, args.masks)
        mask = read_mask(mp) if mp else None
        if mask is None:  # good image without defect -> empty label
            h, w = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE).shape
            mask = np.zeros((h, w), np.uint8)
        if args.yolo_dir:
            write_yolo(Path(args.yolo_dir) / f"{img.stem}.txt", mask_to_boxes(mask, args.min_area),
                       mask.shape[1], mask.shape[0], args.class_id)
        if coco:
            coco.add(img.name, mask, args.class_id, args.min_area)
        n += 1
    if coco:
        coco.save(args.coco)
    print(f"Exported labels for {n} images")


if __name__ == "__main__":
    main()
