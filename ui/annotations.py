"""Annotation logic for the UI (no Gradio here, so it can be unit-tested).

Annotations are stored as labelme JSON next to each image, so the UI, labelme
and imported YOLO/VOC/COCO labels (src/label_import.py) are interchangeable and
every pipeline script reads them.
  - defect boxes:     shape_type "rectangle", label = defect type
  - defect polygons:  shape_type "polygon",   label = defect type (e.g. from YOLO-seg / COCO)
  - "roi" polygons are part outlines (optional, command-line workflow only)
"""
import json
from pathlib import Path


ROI_LABEL = "roi"


def json_path(image_path):
    return Path(image_path).with_suffix(".json")


def load(image_path, img_shape):
    p = json_path(image_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    h, w = img_shape[:2]
    return {"version": "5.0", "flags": {}, "shapes": [], "imagePath": Path(image_path).name,
            "imageData": None, "imageHeight": int(h), "imageWidth": int(w)}


def save(image_path, ann):
    p = json_path(image_path)
    if ann["shapes"]:
        p.write_text(json.dumps(ann, indent=1), encoding="utf-8")
    elif p.exists():
        p.unlink()


def _shape(label, kind, points):
    return {"label": label, "shape_type": kind, "points": [[float(x), float(y)] for x, y in points],
            "group_id": None, "flags": {}}


def add_box(ann, label, p1, p2, min_size=2):
    (x0, y0), (x1, y1) = p1, p2
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return False
    ann["shapes"].append(_shape(label.strip() or "defect", "rectangle", [(x0, y0), (x1, y1)]))
    return True


def labels_in(folder):
    """Sorted defect labels (boxes and non-ROI polygons) of all JSONs in folder."""
    out = set()
    for jf in Path(folder).glob("*.json"):
        for s in json.loads(jf.read_text(encoding="utf-8")).get("shapes", []):
            if s.get("shape_type") == "rectangle" or (s.get("shape_type") == "polygon" and s["label"] != ROI_LABEL):
                out.add(s["label"])
    return sorted(out)


def summary(ann):
    """(boxes, defect polygons) in an annotation."""
    boxes = sum(s["shape_type"] == "rectangle" for s in ann["shapes"])
    polys = sum(s["shape_type"] == "polygon" and s["label"] != ROI_LABEL for s in ann["shapes"])
    return boxes, polys
