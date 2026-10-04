"""Annotation logic for the UI (no Gradio here, so it can be unit-tested).

Annotations are stored as labelme JSON next to each image, so the UI and
labelme can be used interchangeably and every pipeline script reads them.
  - defect boxes:  shape_type "rectangle", label = defect type
  - part outline:  shape_type "polygon",   label = "roi"
"""
import json
from pathlib import Path

import cv2
import numpy as np

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


def add_polygon(ann, points, label=ROI_LABEL):
    if len(points) < 3:
        return False
    ann["shapes"].append(_shape(label, "polygon", points))
    return True


def remove_last(ann):
    if ann["shapes"]:
        ann["shapes"].pop()


def labels_in(folder):
    """Sorted defect labels used in rectangle shapes of all JSONs in folder."""
    out = set()
    for jf in Path(folder).glob("*.json"):
        for s in json.loads(jf.read_text(encoding="utf-8")).get("shapes", []):
            if s.get("shape_type") == "rectangle":
                out.add(s["label"])
    return sorted(out)


def summary(ann):
    boxes = sum(s["shape_type"] == "rectangle" for s in ann["shapes"])
    rois = sum(s["shape_type"] == "polygon" and s["label"] == ROI_LABEL for s in ann["shapes"])
    return boxes, rois


def view_window(h, w, zoom=1, center=None):
    """(x0, y0, vw, vh) of the visible part of an h x w image at a zoom level."""
    zoom = max(1, int(zoom))
    vw, vh = max(1, round(w / zoom)), max(1, round(h / zoom))
    cx, cy = center if center else (w / 2, h / 2)
    x0 = int(min(max(cx - vw / 2, 0), w - vw))
    y0 = int(min(max(cy - vh / 2, 0), h - vh))
    return x0, y0, vw, vh


def render(img, ann, pending=(), pending_kind=None, window=None):
    """Draw saved shapes (boxes red, ROI green) and the in-progress shape (yellow).

    window=(x0, y0, vw, vh) crops the result to the zoomed view; line widths
    are scaled to the view so they stay readable at any zoom.
    """
    out = img.copy()
    h, w = out.shape[:2]
    x0, y0, vw, vh = window or (0, 0, w, h)
    t = max(1, round(max(vw, vh) / 500))
    fs = max(0.4, max(vw, vh) / 1600)
    overlay = out.copy()
    for s in ann["shapes"]:
        pts = np.round(np.asarray(s["points"])).astype(np.int32)
        if s["shape_type"] == "rectangle":
            cv2.rectangle(out, tuple(pts[0]), tuple(pts[1]), (255, 40, 40), t)
            cv2.putText(out, s["label"], (int(pts[:, 0].min()), max(15, int(pts[:, 1].min()) - 2 * t)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 40, 40), t)
        elif s["shape_type"] == "polygon":
            cv2.fillPoly(overlay, [pts], (40, 200, 40))
            cv2.polylines(out, [pts], True, (40, 200, 40), t)
    out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0) if any(
        s["shape_type"] == "polygon" for s in ann["shapes"]) else out
    if pending:
        pts = np.round(np.asarray(pending)).astype(np.int32)
        if pending_kind == "polygon" and len(pts) > 1:
            cv2.polylines(out, [pts], False, (255, 210, 0), t)
        for p in pts:
            cv2.circle(out, tuple(p), 3 * t, (255, 210, 0), -1)
    return out[y0:y0 + vh, x0:x0 + vw]
