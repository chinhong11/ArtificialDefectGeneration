"""Import existing defect labels in common formats.

Supported (auto-detected per file):
  - YOLO detection     <stem>.txt   "cls cx cy w h"            (normalised 0-1)
  - YOLO segmentation  <stem>.txt   "cls x1 y1 x2 y2 x3 y3 ..." (normalised polygon)
  - Pascal VOC         <stem>.xml   <object><name>/<bndbox>
  - COCO               *.json       {"images", "annotations", "categories"} (bbox and/or polygons)
  - labelme            <stem>.json  {"shapes": [...]}  (used as is)
Class names for YOLO come from classes.txt / obj.names / data.yaml (names); otherwise
"class0", "class1", ...  An empty YOLO .txt means "no defect on this image".

Everything is converted to labelme JSON next to each image (rectangles for boxes,
polygons for segmentation), which every other step of the pipeline reads.

CLI:
    python -m src.label_import --images data/p/defect_raw --labels data/p/labels
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from src.common import IMAGE_EXTS

LABEL_EXTS = {".txt", ".xml", ".json"}
CLASS_FILES = ("classes.txt", "obj.names", "data.yaml", "dataset.yaml")


def read_class_names(files):
    """{id: name} from classes.txt / obj.names / data.yaml among files (paths)."""
    by_name = {Path(f).name.lower(): Path(f) for f in files}
    for fn in ("classes.txt", "obj.names"):
        if fn in by_name:
            names = [l.strip() for l in by_name[fn].read_text(encoding="utf-8").splitlines() if l.strip()]
            return dict(enumerate(names))
    for fn in ("data.yaml", "dataset.yaml"):
        if fn in by_name:
            import yaml
            names = (yaml.safe_load(by_name[fn].read_text(encoding="utf-8")) or {}).get("names", [])
            return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    return {}


def _shape(label, kind, points):
    return {"label": str(label), "shape_type": kind, "points": [[float(x), float(y)] for x, y in points],
            "group_id": None, "flags": {}}


def parse_yolo(path, w, h, names):
    """YOLO detection or segmentation txt -> shapes. Raises ValueError on malformed lines."""
    shapes = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        v = line.split()
        if not v:
            continue
        cls = int(float(v[0]))
        nums = [float(t) for t in v[1:]]
        label = names.get(cls, f"class{cls}")
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            shapes.append(_shape(label, "rectangle", [((cx - bw / 2) * w, (cy - bh / 2) * h),
                                                       ((cx + bw / 2) * w, (cy + bh / 2) * h)]))
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            pts = [(nums[j] * w, nums[j + 1] * h) for j in range(0, len(nums), 2)]
            shapes.append(_shape(label, "polygon", pts))
        else:
            raise ValueError(f"{Path(path).name} line {i}: expected 5 values (box) or 7+ (polygon)")
    return shapes


def parse_voc(path):
    root = ET.parse(path).getroot()
    shapes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0, y0, x1, y1 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        shapes.append(_shape(obj.findtext("name", "defect").strip(), "rectangle", [(x0, y0), (x1, y1)]))
    return shapes


def parse_coco(data):
    """COCO dict -> {image basename: shapes}. Polygons preferred over bbox when present."""
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    files = {im["id"]: Path(im["file_name"]).name for im in data.get("images", [])}
    out = {name: [] for name in files.values()}
    for a in data.get("annotations", []):
        name = files.get(a.get("image_id"))
        if name is None:
            continue
        label = cats.get(a.get("category_id"), "defect")
        seg = a.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            for poly in seg:
                if len(poly) >= 6:
                    out[name].append(_shape(label, "polygon", list(zip(poly[0::2], poly[1::2]))))
        elif a.get("bbox"):
            x, y, bw, bh = a["bbox"]
            out[name].append(_shape(label, "rectangle", [(x, y), (x + bw, y + bh)]))
    return out


def _is_coco(data):
    return isinstance(data, dict) and "images" in data and "annotations" in data


def import_labels(images, label_files, keep_existing=False):
    """Write labelme JSON next to each image from the matching label file.

    images: image paths (already in their final folder). label_files: any mix of
    .txt/.xml/.json files (+ classes.txt / data.yaml). Returns a report dict.
    """
    label_files = [Path(f) for f in label_files]
    names = read_class_names(label_files)
    coco = {}
    by_stem = {}
    for f in label_files:
        if f.name.lower() in CLASS_FILES or f.suffix.lower() not in LABEL_EXTS:
            continue
        if f.suffix.lower() == ".json":
            data = json.loads(f.read_text(encoding="utf-8"))
            if _is_coco(data):
                coco.update(parse_coco(data))
                continue
        by_stem[f.stem] = f
    report = {"images": 0, "boxes": 0, "polygons": 0, "labelled": [], "empty": [], "missing": [], "errors": []}
    for img_path in map(Path, images):
        report["images"] += 1
        out = img_path.with_suffix(".json")
        if keep_existing and out.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            report["errors"].append(f"{img_path.name}: cannot read image")
            continue
        h, w = img.shape[:2]
        shapes = None
        try:
            if img_path.name in coco:
                shapes = coco[img_path.name]
            elif img_path.stem in by_stem:
                f = by_stem[img_path.stem]
                ext = f.suffix.lower()
                if ext == ".txt":
                    shapes = parse_yolo(f, w, h, names)
                elif ext == ".xml":
                    shapes = parse_voc(f)
                else:
                    shapes = json.loads(f.read_text(encoding="utf-8")).get("shapes", [])
        except (ValueError, KeyError, ET.ParseError, json.JSONDecodeError) as e:
            report["errors"].append(f"{img_path.name}: {e}")
            continue
        if shapes is None:
            report["missing"].append(img_path.name)
            continue
        shapes = [s for s in shapes if s.get("shape_type") in ("rectangle", "polygon")]
        if not shapes:
            report["empty"].append(img_path.name)
            if out.exists():
                out.unlink()
            continue
        out.write_text(json.dumps({"version": "5.0", "flags": {}, "shapes": shapes, "imagePath": img_path.name,
                                   "imageData": None, "imageHeight": h, "imageWidth": w}, indent=1),
                       encoding="utf-8")
        report["labelled"].append(img_path.name)
        report["boxes"] += sum(s["shape_type"] == "rectangle" for s in shapes)
        report["polygons"] += sum(s["shape_type"] == "polygon" for s in shapes)
    return report


def format_report(r):
    lines = [f"{r['images']} image(s): {len(r['labelled'])} labelled "
             f"({r['boxes']} box(es), {r['polygons']} polygon(s))"]
    if r["empty"]:
        lines.append(f"{len(r['empty'])} with an empty label file: {', '.join(r['empty'][:10])}")
    if r["missing"]:
        lines.append(f"{len(r['missing'])} without a label file (draw boxes in Annotate): "
                     f"{', '.join(r['missing'][:10])}")
    for e in r["errors"][:10]:
        lines.append(f"ERROR {e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Folder with the images (JSON is written next to them)")
    ap.add_argument("--labels", help="Folder with label files (default: same as --images)")
    ap.add_argument("--keep-existing", action="store_true", help="Don't overwrite existing labelme JSON")
    args = ap.parse_args()
    imgs = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    lab_dir = Path(args.labels or args.images)
    labels = sorted(p for p in lab_dir.rglob("*") if p.is_file() and
                    (p.suffix.lower() in LABEL_EXTS or p.name.lower() in CLASS_FILES))
    print(format_report(import_labels(imgs, labels, args.keep_existing)))


if __name__ == "__main__":
    main()
