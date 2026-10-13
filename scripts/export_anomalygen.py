"""Build an NVIDIA Cosmos-AnomalyGen dataset from defect images + labels and good images.

Standalone (numpy + Pillow only) so it runs inside the paidf-anomalygen container.

Input:
  --defects DIR   defect images plus, per image, one of:
                    <stem>.txt   YOLO box "cls cx cy w h" or YOLO-seg polygon "cls x1 y1 x2 y2 ..."
                    <stem>.json  labelme (rectangle / polygon shapes)
                  class names from classes.txt / obj.names / data.yaml (else class0, class1, ...)
  --masks DIR     optional: ready-made binary masks <stem>.png or <stem>_mask.png
                  (or <label>/<stem>.png); used instead of labels for that image
  --good DIR      good (defect-free) images; any label files there are ignored

Output (the layout AnomalyGen's validate_dataset expects):
  OUT/<Texture>/anomaly_image/<type>/image_001.png
  OUT/<Texture>/mask/<type>/image_001_mask.png      (0 = background, 255 = defect)
  OUT/<Texture>/clean_image/good_001.png
  OUT/defect_spec.jsonl                              (one line per defect type)

Example (inside the container):
  python3 /work/home/export_anomalygen.py --defects /work/home/raw/defect \
      --good /work/home/raw/good --out /work/home/datasets/connector --texture Connector \
      --roi-prompt "the transparent plastic connector part, excluding the background"
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MIN_PER_TYPE = 3  # AnomalyGen needs >= 3 per type for a meaningful nn_score


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_class_names(d):
    d = Path(d)
    for fn in ("classes.txt", "obj.names"):
        if (d / fn).exists():
            names = [l.strip() for l in (d / fn).read_text(encoding="utf-8").splitlines() if l.strip()]
            return dict(enumerate(names))
    for fn in ("data.yaml", "dataset.yaml"):
        if (d / fn).exists():
            text = (d / fn).read_text(encoding="utf-8")
            try:
                import yaml
                names = (yaml.safe_load(text) or {}).get("names", [])
            except ImportError:  # minimal fallback: "names: [a, b]" or "  0: a" lines
                m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
                names = ([n.strip().strip("'\"") for n in m.group(1).split(",")] if m else
                         {int(k): v.strip().strip("'\"") for k, v in re.findall(r"^\s+(\d+):\s*(.+)$", text, re.M)})
            return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    return {}


def safe_name(s):
    """AnomalyGen joins texture and type with '+', so keep names to [A-Za-z0-9_-]."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(s).strip()).strip("_")
    return s or "defect"


def shapes_from_yolo(path, w, h, names):
    shapes = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        v = line.split()
        if not v:
            continue
        cls, nums = int(float(v[0])), [float(t) for t in v[1:]]
        label = names.get(cls, f"class{cls}")
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            shapes.append((label, "rectangle", [((cx - bw / 2) * w, (cy - bh / 2) * h),
                                                 ((cx + bw / 2) * w, (cy + bh / 2) * h)]))
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            shapes.append((label, "polygon", [(nums[j] * w, nums[j + 1] * h) for j in range(0, len(nums), 2)]))
        else:
            raise ValueError(f"{Path(path).name} line {i}: expected 5 values (box) or 7+ (polygon)")
    return shapes


def shapes_from_labelme(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(s.get("label", "defect"), s["shape_type"], [tuple(p) for p in s["points"]])
            for s in data.get("shapes", []) if s.get("shape_type") in ("rectangle", "polygon")]


def rasterize(shapes, w, h):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for _, kind, pts in shapes:
        if kind == "rectangle":
            (x0, y0), (x1, y1) = pts[0], pts[1]
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
        elif len(pts) >= 3:
            d.polygon([tuple(p) for p in pts], fill=255)
    return m


def find_mask(masks_dir, stem):
    """Ready-made masks: {label: path}. <stem>.png / <stem>_mask.png -> label None."""
    if not masks_dir:
        return {}
    md = Path(masks_dir)
    for name in (f"{stem}.png", f"{stem}_mask.png"):
        if (md / name).exists():
            return {None: md / name}
    return {sub.name: sub / f"{stem}.png" for sub in sorted(md.iterdir())
            if sub.is_dir() and (sub / f"{stem}.png").exists()}


def load_binary(path, size):
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return Image.fromarray(np.where(np.asarray(m) > 127, 255, 0).astype(np.uint8))


def export(defects, good, out, texture="Part", masks=None, default_label="defect",
           spatial="text", roi_prompt=""):
    out, texture = Path(out), safe_name(texture)
    names = read_class_names(defects)
    per_type, report = {}, {"skipped": [], "box_only": [], "errors": []}
    for img_path in list_images(defects):
        img = Image.open(img_path)
        w, h = img.size
        per_label = {}
        try:
            ready = find_mask(masks, img_path.stem)
            if ready:
                for label, mp in ready.items():
                    per_label[label or default_label] = load_binary(mp, (w, h))
            else:
                txt, js = img_path.with_suffix(".txt"), img_path.with_suffix(".json")
                shapes = (shapes_from_yolo(txt, w, h, names) if txt.exists() else
                          shapes_from_labelme(js) if js.exists() else None)
                if shapes is None:
                    report["skipped"].append(f"{img_path.name}: no label file or mask")
                    continue
                if shapes and all(k == "rectangle" for _, k, _ in shapes):
                    report["box_only"].append(img_path.name)
                for label in dict.fromkeys(s[0] for s in shapes):
                    per_label[label] = rasterize([s for s in shapes if s[0] == label], w, h)
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as e:
            report["errors"].append(f"{img_path.name}: {e}")
            continue
        per_label = {k: v for k, v in per_label.items() if np.asarray(v).any()}
        if not per_label:
            report["skipped"].append(f"{img_path.name}: empty label (no defect)")
            continue
        rgb = img.convert("RGB")
        for label, mask in per_label.items():
            t = safe_name(label)
            n = per_type.get(t, 0) + 1
            per_type[t] = n
            (out / texture / "anomaly_image" / t).mkdir(parents=True, exist_ok=True)
            (out / texture / "mask" / t).mkdir(parents=True, exist_ok=True)
            rgb.save(out / texture / "anomaly_image" / t / f"image_{n:03d}.png")
            mask.save(out / texture / "mask" / t / f"image_{n:03d}_mask.png")

    clean = out / texture / "clean_image"
    clean.mkdir(parents=True, exist_ok=True)
    n_good = 0
    for n_good, p in enumerate(list_images(good) if good else [], 1):
        Image.open(p).convert("RGB").save(clean / f"good_{n_good:03d}.png")

    with open(out / "defect_spec.jsonl", "w", encoding="utf-8") as f:
        for t in sorted(per_type):
            f.write(json.dumps({"defect_type": f"{texture}+{t}", "spatial_dependency": spatial,
                                "roi_prompt_defect_location": roi_prompt if spatial == "text" else ""}) + "\n")
    report.update(per_type=per_type, good=n_good, texture=texture)
    return report


def format_report(r, out):
    lines = [f"Wrote {out}", f"  texture: {r['texture']}   good images: {r['good']}"]
    for t, n in sorted(r["per_type"].items()):
        warn = f"   <-- fewer than {MIN_PER_TYPE}; AnomalyGen's quality score needs >= {MIN_PER_TYPE}" \
            if n < MIN_PER_TYPE else ""
        lines.append(f"  {r['texture']}+{t}: {n} image(s) + mask(s){warn}")
    if not r["per_type"]:
        lines.append("  ERROR: no defect image with a label/mask was exported")
    if r["good"] == 0:
        lines.append("  WARNING: no good images (clean_image/ is empty)")
    if r["box_only"]:
        lines.append(f"  NOTE: {len(r['box_only'])} image(s) have boxes only -> masks are filled rectangles "
                     "(include background). Polygons or SAM masks give better results.")
    for s in r["skipped"][:20]:
        lines.append(f"  skipped {s}")
    for e in r["errors"][:20]:
        lines.append(f"  ERROR {e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defects", required=True, help="Folder with defect images + label files")
    ap.add_argument("--good", help="Folder with good images")
    ap.add_argument("--masks", help="Optional folder with ready-made binary masks")
    ap.add_argument("--out", required=True, help="Output dataset folder (must be empty or not exist)")
    ap.add_argument("--texture", default="Part", help="Part name, e.g. Connector")
    ap.add_argument("--label", default="defect", help="Defect type for masks that carry no label")
    ap.add_argument("--spatial", choices=["text", "free"], default="text",
                    help="text = defects only inside the area described by --roi-prompt; free = anywhere")
    ap.add_argument("--roi-prompt", default="the inspected part, excluding the background",
                    help="Where defects may appear (used when --spatial text)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; choose a new --out folder")
    r = export(args.defects, args.good, out, args.texture, args.masks, args.label, args.spatial, args.roi_prompt)
    print(format_report(r, out))
    sys.exit(0 if r["per_type"] and not r["errors"] else 1)


if __name__ == "__main__":
    main()
