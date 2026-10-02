"""Turn rough box annotations into pixel masks with Segment Anything (SAM).

Draw only rectangles in labelme (fast), then run this to get a pixel mask per
box. ALWAYS review the result (scripts/make_grid.py --overlay) — SAM can grab
too much or too little on low-contrast defects; fix bad ones by hand.

Example:
    python scripts/sam_box_to_mask.py \
        --json-dir data/metal_part/defect_raw \
        --out-dir data/metal_part/masks --copy-images-to data/metal_part/defect

Writes data/metal_part/masks/<label>/<image_stem>.png
(and copies the image to data/metal_part/defect/<label>/)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import dilate, write_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", required=True, help="labelme .json files containing rectangle shapes")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="facebook/sam-vit-base", help="e.g. facebook/sam-vit-large / -huge")
    ap.add_argument("--box-margin", type=int, default=4, help="Mask is clipped to the box grown by this many px")
    ap.add_argument("--fallback-box", action="store_true", help="Use the filled box if SAM returns an empty mask")
    ap.add_argument("--copy-images-to", help="Also copy source images to <this>/<label>/")
    args = ap.parse_args()

    import torch
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device).eval()

    for jf in sorted(Path(args.json_dir).glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        img_path = jf.parent / data["imagePath"]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        by_label = {}
        for s in data.get("shapes", []):
            if s.get("shape_type") != "rectangle":
                continue
            (x0, y0), (x1, y1) = s["points"][0], s["points"][1]
            by_label.setdefault(s["label"].strip(), []).append(
                [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        if not by_label:
            continue

        for label, boxes in by_label.items():
            inputs = processor(image, input_boxes=[boxes], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, multimask_output=False)
            masks = processor.image_processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
            # masks: (num_boxes, 1, H, W) bool
            final = np.zeros((h, w), np.uint8)
            for box, m in zip(boxes, masks[:, 0].numpy()):
                bx0, by0, bx1, by1 = (int(round(v)) for v in box)
                clip = np.zeros((h, w), np.uint8)
                clip[max(0, by0):by1 + 1, max(0, bx0):bx1 + 1] = 255
                clip = dilate(clip, args.box_margin)
                part = np.where(m & (clip > 0), 255, 0).astype(np.uint8)
                if part.sum() == 0 and args.fallback_box:
                    part[max(0, by0):by1 + 1, max(0, bx0):bx1 + 1] = 255
                final = np.maximum(final, part)
            write_mask(Path(args.out_dir) / label / f"{jf.stem}.png", final)
            if args.copy_images_to:
                dst = Path(args.copy_images_to) / label / (jf.stem + img_path.suffix)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst)
            print(f"{jf.stem} [{label}]: {len(boxes)} box(es), {int((final > 0).sum())} px")


if __name__ == "__main__":
    main()
