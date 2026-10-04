"""Label import (YOLO det/seg, VOC, COCO, labelme) and automatic part area."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.label_import import format_report, has_defect_labels, import_labels  # noqa: E402
from src.roi import auto_part_mask  # noqa: E402

W, H = 200, 100


def _img(path):
    cv2.imwrite(str(path), np.full((H, W, 3), 128, np.uint8))
    return path


def _shapes(img_path):
    data = json.loads(Path(img_path).with_suffix(".json").read_text())
    return [(s["label"], s["shape_type"], [[round(v, 1) for v in p] for p in s["points"]]) for s in data["shapes"]]


def test_yolo_detection_with_class_names(tmp_path):
    img = _img(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    (tmp_path / "classes.txt").write_text("scratch\nparticle\n")
    r = import_labels([img], [tmp_path / "a.txt", tmp_path / "classes.txt"])
    assert r["boxes"] == 1 and r["labelled"] == ["a.png"]
    assert _shapes(img) == [("particle", "rectangle", [[80.0, 30.0], [120.0, 70.0]])]


def test_yolo_segmentation_and_data_yaml(tmp_path):
    img = _img(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.5\n")
    (tmp_path / "data.yaml").write_text("names:\n  0: crack\n")
    r = import_labels([img], [tmp_path / "a.txt", tmp_path / "data.yaml"])
    assert r["polygons"] == 1
    assert _shapes(img) == [("crack", "polygon", [[20.0, 10.0], [60.0, 10.0], [60.0, 50.0]])]


def test_yolo_without_names_and_empty_and_missing(tmp_path):
    a, b, c = _img(tmp_path / "a.png"), _img(tmp_path / "b.png"), _img(tmp_path / "c.png")
    (tmp_path / "a.txt").write_text("3 0.5 0.5 0.1 0.1\n")
    (tmp_path / "b.txt").write_text("")
    r = import_labels([a, b, c], [tmp_path / "a.txt", tmp_path / "b.txt"])
    assert _shapes(a)[0][0] == "class3"
    assert r["empty"] == ["b.png"] and r["missing"] == ["c.png"]
    assert not (tmp_path / "b.json").exists()
    assert "without a label file" in format_report(r)


def test_malformed_yolo_reports_error(tmp_path):
    img = _img(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("0 0.5 0.5\n")
    r = import_labels([img], [tmp_path / "a.txt"])
    assert r["errors"] and not (tmp_path / "a.json").exists()


def test_pascal_voc(tmp_path):
    img = _img(tmp_path / "a.jpg")
    (tmp_path / "a.xml").write_text("<annotation><object><name>dent</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
                                    "<xmax>50</xmax><ymax>60</ymax></bndbox></object></annotation>")
    import_labels([img], [tmp_path / "a.xml"])
    assert _shapes(img) == [("dent", "rectangle", [[10.0, 20.0], [50.0, 60.0]])]


def test_coco_bbox_and_polygon(tmp_path):
    a, b = _img(tmp_path / "a.png"), _img(tmp_path / "b.png")
    coco = {"images": [{"id": 1, "file_name": "imgs/a.png"}, {"id": 2, "file_name": "b.png"}],
            "categories": [{"id": 7, "name": "stain"}],
            "annotations": [{"image_id": 1, "category_id": 7, "bbox": [10, 10, 30, 20]},
                            {"image_id": 2, "category_id": 7, "bbox": [0, 0, 1, 1],
                             "segmentation": [[5, 5, 25, 5, 25, 25]]}]}
    (tmp_path / "instances.json").write_text(json.dumps(coco))
    r = import_labels([a, b], [tmp_path / "instances.json"])
    assert _shapes(a) == [("stain", "rectangle", [[10.0, 10.0], [40.0, 30.0]])]
    assert _shapes(b)[0][:2] == ("stain", "polygon") and r["polygons"] == 1


def test_labelme_passthrough_and_cli(tmp_path):
    import subprocess
    img = _img(tmp_path / "a.png")
    lab = tmp_path / "labels"
    lab.mkdir()
    (lab / "a.json").write_text(json.dumps({"shapes": [{"label": "pit", "shape_type": "rectangle",
                                                        "points": [[1, 2], [30, 40]]}]}))
    r = subprocess.run([sys.executable, "-m", "src.label_import", "--images", str(tmp_path), "--labels", str(lab)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert _shapes(img) == [("pit", "rectangle", [[1.0, 2.0], [30.0, 40.0]])]


def test_has_defect_labels(tmp_path):
    (tmp_path / "g1.txt").write_text("")
    (tmp_path / "g2.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    files = [tmp_path / "g1.txt", tmp_path / "g2.txt"]
    assert not has_defect_labels(files, "g1") and has_defect_labels(files, "g2")


def test_auto_part_mask_on_backlit_scene():
    """Bright vignetted background + a transparent part (only its outline/edges differ)."""
    h, w = 600, 800
    yy, xx = np.mgrid[0:h, 0:w]
    bg = 220 - 40 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    img = bg.copy()
    cv2.rectangle(img, (200, 200), (600, 400), 150, 6)        # part outline
    cv2.line(img, (250, 300), (550, 300), 170, 3)              # inner detail
    img = np.clip(img + np.random.default_rng(0).normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    m = auto_part_mask(img)
    assert m[300, 400] == 255                                  # inside the (transparent) part
    assert m[50, 50] == 0 and m[550, 750] == 0                 # background / vignetted corner
    inside = (m[210:390, 210:590] > 0).mean()
    assert inside > 0.95 and (m > 0).mean() < 0.25


pytest.importorskip("transformers")


def test_sam_script_uses_polygons_directly(tmp_path):
    import subprocess
    from tests.tiny_sam import build_tiny_sam
    sam = build_tiny_sam(tmp_path / "tiny_sam")
    d = tmp_path / "d"
    d.mkdir()
    _img(d / "x.png")
    (d / "x.json").write_text(json.dumps({"imagePath": "x.png", "imageHeight": H, "imageWidth": W, "shapes": [
        {"label": "crack", "shape_type": "polygon", "points": [[10, 10], [60, 10], [60, 60], [10, 60]]}]}))
    r = subprocess.run([sys.executable, "scripts/sam_box_to_mask.py", "--json-dir", str(d), "--out-dir",
                        str(tmp_path / "m"), "--model", sam], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = cv2.imread(str(tmp_path / "m/crack/x.png"), cv2.IMREAD_GRAYSCALE)
    assert m[30, 30] == 255 and m[80, 150] == 0
