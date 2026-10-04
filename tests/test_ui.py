"""UI tests: annotation helpers (no Gradio needed) + UI handlers on a tiny project."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import annotations as A  # noqa: E402


def test_box_save_load_roundtrip(tmp_path):
    img_path = tmp_path / "a.png"
    cv2.imwrite(str(img_path), np.full((100, 200, 3), 128, np.uint8))
    ann = A.load(img_path, (100, 200, 3))
    assert ann["shapes"] == [] and ann["imageWidth"] == 200
    assert A.add_box(ann, "particle", (50, 40), (10, 20))          # corners in any order
    assert not A.add_box(ann, "particle", (5, 5), (6, 6))          # too small
    ann["shapes"].append({"label": "crack", "shape_type": "polygon", "points": [[0, 0], [9, 0], [9, 9]]})
    A.save(img_path, ann)
    ann2 = A.load(img_path, (1, 1))
    assert ann2["shapes"][0]["points"] == [[10.0, 20.0], [50.0, 40.0]]
    assert A.summary(ann2) == (1, 1) and A.labels_in(tmp_path) == ["crack", "particle"]
    ann2["shapes"] = []
    A.save(img_path, ann2)
    assert not A.json_path(img_path).exists()                       # empty -> JSON removed


gr = pytest.importorskip("gradio")


def _project(tmp_path):
    for sub in ("good", "defect_raw"):
        (tmp_path / sub).mkdir(exist_ok=True)
    src = tmp_path / "upload"
    src.mkdir()
    for n in ("d1", "d2", "d3"):
        cv2.imwrite(str(src / f"{n}.png"), np.full((1500, 2000, 3), 128, np.uint8))
    (src / "d1.txt").write_text("0 0.798 0.396 0.013 0.017\n")              # YOLO box
    (src / "d2.txt").write_text("1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")      # YOLO-seg polygon
    (src / "classes.txt").write_text("particle\ncrack\n")                  # d3 has no label file
    g = src / "good"
    g.mkdir()
    img = np.full((600, 800, 3), 220, np.uint8)
    cv2.rectangle(img, (200, 200), (600, 400), (150, 150, 150), 6)
    cv2.imwrite(str(g / "g1.png"), img)
    (g / "g1.txt").write_text("")
    return src


class _File:
    def __init__(self, p):
        self.name = str(p)


def test_ui_upload_with_labels_annotate_and_roi(tmp_path):
    from ui import app
    src = _project(tmp_path)
    app.build(str(tmp_path))
    pr = str(tmp_path)

    msg, status = app.upload_good(pr, [str(src / "good/g1.png"), str(src / "good/g1.txt")])
    assert "Added 1 good" in msg and "Warning" not in msg
    files = [str(src / f) for f in ("d1.png", "d2.png", "d3.png", "d1.txt", "d2.txt", "classes.txt")]
    msg, status, _ = app.upload_defects(pr, files)
    assert "3 image(s): 2 labelled (1 box(es), 1 polygon(s))" in msg and "d3.png" in msg
    assert "particle, crack" in status or "crack, particle" in status
    assert app.image_choices(pr) == ["defect_raw/d1.png", "defect_raw/d2.png", "defect_raw/d3.png"]

    # imported YOLO box shows in the annotator with its class name
    val, hint = app.show_image(pr, "defect_raw/d1.png")
    assert len(val["boxes"]) == 1 and val["boxes"][0]["label"] == "particle"
    # imported polygon is shown as its bounding box; an unchanged save keeps the polygon
    val, hint = app.show_image(pr, "defect_raw/d2.png")
    assert val["boxes"][0]["label"].startswith("crack") and "1 polygon" in hint
    val["image"] = "/tmp/gradio/xyz/d2.png"
    app.save_boxes(pr, "defect_raw/d2.png", val, "defect")
    shapes = json.loads((tmp_path / "defect_raw/d2.json").read_text())["shapes"]
    assert [s["shape_type"] for s in shapes] == ["polygon"]
    # deleting the box in the annotator deletes the polygon
    app.save_boxes(pr, "defect_raw/d2.png", {"image": "d2.png", "boxes": []}, "defect")
    assert not (tmp_path / "defect_raw/d2.json").exists()

    # unlabeled image: draw a box -> textbox label; move it -> label kept; stale event ignored
    v = {"image": "/tmp/gradio/abc/d3.png", "boxes": [{"xmin": 1583, "ymin": 576, "xmax": 1609, "ymax": 602,
                                                       "label": app.NEW_TAG}]}
    app.save_boxes(pr, "defect_raw/d3.png", v, "particle")
    v["boxes"][0].update(xmin=1580, xmax=1612)
    app.save_boxes(pr, "defect_raw/d3.png", v, "scratch")
    app.save_boxes(pr, "defect_raw/d3.png", {"image": "/tmp/x/other.png", "boxes": []}, "particle")
    shapes = json.loads((tmp_path / "defect_raw/d3.json").read_text())["shapes"]
    assert [(s["label"], s["points"]) for s in shapes] == [("particle", [[1580.0, 576.0], [1612.0, 602.0]])]
    app.relabel_all(pr, "defect_raw/d3.png", "scratch")
    assert json.loads((tmp_path / "defect_raw/d3.json").read_text())["shapes"][0]["label"] == "scratch"

    # automatic part area on the good image
    roi_dir, cov = app.build_auto_roi(Path(pr), 6)
    m = cv2.imread(str(roi_dir / "g1.png"), cv2.IMREAD_GRAYSCALE)
    assert m[300, 400] == 255 and m[20, 20] == 0 and "g1.png" in cov
    assert len(app.preview_roi(pr, 6)) == 1


def test_upload_zip_and_warn_on_labelled_good_image(tmp_path):
    import zipfile
    from ui import app
    src = _project(tmp_path)
    z = tmp_path / "defects.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in ("d1.png", "d1.txt", "classes.txt"):
            zf.write(src / f, f"dataset/{f}")
    msg, _, _ = app.upload_defects(str(tmp_path), [str(z)])
    assert "1 labelled (1 box(es)" in msg
    (src / "good/g1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    msg, _ = app.upload_good(str(tmp_path), [str(src / "good/g1.png"), str(src / "good/g1.txt")])
    assert "Warning" in msg and "g1.png" in msg
