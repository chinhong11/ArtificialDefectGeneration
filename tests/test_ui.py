"""UI tests: annotation logic (no Gradio needed) + UI handlers on a tiny project."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import annotations as A  # noqa: E402


def test_view_window_clamps_and_zooms():
    assert A.view_window(1500, 2000, 1) == (0, 0, 2000, 1500)
    assert A.view_window(1500, 2000, 4) == (750, 562, 500, 375)                     # centred
    assert A.view_window(1500, 2000, 4, (1596, 589)) == (1346, 401, 500, 375)
    assert A.view_window(1500, 2000, 4, (10, 10)) == (0, 0, 500, 375)               # clamped top-left
    assert A.view_window(1500, 2000, 4, (1990, 1490)) == (1500, 1125, 500, 375)     # clamped bottom-right


def test_box_polygon_save_load_roundtrip(tmp_path):
    img_path = tmp_path / "a.png"
    cv2.imwrite(str(img_path), np.full((100, 200, 3), 128, np.uint8))
    ann = A.load(img_path, (100, 200, 3))
    assert ann["shapes"] == [] and ann["imageWidth"] == 200
    assert A.add_box(ann, "particle", (50, 40), (10, 20))          # corners in any order
    assert not A.add_box(ann, "particle", (5, 5), (6, 6))          # too small
    assert not A.add_polygon(ann, [(0, 0), (10, 0)])               # < 3 points
    assert A.add_polygon(ann, [(0, 0), (100, 0), (100, 90)])
    A.save(img_path, ann)
    ann2 = A.load(img_path, (1, 1))
    assert ann2["shapes"][0]["points"] == [[10.0, 20.0], [50.0, 40.0]]
    assert A.summary(ann2) == (1, 1) and A.labels_in(tmp_path) == ["particle"]
    A.remove_last(ann2)
    A.remove_last(ann2)
    A.save(img_path, ann2)
    assert not A.json_path(img_path).exists()                       # empty -> JSON removed


def test_render_crops_to_window():
    img = np.zeros((1500, 2000, 3), np.uint8)
    ann = {"shapes": [{"label": "d", "shape_type": "rectangle", "points": [[1583, 576], [1609, 602]]}]}
    out = A.render(img, ann, window=(1346, 401, 500, 375))
    assert out.shape == (375, 500, 3)
    assert out[576 - 401, 1590 - 1346].any()                        # box edge visible inside the view


def test_annotation_json_works_with_pipeline(tmp_path):
    """JSON written by the UI is read correctly by labelme_to_masks."""
    import subprocess
    img_path = tmp_path / "d.png"
    cv2.imwrite(str(img_path), np.full((100, 200, 3), 128, np.uint8))
    ann = A.load(img_path, (100, 200, 3))
    A.add_box(ann, "particle", (20, 30), (60, 70))
    A.save(img_path, ann)
    r = subprocess.run([sys.executable, "scripts/labelme_to_masks.py", "--json-dir", str(tmp_path),
                        "--out-dir", str(tmp_path / "m")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = cv2.imread(str(tmp_path / "m/particle/d.png"), cv2.IMREAD_GRAYSCALE)
    assert m[50, 40] == 255 and m[10, 10] == 0


gr = pytest.importorskip("gradio")


class _Click:
    def __init__(self, x, y):
        self.index = [x, y]


def test_ui_builds_boxes_and_outline(tmp_path):
    from ui import app
    for sub in ("good", "defect_raw"):
        (tmp_path / sub).mkdir()
    cv2.imwrite(str(tmp_path / "defect_raw/d.png"), np.full((1500, 2000, 3), 128, np.uint8))
    cv2.imwrite(str(tmp_path / "good/g.png"), np.full((1500, 2000, 3), 128, np.uint8))
    app.build(str(tmp_path))
    pr, name = str(tmp_path), "defect_raw/d.png"

    # defect image -> annotator panel visible, empty boxes
    ann_v, _, _, _, hint, box_panel, outline_panel = app.show_image(pr, name, "particle", "1×", {})
    assert ann_v["boxes"] == [] and "0 defect box(es)" in hint

    # drag-drawn box arrives from the annotator (with its own default label) -> saved with textbox label
    v = {"image": "/tmp/gradio/abc/d.png", "boxes": [{"xmin": 1583, "ymin": 576, "xmax": 1609, "ymax": 602,
                                                      "label": app.NEW_TAG}]}
    app.save_boxes(pr, name, v, "particle")
    # moving/resizing it keeps its saved label even if the textbox label changed
    v["boxes"][0].update(xmin=1580, xmax=1612)
    app.save_boxes(pr, name, v, "scratch")
    shapes = json.loads((tmp_path / "defect_raw/d.json").read_text())["shapes"]
    assert [(s["label"], s["points"]) for s in shapes] == [("particle", [[1580.0, 576.0], [1612.0, 602.0]])]
    # event from another image is ignored (stale event while switching images)
    app.save_boxes(pr, name, {"image": "/tmp/x/other.png", "boxes": []}, "particle")
    assert (tmp_path / "defect_raw/d.json").exists()
    # relabel all
    app.relabel_all(pr, name, "scratch", "1×", {})
    assert json.loads((tmp_path / "defect_raw/d.json").read_text())["shapes"][0]["label"] == "scratch"
    # reload shows the saved box
    ann_v = app.show_image(pr, name, "particle", "1×", {})[0]
    assert ann_v["boxes"][0]["label"] == "scratch" and ann_v["boxes"][0]["xmin"] == 1580
    # deleting all boxes removes the JSON
    app.save_boxes(pr, name, {"image": "d.png", "boxes": []}, "particle")
    assert not (tmp_path / "defect_raw/d.json").exists()

    # good image -> outline by clicks, with zoom + move view
    g = "good/g.png"
    _, img, pend, view, _, _, _ = app.show_image(pr, g, "particle", "1×", {})
    assert img.shape == (1500, 2000, 3)
    for x, y in [(100, 100), (1900, 100), (1900, 1400)]:
        _, pend, view, _ = app.on_click(pr, g, app.ADD_POINT, pend, "1×", view, _Click(x, y))
    app.finish_outline(pr, g, pend, "1×", view)
    assert "1 with part outline" in app.status_text(pr)
    _, pend, view, _ = app.on_click(pr, g, app.MOVE, [], "1×", view, _Click(1596, 589))
    _, img, pend, view, _, _, _ = app.show_image(pr, g, "particle", "4×", view)
    assert img.shape == (1500, 2000, 3)                              # zoomed view displayed at full size
    # click at displayed (0, 0) of the 4x view centred on (1596, 589) -> image (1346, 401)
    _, pend, view, _ = app.on_click(pr, g, app.ADD_POINT, [], "4×", view, _Click(0, 0))
    assert pend == [(1346, 401)]
