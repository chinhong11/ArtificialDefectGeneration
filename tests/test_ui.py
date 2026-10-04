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


def test_ui_builds_and_annotates_with_zoom(tmp_path):
    from ui import app
    for sub in ("good", "defect_raw"):
        (tmp_path / sub).mkdir()
    cv2.imwrite(str(tmp_path / "defect_raw/d.png"), np.full((1500, 2000, 3), 128, np.uint8))
    cv2.imwrite(str(tmp_path / "good/g.png"), np.full((1500, 2000, 3), 128, np.uint8))
    app.build(str(tmp_path))
    pr, name = str(tmp_path), "defect_raw/d.png"
    img, pend, view, _ = app.show_image(pr, name, "Defect box", "1×", {})
    assert img.shape == (1500, 2000, 3)
    # centre on the defect, zoom 4x, then click 2 corners on the displayed (upscaled) view
    _, pend, view, _ = app.on_click(pr, name, app.MOVE, "particle", pend, "1×", view, _Click(1596, 589))
    img, pend, view, _ = app.show_image(pr, name, "Defect box", "4×", view)
    assert img.shape == (1500, 2000, 3)                              # zoomed view displayed at full size
    x0, y0 = 1346, 401
    to_disp = lambda x, y: _Click((x - x0) * 4, (y - y0) * 4)      # noqa: E731
    _, pend, view, _ = app.on_click(pr, name, "Defect box", "particle", pend, "4×", view, to_disp(1583, 576))
    _, pend, view, hint = app.on_click(pr, name, "Defect box", "particle", pend, "4×", view, to_disp(1609, 602))
    pts = json.loads((tmp_path / "defect_raw/d.json").read_text())["shapes"][0]["points"]
    assert pts == [[1583.0, 576.0], [1609.0, 602.0]] and "1 box(es)" in hint
    # outline on the good image
    pend = []
    for x, y in [(100, 100), (1900, 100), (1900, 1400)]:
        _, pend, view, _ = app.on_click(pr, "good/g.png", "Part outline (good images)", "x", pend, "1×", {},
                                        _Click(x, y))
    app.finish_outline(pr, "good/g.png", pend, "1×", {})
    assert "1 with part outline" in app.status_text(pr)
