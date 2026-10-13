"""Export to the NVIDIA Cosmos-AnomalyGen dataset layout."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_anomalygen import export  # noqa: E402


def _img(path, size=(200, 100)):
    Image.fromarray(np.full((size[1], size[0], 3), 128, np.uint8)).save(path)
    return path


def test_export_layout_boxes_polygons_masks(tmp_path):
    d, g, m = tmp_path / "defect", tmp_path / "good", tmp_path / "masks"
    for p in (d, g, m):
        p.mkdir()
    _img(d / "a.webp")
    (d / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n1 0.1 0.1 0.3 0.1 0.3 0.5\n")   # box + polygon, 2 classes
    _img(d / "b.jpg")
    (d / "b.json").write_text(json.dumps({"shapes": [
        {"label": "particle", "shape_type": "rectangle", "points": [[10, 10], [30, 30]]}]}))
    _img(d / "c.png")
    Image.fromarray(np.pad(np.full((10, 10), 255, np.uint8), ((40, 50), (90, 100)))).save(m / "c_mask.png")
    _img(d / "empty.png")
    (d / "empty.txt").write_text("")
    _img(d / "nolabel.png")
    (d / "classes.txt").write_text("particle\nscratch mark\n")
    _img(g / "g1.webp")
    (g / "g1.txt").write_text("0 0.5 0.5 0.1 0.1\n")                               # ignored

    r = export(d, g, tmp_path / "out", texture="Connector", masks=m, default_label="particle",
               roi_prompt="the connector")
    base = tmp_path / "out" / "Connector"
    assert r["per_type"] == {"particle": 3, "scratch_mark": 1} and r["good"] == 1
    for n in (1, 2, 3):
        img = Image.open(base / f"anomaly_image/particle/image_{n:03d}.png")
        mask = np.asarray(Image.open(base / f"mask/particle/image_{n:03d}_mask.png"))
        assert img.mode == "RGB" and mask.shape == (100, 200) and set(np.unique(mask)) == {0, 255}
    a_box = np.asarray(Image.open(base / "mask/particle/image_001_mask.png"))
    assert a_box[50, 100] == 255 and a_box[5, 5] == 0                            # box filled, polygon not
    a_poly = np.asarray(Image.open(base / "mask/scratch_mark/image_001_mask.png"))
    assert a_poly[30, 50] == 255 and a_poly[50, 100] == 0
    assert (base / "clean_image/good_001.png").exists()
    spec = [json.loads(l) for l in (tmp_path / "out/defect_spec.jsonl").read_text().splitlines()]
    assert spec == [{"defect_type": "Connector+particle", "spatial_dependency": "text",
                     "roi_prompt_defect_location": "the connector"},
                    {"defect_type": "Connector+scratch_mark", "spatial_dependency": "text",
                     "roi_prompt_defect_location": "the connector"}]
    assert any("nolabel.png" in s for s in r["skipped"]) and any("empty.png" in s for s in r["skipped"])
    assert r["box_only"] == ["b.jpg"]


def test_cli_warns_on_few_images_and_refuses_nonempty_out(tmp_path):
    d = tmp_path / "defect"
    d.mkdir()
    _img(d / "a.png")
    (d / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    cmd = [sys.executable, "scripts/export_anomalygen.py", "--defects", str(d), "--out", str(tmp_path / "o")]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "Part+class0: 1 image(s)" in r.stdout and "fewer than 3" in r.stdout and "no good images" in r.stdout
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "not empty" in r.stderr
