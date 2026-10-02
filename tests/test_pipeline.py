"""CPU tests. Run:  python -m pytest -q tests

The diffusion tests use a tiny random model built locally (tests/tiny_sd.py),
so they check the code runs end-to-end, not image quality.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_mask, write_mask, write_rgb  # noqa: E402
from src.export import CocoBuilder, mask_to_boxes, yolo_lines  # noqa: E402
from src.mask_gen import MaskGenerator, blob_mask, deform_real, stroke_mask  # noqa: E402


def make_dataset(root, n_defect=3, n_good=3, h=160, w=200):
    rng = np.random.default_rng(0)
    for i in range(n_good):
        img = (rng.normal(128, 10, (h, w, 3))).clip(0, 255).astype(np.uint8)
        write_rgb(root / "good" / f"g{i}.png", img)
    for i in range(n_defect):
        img = (rng.normal(128, 10, (h, w, 3))).clip(0, 255).astype(np.uint8)
        m = np.zeros((h, w), np.uint8)
        cv2.line(m, (40 + 10 * i, 50), (90 + 10 * i, 80), 255, 4)
        img[m > 0] = 30
        write_rgb(root / "defect" / "scratch" / f"d{i}.png", img)
        write_mask(root / "masks" / "scratch" / f"d{i}.png", m)
    return root


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


# ---------- CPU-only parts ----------

def test_mask_to_boxes_and_yolo():
    m = np.zeros((100, 200), np.uint8)
    m[10:20, 30:50] = 255
    m[60:90, 100:110] = 255
    boxes = sorted(mask_to_boxes(m))
    assert boxes == [(30, 10, 50, 20), (100, 60, 110, 90)]
    line = yolo_lines([boxes[0]], 200, 100, 3)[0].split()
    assert line[0] == "3" and abs(float(line[1]) - 0.2) < 1e-6 and abs(float(line[3]) - 0.1) < 1e-6


def test_coco_builder(tmp_path):
    m = np.zeros((50, 50), np.uint8)
    m[5:15, 5:25] = 255
    cb = CocoBuilder({0: "scratch"})
    cb.add("a.png", m, 0)
    cb.save(tmp_path / "c.json")
    d = json.loads((tmp_path / "c.json").read_text())
    assert d["annotations"][0]["bbox"] == [5, 5, 20, 10] and d["annotations"][0]["segmentation"]


def test_mask_generators(tmp_path):
    import random
    rng = random.Random(0)
    for m in (stroke_mask(128, rng), blob_mask(128, rng)):
        assert m.shape == (128, 128) and (m > 0).sum() > 0 and set(np.unique(m)) <= {0, 255}
    real = np.zeros((128, 128), np.uint8)
    cv2.circle(real, (64, 64), 10, 255, -1)
    assert (deform_real(real, rng, elastic=2) > 0).sum() > 0
    write_mask(tmp_path / "r.png", real)
    gen = MaskGenerator(128, "mixed", tmp_path, seed=1)
    modes = {gen()[1] for _ in range(30)}
    assert modes <= {"real", "stroke", "blob"} and len(modes) >= 2


def test_labelme_to_masks(tmp_path):
    ann = {"imageHeight": 60, "imageWidth": 80, "imagePath": "x.png", "shapes": [
        {"label": "scratch", "shape_type": "polygon", "points": [[10, 10], [30, 10], [30, 30]]},
        {"label": "dent", "shape_type": "rectangle", "points": [[50, 40], [40, 50]]},
        {"label": "scratch", "shape_type": "linestrip", "points": [[0, 55], [70, 55]]}]}
    (tmp_path / "x.json").write_text(json.dumps(ann))
    run("scripts/labelme_to_masks.py", "--json-dir", str(tmp_path), "--out-dir", str(tmp_path / "m"))
    s, d = read_mask(tmp_path / "m/scratch/x.png"), read_mask(tmp_path / "m/dent/x.png")
    assert s.shape == (60, 80) and s[15, 25] == 255 and s[55, 35] == 255
    assert d[45, 45] == 255 and d[5, 5] == 0


def test_prepare_crops_and_grid(tmp_path):
    root = make_dataset(tmp_path)
    run("-m", "src.data.prepare_crops", "--images", str(root / "defect/scratch"),
        "--masks", str(root / "masks/scratch"), "--out", str(root / "crops"), "--caption", "a photo of sks scratch",
        "--size", "64", "--crops-per-image", "2")
    items = [json.loads(l) for l in (root / "crops/metadata.jsonl").read_text().splitlines()]
    assert len(items) == 6
    for it in items:
        m = read_mask(root / "crops" / it["mask"])
        assert m.shape == (64, 64) and (m > 0).sum() > 0
    run("scripts/make_grid.py", "--images", str(root / "crops/images"), "--masks", str(root / "crops/masks"),
        "--overlay", "--boxes", "--out", str(root / "grid.png"))
    assert (root / "grid.png").exists()


# ---------- diffusion parts (tiny random model) ----------

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tests.tiny_sd import build_tiny_inpaint_model
    return build_tiny_inpaint_model(tmp_path_factory.mktemp("tiny_sd"))


def test_train_lora_then_generate(tmp_path, tiny_model):
    root = make_dataset(tmp_path)
    run("-m", "src.data.prepare_crops", "--images", str(root / "defect/scratch"),
        "--masks", str(root / "masks/scratch"), "--out", str(root / "crops"), "--caption", "a photo of sks scratch",
        "--size", "64", "--crops-per-image", "2")
    run("-m", "src.diffusion.train_lora", "--data", str(root / "crops"), "--output", str(root / "lora"),
        "--base-model", tiny_model, "--resolution", "64", "--max-steps", "4", "--grad-accum", "1",
        "--warmup-steps", "1", "--save-every", "2", "--sample-every", "4", "--num-samples", "2",
        "--sample-steps", "2", "--val-good-dir", str(root / "good"), "--num-workers", "0",
        "--mask-dilate", "2", "--mask-dilate-jitter", "2")
    assert (root / "lora/pytorch_lora_weights.safetensors").exists()
    assert (root / "lora/checkpoint-2/pytorch_lora_weights.safetensors").exists()
    assert (root / "lora/samples/step_000004.png").exists()

    out = root / "gen"
    run("-m", "src.diffusion.inpaint_generate", "--good-dir", str(root / "good"), "--out", str(out),
        "--prompt", "a photo of sks scratch", "--base-model", tiny_model, "--lora", str(root / "lora"),
        "--real-mask-dir", str(root / "crops/masks"), "--num", "3", "--size", "64", "--steps", "2",
        "--diff-threshold", "1", "--min-changed-ratio", "0.01", "--mask-dilate", "2", "--feather", "1")
    metas = [json.loads(l) for l in (out / "metadata.jsonl").read_text().splitlines()]
    assert len(metas) == 3
    for m in metas:
        img = cv2.imread(str(out / m["image"]))
        mask = read_mask(out / m["mask"])
        assert img.shape[:2] == (160, 200) == mask.shape
        x0, y0, side = m["window"]
        outside = mask.copy()
        outside[y0:y0 + side, x0:x0 + side] = 0
        assert outside.sum() == 0  # labels only inside the edited window
        name = Path(m["image"]).stem
        assert (out / "labels" / f"{name}.txt").exists()

    # pixels outside the edited window are untouched
    src = cv2.imread(str(root / "good" / metas[0]["source"]))
    gen = cv2.imread(str(out / metas[0]["image"]))
    x0, y0, side = metas[0]["window"]
    diff = np.abs(src.astype(int) - gen.astype(int))
    diff[y0:y0 + side, x0:x0 + side] = 0
    assert diff.max() == 0


def test_zero_shot_script(tmp_path, tiny_model):
    root = make_dataset(tmp_path)
    run("scripts/zero_shot_inpaint.py", "--good-dir", str(root / "good"), "--out", str(root / "zs"),
        "--prompt", "a scratch", "--base-model", tiny_model, "--num", "2", "--size", "64", "--steps", "2",
        "--mask-mode", "blob", "--diff-threshold", "1", "--min-changed-ratio", "0.01")
    assert len(list((root / "zs/images").glob("*.png"))) == 2
