"""One-command test of the whole pipeline with the REAL models on your images.

Prepare (labelme):
    <data>/good/*.png|jpg|webp        + optional polygon labelled "roi" around the part (JSON next to image)
    <data>/defect_raw/*.png|jpg|webp  + rectangles around each defect (JSON next to image)

Run:
    python scripts/run_real_test.py --data data/connector

Then send back everything in outputs/real_test/: the *.png grids and report.txt
(log.txt too if something failed).

Steps: environment -> SAM masks -> ROI masks -> crops -> zero-shot inpaint ->
short LoRA training -> generation with LoRA -> output checks.
The short training only proves it runs and gives a first impression; real
training needs ~1000-3000 steps (see docs/GETTING_STARTED.md).
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import list_images, read_mask  # noqa: E402

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SAM_MODEL = "facebook/sam-vit-base"


class Runner:
    def __init__(self, out):
        self.out = out
        self.log = open(out / "log.txt", "w")
        self.results = []  # (step, status, seconds, note)

    def note(self, text):
        print(text)
        self.log.write(text + "\n")
        self.log.flush()

    def run(self, step, args):
        cmd = [sys.executable, *map(str, args)]
        self.note(f"\n===== {step} =====\n$ {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        lines = []
        for line in proc.stdout:
            lines.append(line)
            self.log.write(line)
            if not line.startswith(("Loading", "Fetching", "Writing")):
                sys.stdout.write(line)
        proc.wait()
        self.log.flush()
        dt = time.time() - t0
        ok = proc.returncode == 0
        self.results.append((step, "PASS" if ok else "FAIL", dt, "" if ok else f"exit code {proc.returncode}"))
        if not ok:
            raise RuntimeError(f"{step} failed (exit code {proc.returncode}) - see log.txt")
        return "".join(lines)

    def check(self, step, ok, note):
        self.results.append((step, "PASS" if ok else "FAIL", 0.0, note))
        self.note(f"[{'PASS' if ok else 'FAIL'}] {step}: {note}")


def env_report():
    lines = [f"python {platform.python_version()} on {platform.platform()}"]
    try:
        import torch
        lines.append(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            lines.append(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
    except ImportError:
        lines.append("torch NOT installed")
    for mod in ("diffusers", "transformers", "peft", "accelerate", "cv2"):
        try:
            lines.append(f"{mod} {__import__(mod).__version__}")
        except ImportError:
            lines.append(f"{mod} NOT installed")
    return lines


def first_label(json_dir):
    for jf in sorted(Path(json_dir).glob("*.json")):
        for s in json.loads(jf.read_text(encoding="utf-8")).get("shapes", []):
            if s.get("shape_type") == "rectangle":
                return s["label"].strip()
    return None


def check_generated(gen_dir, good_dir, roi_dir):
    """Return list of problems found in generator output (empty = all good)."""
    problems, n = [], 0
    goods = {p.name: p for p in list_images(good_dir)}
    for line in (gen_dir / "metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        n += 1
        img = cv2.imread(str(gen_dir / m["image"]))
        src = cv2.imread(str(goods[m["source"]]))
        mask = read_mask(gen_dir / m["mask"])
        name = Path(m["image"]).stem
        if img.shape != src.shape or mask.shape != src.shape[:2]:
            problems.append(f"{name}: size {img.shape} differs from source {src.shape}")
            continue
        x0, y0, side = m["window"]
        diff = np.abs(img.astype(int) - src.astype(int))
        diff[y0:y0 + side, x0:x0 + side] = 0
        if diff.max() > 0:
            problems.append(f"{name}: pixels changed outside the edit window")
        if (mask > 0).sum() == 0:
            problems.append(f"{name}: empty mask")
        for row in (gen_dir / "labels" / f"{name}.txt").read_text().split("\n"):
            if row.strip():
                v = row.split()
                if len(v) != 5 or not all(0 <= float(x) <= 1 for x in v[1:]):
                    problems.append(f"{name}: bad YOLO line '{row}'")
        if roi_dir:
            roi = read_mask(Path(roi_dir) / (Path(m["source"]).stem + ".png"))
            frac = ((mask > 0) & (roi > 0)).sum() / max(1, (mask > 0).sum())
            if frac < 0.9:
                problems.append(f"{name}: only {frac:.0%} of the label is inside the ROI")
    if n == 0:
        problems.append("no images generated")
    return problems, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Folder with good/ and defect_raw/")
    ap.add_argument("--out", default="outputs/real_test")
    ap.add_argument("--label", help="Defect label to use (default: first rectangle label found)")
    ap.add_argument("--steps", type=int, help="LoRA training steps (default 200 on GPU, 10 on CPU)")
    ap.add_argument("--num", type=int, default=8, help="Images to generate with the LoRA")
    ap.add_argument("--skip-train", action="store_true", help="Stop after the zero-shot test")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (very slow; tiny step counts)")
    ap.add_argument("--no-sam", action="store_true", help="Use the filled boxes as masks instead of SAM")
    # overrides (mainly for offline testing with tiny models)
    ap.add_argument("--sd-model", default=SD_MODEL)
    ap.add_argument("--sam-model", default=SAM_MODEL)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--infer-steps", type=int, default=30)
    ap.add_argument("--gen-extra", default="", help="Extra args for the generator, e.g. '--diff-threshold 8'")
    args = ap.parse_args()

    data, out = Path(args.data).resolve(), (ROOT / args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    work = out / "work"
    work.mkdir(parents=True)
    r = Runner(out)

    import torch
    gpu = torch.cuda.is_available() and not args.cpu
    steps = args.steps or (200 if gpu else 10)
    infer = args.infer_steps if gpu else min(args.infer_steps, 20)
    env = env_report()
    r.note("\n".join(env))
    extra = args.gen_extra.split()

    try:
        raw, good = data / "defect_raw", data / "good"
        if not list_images(raw) or not list(raw.glob("*.json")):
            raise RuntimeError(f"{raw} needs defect images + labelme rectangle .json files")
        if not list_images(good):
            raise RuntimeError(f"{good} has no images")
        label = args.label or first_label(raw)
        if not label:
            raise RuntimeError("no rectangle shapes found in defect_raw/*.json")
        r.note(f"defect label: {label}")
        caption = f"a photo of sks {label}"

        # 1. masks (SAM, or filled boxes with --no-sam)
        if args.no_sam:
            r.run("1 box -> mask (no SAM)", ["scripts/labelme_to_masks.py", "--json-dir", raw,
                                             "--out-dir", work / "masks", "--labels", label,
                                             "--copy-images-to", work / "defect"])
        else:
            r.run("1 SAM box -> mask", ["scripts/sam_box_to_mask.py", "--json-dir", raw, "--out-dir", work / "masks",
                                        "--copy-images-to", work / "defect", "--model", args.sam_model,
                                        "--fallback-box"])
        r.run("1b grid masks", ["scripts/make_grid.py", "--images", work / "defect" / label,
                                "--masks", work / "masks" / label, "--overlay", "--cols", 3, "--tile", 600,
                                "--out", out / "01_sam_masks.png"])

        # 2. ROI
        roi_dir = None
        if any(good.glob("*.json")):
            r.run("2 ROI masks", ["scripts/labelme_to_masks.py", "--json-dir", good, "--out-dir", work / "roi",
                                  "--labels", "roi"])
            roi_dir = work / "roi" / "roi"
            missing = [p.name for p in list_images(good) if not (roi_dir / (p.stem + ".png")).exists()]
            r.check("2b ROI coverage", not missing,
                    "all good images have an ROI" if not missing else f"no ROI for {missing}")
            if missing:
                raise RuntimeError("every good image needs a 'roi' polygon (or remove all good/*.json)")
        else:
            r.check("2 ROI masks", True, "WARNING: no ROI polygons in good/ - defects may land on background")
        roi_args = ["--roi-dir", roi_dir] if roi_dir else []

        # 3. crops
        crops = work / "crops"
        r.run("3 training crops", ["-m", "src.data.prepare_crops", "--images", work / "defect" / label,
                                   "--masks", work / "masks" / label, "--out", crops, "--caption", caption,
                                   "--size", args.size, "--crops-per-image", 4])
        r.run("3b grid crops", ["scripts/make_grid.py", "--images", crops / "images", "--masks", crops / "masks",
                                "--overlay", "--out", out / "02_crops.png"])

        # 4. zero-shot
        zs = work / "zero_shot"
        r.run("4 zero-shot inpaint", ["scripts/zero_shot_inpaint.py", "--good-dir", good, "--out", zs,
                                      "--prompt", f"a small {label} on a transparent plastic part",
                                      "--base-model", args.sd_model, "--num", 4, "--size", args.size,
                                      "--steps", infer, "--real-mask-dir", crops / "masks", *roi_args, *extra])
        r.run("4b grid zero-shot", ["scripts/make_grid.py", "--images", zs / "images", "--masks", zs / "masks",
                                    "--overlay", "--boxes", "--cols", 2, "--tile", 600,
                                    "--out", out / "03_zero_shot.png"])
        probs, n = check_generated(zs, good, roi_dir)
        r.check("4c zero-shot output checks", not probs, f"{n} images OK" if not probs else "; ".join(probs))

        if not args.skip_train:
            # 5. LoRA
            lora = work / "lora"
            train = ["-m", "src.diffusion.train_lora", "--data", crops, "--output", lora,
                     "--base-model", args.sd_model, "--resolution", args.size, "--max-steps", steps,
                     "--save-every", steps, "--sample-every", steps, "--num-samples", 4,
                     "--sample-steps", infer, "--val-good-dir", good, "--num-workers", 0,
                     "--warmup-steps", min(100, max(1, steps // 10))]
            if gpu:
                train += ["--mixed-precision", "fp16", "--gradient-checkpointing"]
            log = r.run(f"5 LoRA training ({steps} steps)", train)
            peak = re.findall(r"peak GPU memory: ([\d.]+ GB)", log)
            if peak:
                r.note(f"peak GPU memory during training: {peak[-1]}")
            samples = sorted((lora / "samples").glob("step_*.png"))
            if samples:
                shutil.copy(samples[-1], out / "04_lora_samples.png")

            # 6. generate with LoRA
            gen = work / "generated"
            r.run("6 generate with LoRA", ["-m", "src.diffusion.inpaint_generate", "--good-dir", good,
                                           "--out", gen, "--prompt", caption, "--base-model", args.sd_model,
                                           "--lora", lora, "--num", args.num, "--size", args.size,
                                           "--steps", infer, "--mask-mode", "mixed",
                                           "--real-mask-dir", crops / "masks", *roi_args, *extra])
            r.run("6b grid generated", ["scripts/make_grid.py", "--images", gen / "images", "--masks",
                                        gen / "masks", "--overlay", "--boxes", "--cols", 2, "--tile", 600,
                                        "--out", out / "05_lora_generated.png"])
            probs, n = check_generated(gen, good, roi_dir)
            r.check("6c generated output checks", not probs, f"{n} images OK" if not probs else "; ".join(probs))
    except Exception as e:  # noqa: BLE001 - report any failure, then exit non-zero
        r.note(f"\nSTOPPED: {e}")
        failed = True
    else:
        failed = any(s == "FAIL" for _, s, _, _ in r.results)

    report = ["REAL MODEL TEST REPORT", "=" * 22, *env, ""]
    report += [f"{s:4}  {t:7.1f}s  {step}" + (f"  -- {n}" if n else "") for step, s, t, n in r.results]
    report += ["", "RESULT: " + ("FAILED - send report.txt and log.txt" if failed else
                                 "ALL STEPS PASSED - now judge the *.png grids by eye")]
    (out / "report.txt").write_text("\n".join(report) + "\n")
    print("\n" + "\n".join(report))
    print(f"\nOutputs in {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
