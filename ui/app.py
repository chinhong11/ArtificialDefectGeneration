"""Browser UI for the whole pipeline (runs locally, images never leave your PC).

    python -m ui.app                      # then open http://127.0.0.1:7860
    python -m ui.app --host 0.0.0.0       # reachable from other PCs on your network

Tabs: 1 Images -> 2 Annotate -> 3 Masks -> 4 Train -> 5 Generate -> 6 Export.
Everything is stored in a project folder (default data/my_project):
    good/  defect_raw/          your images + labelme-style JSON annotations
    work/                       masks, crops, ROI masks, LoRA weights
    generated/<label>/          synthetic images, masks, YOLO labels
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import find_mask_for, list_images, read_mask, read_rgb  # noqa: E402
from src.export import mask_to_boxes  # noqa: E402
from ui import annotations as A  # noqa: E402

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SAM_MODEL = "facebook/sam-vit-base"
MAX_LOG_LINES = 300


# ---------------------------------------------------------------- helpers

def proj(path):
    p = Path(path).expanduser()
    p = p if p.is_absolute() else ROOT / p
    for sub in ("good", "defect_raw"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def image_choices(project):
    p = proj(project)
    return [f"defect_raw/{x.name}" for x in list_images(p / "defect_raw")] + \
           [f"good/{x.name}" for x in list_images(p / "good")]


def status_text(project):
    p = proj(project)
    lines = []
    for sub, what in (("defect_raw", "boxes"), ("good", "part outline")):
        imgs = list_images(p / sub)
        done = 0
        for im in imgs:
            jp = A.json_path(im)
            if jp.exists():
                b, r = A.summary(A.load(im, (1, 1)))
                done += (b > 0) if sub == "defect_raw" else (r > 0)
        lines.append(f"**{sub}/**: {len(imgs)} images, {done} with {what}")
    labels = A.labels_in(p / "defect_raw")
    lines.append(f"**defect labels:** {', '.join(labels) if labels else '(none yet)'}")
    return "  \n".join(lines)


def gpu_text():
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            return f"GPU: {pr.name} ({pr.total_memory / 2**30:.0f} GB)"
        return "GPU: not found - training/generation will be very slow on CPU"
    except ImportError:
        return "PyTorch not installed"


def stream(cmd, log=""):
    """Run a pipeline script, yield the growing log text. Raises on failure."""
    cmd = [sys.executable, *map(str, cmd)]
    log += "\n$ " + " ".join(cmd) + "\n"
    yield log
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, bufsize=1)
    for line in proc.stdout:
        # progress bars use \r; keep only the latest state of the line
        line = line.rstrip("\n").split("\r")[-1]
        if line.strip():
            log += line + "\n"
            log = "\n".join(log.splitlines()[-MAX_LOG_LINES:]) + "\n"
            yield log
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(log + f"\nFAILED (exit code {proc.returncode})")
    yield log


def overlay(img, mask, boxes=True, max_side=900):
    out = img.copy()
    if mask is not None and mask.any():
        t = max(2, max(out.shape[:2]) // 400)
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 0, 0), t)
        if boxes:
            for x0, y0, x1, y1 in mask_to_boxes(mask):
                cv2.rectangle(out, (x0 - 3 * t, y0 - 3 * t), (x1 + 3 * t, y1 + 3 * t), (0, 220, 0), t)
    s = max_side / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else out


def gallery_from(img_dir, mask_dir, limit=60):
    items = []
    if not Path(img_dir).is_dir():
        return items
    for p in list_images(img_dir)[:limit]:
        mp = find_mask_for(p, mask_dir) if mask_dir and Path(mask_dir).is_dir() else None
        items.append((overlay(read_rgb(p), read_mask(mp) if mp else None), p.name))
    return items


def label_choices(project):
    labels = A.labels_in(proj(project) / "defect_raw")
    return gr.Dropdown(choices=labels, value=labels[0] if labels else None)


def build_roi_masks(p):
    """good/*.json polygons labelled roi -> work/roi/roi/*.png ; returns dir or None."""
    if not any((p / "good").glob("*.json")):
        return None, ""
    log = ""
    for log in stream(["scripts/labelme_to_masks.py", "--json-dir", p / "good", "--out-dir", p / "work" / "roi",
                       "--labels", A.ROI_LABEL]):
        pass
    d = p / "work" / "roi" / A.ROI_LABEL
    return (d if d.is_dir() else None), log


# ---------------------------------------------------------------- tab 1: images

def upload(project, files, sub):
    p = proj(project)
    n = 0
    for f in files or []:
        src = Path(f if isinstance(f, str) else f.name)
        if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            shutil.copy2(src, p / sub / src.name)
            n += 1
    choices = image_choices(project)
    return (f"Added {n} image(s) to {sub}/", status_text(project),
            gr.Dropdown(choices=choices, value=choices[0] if choices else None))


def refresh(project):
    choices = image_choices(project)
    return status_text(project), gr.Dropdown(choices=choices, value=choices[0] if choices else None)


# ---------------------------------------------------------------- tab 2: annotate

def _load(project, name):
    path = proj(project) / name
    img = read_rgb(path)
    return path, img, A.load(path, img.shape)


MOVE = "Move view (click to centre)"


def _kind(mode):
    return "polygon" if mode.startswith("Part") else ("move" if mode == MOVE else "box")


def _view(view, zoom):
    view = dict(view or {})
    view["zoom"] = int(str(zoom).rstrip("×x") or 1)
    return view


def _show(img, ann, pending, kind, view):
    """Render the (zoomed) view, scaled back up to full size so zoom really magnifies."""
    h, w = img.shape[:2]
    win = A.view_window(h, w, view.get("zoom", 1), view.get("center"))
    out = A.render(img, ann, pending, "polygon" if kind == "polygon" else "box", window=win)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out, win


def _hint(name, ann, pending, kind, view):
    b, r = A.summary(ann)
    if kind == "polygon":
        step = f"outline: {len(pending)} point(s) - click around the part, then 'Finish outline'"
    elif kind == "move":
        step = "click where the view should be centred, then switch back to drawing"
    else:
        step = "click the 2nd corner" if pending else "click the 1st corner of a defect box"
    z = view.get("zoom", 1)
    return f"**{name}** - {b} box(es), {r} outline(s). Zoom {z}×. Next: {step}."


def show_image(project, name, mode, zoom, view):
    if not name:
        return None, [], view, "Upload images in tab 1 first."
    _, img, ann = _load(project, name)
    view = _view(view, zoom)
    if view.get("name") != name:  # new image: reset view centre
        view = {"zoom": view["zoom"], "name": name}
    kind = _kind(mode)
    out, _ = _show(img, ann, [], kind, view)
    return out, [], view, _hint(name, ann, [], kind, view)


def on_click(project, name, mode, label, pending, zoom, view, evt: gr.SelectData):
    if not name:
        return gr.skip(), pending, view, "Select an image first."
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    h, w = img.shape[:2]
    x0, y0, vw, vh = A.view_window(h, w, view["zoom"], view.get("center"))
    # the view is displayed at full size: displayed pixel -> full-image pixel
    x, y = round(x0 + evt.index[0] * vw / w), round(y0 + evt.index[1] * vh / h)
    pending = list(pending or [])
    kind = _kind(mode)
    msg = ""
    if kind == "move":
        view["center"] = (x, y)
    elif kind == "box":
        if not pending:
            pending = [(x, y)]
        else:
            if A.add_box(ann, label, pending[0], (x, y)):
                A.save(path, ann)
            else:
                msg = " (box too small, ignored)"
            pending = []
    else:
        pending.append((x, y))
    out, _ = _show(img, ann, pending, kind, view)
    return out, pending, view, _hint(name, ann, pending, kind, view) + msg


def finish_outline(project, name, pending, zoom, view):
    if not name:
        return gr.skip(), pending, view, "Select an image first."
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    if A.add_polygon(ann, pending or []):
        A.save(path, ann)
        pending = []
        msg = ""
    else:
        msg = " (need at least 3 points)"
    out, _ = _show(img, ann, pending, "polygon", view)
    return out, pending, view, _hint(name, ann, pending, "polygon", view) + msg


def undo(project, name, mode, pending, zoom, view):
    if not name:
        return gr.skip(), [], view, ""
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    kind = _kind(mode)
    if pending:
        pending = list(pending)[:-1]
    else:
        A.remove_last(ann)
        A.save(path, ann)
    out, _ = _show(img, ann, pending, kind, view)
    return out, pending, view, _hint(name, ann, pending, kind, view)


def clear_all(project, name, mode, zoom, view):
    if not name:
        return gr.skip(), [], view, ""
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    ann["shapes"] = []
    A.save(path, ann)
    kind = _kind(mode)
    out, _ = _show(img, ann, [], kind, view)
    return out, [], view, _hint(name, ann, [], kind, view)


def step_image(project, name, delta):
    choices = image_choices(project)
    if not choices:
        return None
    i = choices.index(name) if name in choices else -1
    return choices[(i + delta) % len(choices)]


# ---------------------------------------------------------------- tab 3: masks

def make_masks(project, method, sam_model):
    p = proj(project)
    if not A.labels_in(p / "defect_raw"):
        yield "No defect boxes yet - draw them in tab 2.", []
        return
    shutil.rmtree(p / "work" / "masks", ignore_errors=True)
    shutil.rmtree(p / "work" / "defect", ignore_errors=True)
    if method.startswith("SAM"):
        cmd = ["scripts/sam_box_to_mask.py", "--json-dir", p / "defect_raw", "--out-dir", p / "work" / "masks",
               "--copy-images-to", p / "work" / "defect", "--model", sam_model, "--fallback-box"]
    else:
        cmd = ["scripts/labelme_to_masks.py", "--json-dir", p / "defect_raw", "--out-dir", p / "work" / "masks",
               "--copy-images-to", p / "work" / "defect"]
    log = ""
    try:
        for log in stream(cmd):
            yield log, gr.skip()
    except RuntimeError as e:
        yield str(e), []
        return
    items = []
    for label in A.labels_in(p / "defect_raw"):
        items += [(im, f"{label}: {cap}") for im, cap in
                  gallery_from(p / "work" / "defect" / label, p / "work" / "masks" / label)]
    yield log + "\nDone. Check that the red outlines follow the defects.", items


# ---------------------------------------------------------------- tab 4: train

def train(project, label, steps, rank, lr, crops_per_image, sd_model):
    p = proj(project)
    mask_dir, img_dir = p / "work" / "masks" / label, p / "work" / "defect" / label
    if not label or not mask_dir.is_dir():
        yield "Make masks first (tab 3).", None
        return
    crops, lora = p / "work" / "crops" / label, p / "work" / f"lora_{label}"
    shutil.rmtree(crops, ignore_errors=True)
    shutil.rmtree(lora, ignore_errors=True)
    import torch
    gpu = torch.cuda.is_available()
    steps = int(steps)
    log, preview, seen = "", None, set()
    try:
        for log in stream(["-m", "src.data.prepare_crops", "--images", img_dir, "--masks", mask_dir,
                           "--out", crops, "--caption", f"a photo of sks {label}",
                           "--crops-per-image", int(crops_per_image)], log):
            yield log, gr.skip()
        cmd = ["-m", "src.diffusion.train_lora", "--data", crops, "--output", lora, "--base-model", sd_model,
               "--max-steps", steps, "--rank", int(rank), "--lr", lr, "--save-every", max(50, steps // 4),
               "--sample-every", max(25, steps // 8), "--warmup-steps", min(100, steps // 10),
               "--val-good-dir", p / "good", "--num-workers", 0]
        cmd += ["--mixed-precision", "fp16", "--gradient-checkpointing"] if gpu else ["--grad-accum", 1,
                                                                                      "--sample-steps", 10]
        for log in stream(cmd, log):
            new = sorted(set((lora / "samples").glob("step_*.png")) - seen) if (lora / "samples").is_dir() else []
            if new:
                seen.update(new)
                preview = read_rgb(new[-1])
                yield log, preview
            else:
                yield log, gr.skip()
    except RuntimeError as e:
        yield str(e), preview
        return
    yield log + f"\nDone. LoRA saved in {lora}. Previews: left = good crop + mask, right = generated.", preview


# ---------------------------------------------------------------- tab 5: generate

def generate(project, label, use_lora, num, mask_mode, lora_scale, infer_steps, sd_model):
    p = proj(project)
    if not label:
        yield "No defect label - draw boxes (tab 2) and make masks (tab 3) first.", []
        return
    lora = p / "work" / f"lora_{label}"
    if use_lora and not (lora / "pytorch_lora_weights.safetensors").exists():
        yield f"No trained LoRA for '{label}' - train it in tab 4, or untick 'Use trained LoRA'.", []
        return
    out = p / "generated" / label
    shutil.rmtree(out, ignore_errors=True)
    roi_dir, log = build_roi_masks(p)
    if roi_dir is None:
        log += "\nWARNING: no part outlines on good images - defects may land on the background.\n"
    real_masks = p / "work" / "crops" / label / "masks"
    cmd = ["-m", "src.diffusion.inpaint_generate", "--good-dir", p / "good", "--out", out,
           "--prompt", f"a photo of sks {label}" if use_lora else f"a small {label} defect",
           "--base-model", sd_model, "--num", int(num), "--steps", int(infer_steps), "--mask-mode", mask_mode]
    if mask_mode in ("real", "mixed"):
        if real_masks.is_dir():
            cmd += ["--real-mask-dir", real_masks]
        elif mask_mode == "real":
            yield "Mask mode 'real' needs training crops - run tab 4 first or pick stroke/blob.", []
            return
    if roi_dir:
        cmd += ["--roi-dir", roi_dir]
    if use_lora:
        cmd += ["--lora", lora, "--lora-scale", lora_scale]
    try:
        for log in stream(cmd, log):
            yield log, gr.skip()
    except RuntimeError as e:
        yield str(e), []
        return
    yield (log + f"\nDone. Saved in {out} (images/, masks/, labels/ = YOLO).",
           gallery_from(out / "images", out / "masks"))


# ---------------------------------------------------------------- tab 6: export

def export(project, label):
    p = proj(project)
    src = p / "generated" / (label or "")
    if not label or not (src / "images").is_dir():
        return None, "Nothing generated yet for this label."
    (src / "classes.txt").write_text(label + "\n")
    base = p / "exports" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    base.parent.mkdir(exist_ok=True)
    z = shutil.make_archive(str(base), "zip", src)
    n = len(list_images(src / "images"))
    return z, f"{n} images with masks + YOLO labels -> {z}"


# ---------------------------------------------------------------- layout

def build(default_project="data/my_project"):
    with gr.Blocks(title="Artificial Defect Generation") as demo:
        gr.Markdown("# Artificial Defect Generation\nTeach Stable Diffusion your defect from a few real examples, "
                    "then paint new defects onto good images - with masks and YOLO boxes for free.")
        with gr.Row():
            project = gr.Textbox(default_project, label="Project folder", scale=3)
            gr.Markdown(gpu_text())
        with gr.Accordion("Advanced: models (use a local folder if Hugging Face is blocked)", open=False):
            sd_model = gr.Textbox(SD_MODEL, label="Stable Diffusion inpainting model")
            sam_model = gr.Textbox(SAM_MODEL, label="SAM model")

        with gr.Tab("1 · Images"):
            gr.Markdown("Upload **good** (defect-free) images and **defect** images of the same product/camera.")
            with gr.Row():
                up_good = gr.File(file_count="multiple", label="Good images")
                up_def = gr.File(file_count="multiple", label="Defect images")
            with gr.Row():
                btn_good = gr.Button("Add good images")
                btn_def = gr.Button("Add defect images")
                btn_refresh = gr.Button("Refresh")
            up_msg = gr.Markdown()
            status = gr.Markdown(status_text(default_project))

        with gr.Tab("2 · Annotate"):
            gr.Markdown("**Defect images:** draw a tight box around every defect (click 2 corners). "
                        "Small defects: choose *Move view* and click the defect, pick Zoom 4×, then "
                        "switch back to *Defect box*.  \n"
                        "**Good images:** draw the part outline (click points around the part, then "
                        "*Finish outline*) so new defects land on the part, not the background.")
            with gr.Row():
                first = image_choices(default_project)
                img_sel = gr.Dropdown(first, value=first[0] if first else None, label="Image", scale=3)
                btn_prev = gr.Button("◀ Prev", scale=0)
                btn_next = gr.Button("Next ▶", scale=0)
            with gr.Row():
                mode = gr.Radio(["Defect box", "Part outline (good images)", MOVE], value="Defect box",
                                label="Click action")
                zoom = gr.Radio(["1×", "2×", "4×", "8×"], value="1×", label="Zoom (small defects: 4× or 8×)")
                label = gr.Textbox("defect", label="Defect type label (e.g. particle, scratch)")
            hint = gr.Markdown()
            canvas = gr.Image(type="numpy", interactive=False, label="Click on the image", height=650,
                              elem_id="canvas")
            with gr.Row():
                btn_finish = gr.Button("Finish outline", variant="primary")
                btn_undo = gr.Button("Undo")
                btn_clear = gr.Button("Clear this image", variant="stop")
            pending = gr.State([])
            view = gr.State({})

        with gr.Tab("3 · Masks"):
            gr.Markdown("Turns your boxes into exact defect masks.")
            method = gr.Radio(["SAM (exact outline, downloads ~375 MB once)", "Filled boxes (no download)"],
                              value="SAM (exact outline, downloads ~375 MB once)", label="Method")
            btn_masks = gr.Button("Make masks", variant="primary")
            mask_gal = gr.Gallery(label="Masks (red outline)", columns=3, height=500)
            mask_log = gr.Textbox(label="Log", lines=8, max_lines=20, autoscroll=True)

        with gr.Tab("4 · Train"):
            gr.Markdown("Teaches the model your defect (LoRA). Use **20-50 defect images** for good results. "
                        "200 steps = quick check; 1000-3000 = real training. Needs a GPU.")
            with gr.Row():
                train_label = gr.Dropdown(A.labels_in(proj(default_project) / "defect_raw"), label="Defect label")
                steps = gr.Slider(50, 4000, value=1000, step=50, label="Training steps")
                rank = gr.Slider(4, 32, value=8, step=4, label="LoRA rank")
            with gr.Row():
                lr = gr.Number(1e-4, label="Learning rate")
                cpi = gr.Slider(1, 8, value=4, step=1, label="Crops per defect")
            btn_train = gr.Button("Start training", variant="primary")
            preview = gr.Image(label="Latest preview (left: good crop + mask, right: generated)", height=500)
            train_log = gr.Textbox(label="Log", lines=10, max_lines=20, autoscroll=True)

        with gr.Tab("5 · Generate"):
            with gr.Row():
                gen_label = gr.Dropdown(A.labels_in(proj(default_project) / "defect_raw"), label="Defect label")
                use_lora = gr.Checkbox(True, label="Use trained LoRA (untick = zero-shot test)")
                num = gr.Slider(1, 1000, value=20, step=1, label="Number of images")
            with gr.Row():
                mask_mode = gr.Radio(["mixed", "real", "stroke", "blob"], value="mixed",
                                     label="Defect shapes (real = like your defects, stroke = scratches, "
                                           "blob = spots/stains)")
                lora_scale = gr.Slider(0.3, 1.2, value=1.0, step=0.05, label="LoRA strength")
                infer_steps = gr.Slider(10, 60, value=30, step=5, label="Quality steps")
            btn_gen = gr.Button("Generate", variant="primary")
            gen_gal = gr.Gallery(label="Generated (red = mask, green = YOLO box)", columns=4, height=600)
            gen_log = gr.Textbox(label="Log", lines=8, max_lines=20, autoscroll=True)

        with gr.Tab("6 · Export"):
            exp_label = gr.Dropdown(A.labels_in(proj(default_project) / "defect_raw"), label="Defect label")
            btn_exp = gr.Button("Create zip", variant="primary")
            exp_msg = gr.Markdown()
            exp_file = gr.File(label="Download")

        # ---- wiring
        btn_good.click(lambda pr, f: upload(pr, f, "good"), [project, up_good], [up_msg, status, img_sel])
        btn_def.click(lambda pr, f: upload(pr, f, "defect_raw"), [project, up_def], [up_msg, status, img_sel])
        btn_refresh.click(refresh, project, [status, img_sel])
        project.submit(refresh, project, [status, img_sel])

        show_io = ([project, img_sel, mode, zoom, view], [canvas, pending, view, hint])
        demo.load(show_image, *show_io)
        img_sel.change(show_image, *show_io)
        mode.change(show_image, *show_io)
        zoom.change(show_image, *show_io)
        canvas.select(on_click, [project, img_sel, mode, label, pending, zoom, view], [canvas, pending, view, hint])
        btn_finish.click(finish_outline, [project, img_sel, pending, zoom, view], [canvas, pending, view, hint])
        btn_undo.click(undo, [project, img_sel, mode, pending, zoom, view], [canvas, pending, view, hint])
        btn_clear.click(clear_all, [project, img_sel, mode, zoom, view], [canvas, pending, view, hint])
        btn_prev.click(lambda pr, n: step_image(pr, n, -1), [project, img_sel], img_sel)
        btn_next.click(lambda pr, n: step_image(pr, n, 1), [project, img_sel], img_sel)

        btn_masks.click(make_masks, [project, method, sam_model], [mask_log, mask_gal]).then(
            label_choices, project, train_label).then(label_choices, project, gen_label).then(
            label_choices, project, exp_label).then(status_text, project, status)
        btn_train.click(train, [project, train_label, steps, rank, lr, cpi, sd_model], [train_log, preview])
        btn_gen.click(generate, [project, gen_label, use_lora, num, mask_mode, lora_scale, infer_steps, sd_model],
                      [gen_log, gen_gal])
        btn_exp.click(export, [project, exp_label], [exp_file, exp_msg])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="data/my_project")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    proj(args.project)
    build(args.project).queue().launch(server_name=args.host, server_port=args.port,
                                       allowed_paths=[str(proj(args.project))])


if __name__ == "__main__":
    main()
