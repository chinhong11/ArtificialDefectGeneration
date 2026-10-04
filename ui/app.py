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
from gradio_image_annotation import image_annotator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import find_mask_for, list_images, read_mask, read_rgb  # noqa: E402
from src.export import mask_to_boxes  # noqa: E402
from ui import annotations as A  # noqa: E402

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SAM_MODEL = "facebook/sam-vit-base"
MAX_LOG_LINES = 300
# hide annotator tools that would break saved coordinates/labels (rotate) or can't show our labels
CSS = """
#annot button[aria-label^="Rotate"], #annot button[aria-label="Edit label"] { display: none !important; }
"""


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
ADD_POINT = "Add outline point"
NEW_TAG = "(new)"   # shown on just-drawn boxes until the image is reloaded
BOX_COLORS = [(255, 40, 40), (40, 120, 255), (255, 160, 0), (200, 0, 200), (0, 180, 180), (120, 80, 0)]


def _is_defect(name):
    return bool(name) and name.startswith("defect_raw/")


def _view(view, zoom):
    view = dict(view or {})
    view["zoom"] = int(str(zoom).rstrip("×x") or 1)
    return view


def _show(img, ann, pending, view):
    """Render the (zoomed) outline view, scaled back up to full size so zoom really magnifies."""
    h, w = img.shape[:2]
    win = A.view_window(h, w, view.get("zoom", 1), view.get("center"))
    out = A.render(img, ann, pending, "polygon", window=win)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out, win


def _hint(name, ann, pending=(), action=ADD_POINT, view=None):
    b, r = A.summary(ann)
    if _is_defect(name):
        return (f"**{name}** - {b} defect box(es), saved automatically with the label above (new boxes "
                f"show '{NEW_TAG}' until you switch image). **Drag** to draw a box, drag its "
                f"corners to resize, **mouse wheel** to zoom (*Space* resets), hand tool or *D* to pan/select, then "
                f"*Delete* removes the selected box.")
    step = ("click where the view should be centred" if action == MOVE else
            f"{len(pending)} point(s) - click around the part, then 'Finish outline'")
    return f"**{name}** - {r} part outline(s). Zoom {(view or {}).get('zoom', 1)}×. Next: {step}."


def _label_props(label, project):
    labels = [label.strip() or "defect"] + [l for l in A.labels_in(proj(project) / "defect_raw")
                                             if l != (label.strip() or "defect")]
    return labels, [BOX_COLORS[i % len(BOX_COLORS)] for i in range(len(labels))]


def _annotator_value(path, ann, labels, colors):
    cmap = dict(zip(labels, colors))
    boxes = []
    for sh in ann["shapes"]:
        if sh["shape_type"] == "rectangle":
            (x0, y0), (x1, y1) = sh["points"]
            boxes.append({"xmin": int(round(min(x0, x1))), "ymin": int(round(min(y0, y1))),
                          "xmax": int(round(max(x0, x1))), "ymax": int(round(max(y0, y1))),
                          "label": sh["label"], "color": cmap.get(sh["label"], BOX_COLORS[0])})
    return {"image": str(path), "boxes": boxes}


def show_image(project, name, label, zoom, view):
    """Returns: annotator, canvas, pending, view, hint, box panel visible, outline panel visible."""
    if not name:
        return (gr.skip(), None, [], view, "Upload images in tab 1 first.",
                gr.Group(visible=False), gr.Group(visible=False))
    path, img, ann = _load(project, name)
    if _is_defect(name):
        labels, colors = _label_props(label, project)
        ann_value = _annotator_value(path, ann, labels, colors)
        return (ann_value, gr.skip(), [], view, _hint(name, ann),
                gr.Group(visible=True), gr.Group(visible=False))
    view = _view(view, zoom)
    if view.get("name") != name:  # new image: reset view centre
        view = {"zoom": view["zoom"], "name": name}
    out, _ = _show(img, ann, [], view)
    return (gr.skip(), out, [], view, _hint(name, ann, [], ADD_POINT, view),
            gr.Group(visible=False), gr.Group(visible=True))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def save_boxes(project, name, value, label):
    """Annotator changed -> rewrite the rectangles in this image's JSON (outlines are kept).

    The annotator can't change its label list after loading, so labels are decided here:
    boxes matching a saved box (moved/resized) keep their saved label, new boxes get the
    label from the textbox.
    """
    if not _is_defect(name) or not value or not value.get("image"):
        return gr.skip()
    if Path(str(value["image"])).name != Path(name).name:   # event from the previous image
        return gr.skip()
    path, img, ann = _load(project, name)
    old = []
    for sh in ann["shapes"]:
        if sh["shape_type"] == "rectangle":
            (ax, ay), (bx_, by_) = sh["points"]
            old.append((sh["label"], (min(ax, bx_), min(ay, by_), max(ax, bx_), max(ay, by_))))
    ann["shapes"] = [sh for sh in ann["shapes"] if sh["shape_type"] != "rectangle"]
    h, w = img.shape[:2]
    for bx in value.get("boxes") or []:
        x0, x1 = sorted((min(max(bx["xmin"], 0), w), min(max(bx["xmax"], 0), w)))
        y0, y1 = sorted((min(max(bx["ymin"], 0), h), min(max(bx["ymax"], 0), h)))
        best = max(old, key=lambda o: _iou(o[1], (x0, y0, x1, y1)), default=None)
        if best is not None and _iou(best[1], (x0, y0, x1, y1)) > 0.3:
            old.remove(best)
            lb = best[0]
        else:
            lb = label.strip() or "defect"
        A.add_box(ann, lb, (x0, y0), (x1, y1))
    A.save(path, ann)
    return _hint(name, ann)


def relabel_all(project, name, label, zoom, view):
    """Give every box on this image the textbox label, then reload the annotator."""
    if _is_defect(name):
        path, _, ann = _load(project, name)
        for sh in ann["shapes"]:
            if sh["shape_type"] == "rectangle":
                sh["label"] = label.strip() or "defect"
        A.save(path, ann)
    return show_image(project, name, label, zoom, view)


def on_click(project, name, action, pending, zoom, view, evt: gr.SelectData):
    if not name or _is_defect(name):
        return gr.skip(), pending, view, gr.skip()
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    h, w = img.shape[:2]
    x0, y0, vw, vh = A.view_window(h, w, view["zoom"], view.get("center"))
    # the view is displayed at full size: displayed pixel -> full-image pixel
    x, y = round(x0 + evt.index[0] * vw / w), round(y0 + evt.index[1] * vh / h)
    pending = list(pending or [])
    if action == MOVE:
        view["center"] = (x, y)
    else:
        pending.append((x, y))
    out, _ = _show(img, ann, pending, view)
    return out, pending, view, _hint(name, ann, pending, action, view)


def finish_outline(project, name, pending, zoom, view):
    if not name or _is_defect(name):
        return gr.skip(), pending, view, gr.skip()
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    msg = ""
    if A.add_polygon(ann, pending or []):
        A.save(path, ann)
        pending = []
    else:
        msg = " (need at least 3 points)"
    out, _ = _show(img, ann, pending, view)
    return out, pending, view, _hint(name, ann, pending, ADD_POINT, view) + msg


def undo(project, name, pending, zoom, view):
    if not name or _is_defect(name):
        return gr.skip(), pending, view, gr.skip()
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    if pending:
        pending = list(pending)[:-1]
    else:
        polys = [i for i, sh in enumerate(ann["shapes"]) if sh["shape_type"] == "polygon"]
        if polys:
            ann["shapes"].pop(polys[-1])
            A.save(path, ann)
    out, _ = _show(img, ann, pending, view)
    return out, pending, view, _hint(name, ann, pending, ADD_POINT, view)


def clear_all(project, name, zoom, view):
    if not name or _is_defect(name):
        return gr.skip(), [], view, gr.skip()
    path, img, ann = _load(project, name)
    view = _view(view, zoom)
    ann["shapes"] = [sh for sh in ann["shapes"] if sh["shape_type"] != "polygon"]
    A.save(path, ann)
    out, _ = _show(img, ann, [], view)
    return out, [], view, _hint(name, ann, [], ADD_POINT, view)


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
            gr.Markdown("**Defect images:** drag a tight box around every defect.  \n"
                        "**Good images:** click points around the part, then *Finish outline* - so new defects "
                        "land on the part, not the background.")
            with gr.Row():
                first = image_choices(default_project)
                img_sel = gr.Dropdown(first, value=first[0] if first else None, label="Image", scale=3)
                btn_prev = gr.Button("◀ Prev", scale=0)
                btn_next = gr.Button("Next ▶", scale=0)
            hint = gr.Markdown()
            with gr.Group(visible=True) as box_panel:
                with gr.Row():
                    label = gr.Textbox("defect", label="Label for new boxes (e.g. particle, scratch)", scale=3)
                    btn_relabel = gr.Button("Apply label to all boxes in this image", scale=1)
                annot = image_annotator(None, label_list=[NEW_TAG], label_colors=[(255, 160, 0)],
                                        use_default_label=True, sources=[], show_clear_button=False,
                                        image_type="filepath",
                                        box_min_size=3, handle_size=6, height=700, elem_id="annot",
                                        label="Drag to draw · wheel to zoom · hand tool to pan")
            with gr.Group(visible=False) as outline_panel:
                with gr.Row():
                    action = gr.Radio([ADD_POINT, MOVE], value=ADD_POINT, label="Click action")
                    zoom = gr.Radio(["1×", "2×", "4×"], value="1×", label="Zoom")
                canvas = gr.Image(type="numpy", interactive=False, label="Click around the part", height=650,
                                  elem_id="canvas")
                with gr.Row():
                    btn_finish = gr.Button("Finish outline", variant="primary")
                    btn_undo = gr.Button("Undo")
                    btn_clear = gr.Button("Clear outline", variant="stop")
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

        show_io = ([project, img_sel, label, zoom, view],
                   [annot, canvas, pending, view, hint, box_panel, outline_panel])
        demo.load(show_image, *show_io)
        img_sel.change(show_image, *show_io)
        zoom.change(show_image, *show_io)
        annot.change(save_boxes, [project, img_sel, annot, label], hint)
        btn_relabel.click(relabel_all, [project, img_sel, label, zoom, view], show_io[1])
        out4 = [canvas, pending, view, hint]
        canvas.select(on_click, [project, img_sel, action, pending, zoom, view], out4)
        btn_finish.click(finish_outline, [project, img_sel, pending, zoom, view], out4)
        btn_undo.click(undo, [project, img_sel, pending, zoom, view], out4)
        btn_clear.click(clear_all, [project, img_sel, zoom, view], out4)
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
    build(args.project).queue().launch(server_name=args.host, server_port=args.port, css=CSS,
                                       allowed_paths=[str(proj(args.project))])


if __name__ == "__main__":
    main()
