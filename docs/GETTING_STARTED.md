# Getting Started — Diffusion Inpainting Defect Generation

How to go from "I have defect photos without masks" to "I have hundreds of synthetic defect images with masks + boxes". Background on why this approach: [RESEARCH.md](RESEARCH.md).

**Rule #1: start with ONE product and ONE defect type.** Get it working end-to-end, then add more.

```
 Step 1        Step 2         Step 3          Step 4         Step 5          Step 6
annotate  →  zero-shot   →  training    →  train LoRA  →  generate    →  check +
 masks        sanity test     crops           (GPU)          defects        A/B test
```

---

## Step 0 — Environment

Python 3.10+ and a CUDA GPU (≥12 GB VRAM).

```bash
python -m venv .venv && source .venv/bin/activate
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m pytest -q tests        # quick self-test on a tiny random model (CPU, ~1 min)
```

The first real run downloads the base model `stable-diffusion-v1-5/stable-diffusion-inpainting` (several GB) from Hugging Face. Check its license (CreativeML OpenRAIL-M) fits your use.

---

## Easiest: use the browser UI

```bash
python -m ui.app          # then open http://127.0.0.1:7860 in your browser
```
Everything runs on your own PC; images never leave it. Work through the tabs left to right:

| Tab | What you do |
|---|---|
| 1 · Images | Upload good images and defect images |
| 2 · Annotate | **Defect images:** drag a box around each defect (drag corners to resize, mouse wheel to zoom, *Space* resets zoom, hand tool to pan/select, *Delete* removes the selected box). Boxes save automatically with the label in the textbox. **Good images:** click around the part, then *Finish outline* |
| 3 · Masks | *Make masks* (SAM = exact outline, or filled boxes) and check the red outlines |
| 4 · Train | Pick steps (200 = quick check, 1000–3000 = real) → *Start training*; previews appear while it trains |
| 5 · Generate | Pick how many images and defect shapes → *Generate*; results show mask (red) + YOLO box (green) |
| 6 · Export | *Create zip* → images + masks + YOLO labels |

Annotations are saved as labelme JSON next to each image, so you can also open them in labelme. The command-line steps below do the same thing as the UI.

---

## Quick check: test everything on a few images first

Before annotating a full dataset, run the whole pipeline once on ~5 images with the real models:

1. Put images in `data/<product>/good/` (2+) and `data/<product>/defect_raw/` (3+).
2. In labelme: draw a **rectangle** around each defect (one label, e.g. `defect`) on the defect images, and a **polygon labelled `roi`** around the part on each good image. Save (JSON next to each image).
3. Run:
   ```bash
   python scripts/run_real_test.py --data data/<product>
   ```
4. Look at / send back `outputs/real_test/`: `01_sam_masks.png`, `02_crops.png`, `03_zero_shot.png`, `04_lora_samples.png`, `05_lora_generated.png`, `report.txt` (and `log.txt` if something failed).

It runs a short LoRA training (200 steps on GPU) — enough to prove everything works, not enough for good quality.

Options: `--no-sam` (use the filled boxes as masks), `--skip-train`, `--steps N`, `--sd-model` / `--sam-model <local folder>` if Hugging Face is blocked on your network.

---

## Step 1 — Data and masks (most important step)

### 1a. Folder layout
```
data/metal_part/                 # one folder per product
├── good/                        # defect-free images (as many as you have)
├── defect_raw/                  # defect images to annotate (+ labelme .json)
├── defect/scratch/              # defect images per type   (filled by step 1c)
└── masks/scratch/               # masks, same file name    (filled by step 1c)
```

### 1b. Split before anything else
Put aside ~20–30 % of real defect images as a **real test set** now (e.g. `data/metal_part/test/`). Never annotate-and-train or generate from them. They are how you'll prove the synthetic data helps.

### 1c. Make masks — pick one way

**Option A — polygons in labelme (most accurate)**
```bash
pip install labelme
labelme data/metal_part/defect_raw     # draw polygons, label "scratch", save (JSON next to image)
python scripts/labelme_to_masks.py --json-dir data/metal_part/defect_raw \
    --out-dir data/metal_part/masks --copy-images-to data/metal_part/defect
```
Use "Create LineStrip" for thin scratches/cracks (drawn with `--line-width`, default 5 px).

**Option B — boxes + SAM (fastest)**
Draw only rectangles in labelme, then let Segment Anything (SAM) find the defect pixels inside each box:
```bash
pip install labelme
labelme data/metal_part/defect_raw     # "Create Rectangle", label "scratch", save (JSON next to image)
python scripts/sam_box_to_mask.py --json-dir data/metal_part/defect_raw \
    --out-dir data/metal_part/masks --copy-images-to data/metal_part/defect --fallback-box
```
- Box tightly around the defect (a few px margin). One box per defect; several boxes per image are fine.
- The mask is always clipped to the box (+`--box-margin`, default 4 px), so SAM can't spill onto the background.
- `--fallback-box` uses the filled box when SAM finds nothing — check those, a filled box is a poor mask.
- First run downloads `facebook/sam-vit-base` (~375 MB). If masks are poor, try `--model facebook/sam-vit-large` or `facebook/sam-vit-huge` (slower, often better on fine defects).
- Works on CPU (slow) or GPU.
- SAM is trained on natural photos; on faint, low-contrast or very thin defects (fine scratches, light stains) it may grab too much or too little. Fix those few in labelme with polygons (Option A) — mixing both is fine.

**Always review the masks:**
```bash
python scripts/make_grid.py --images data/metal_part/defect/scratch \
    --masks data/metal_part/masks/scratch --overlay --out outputs/check_masks.png
```
Mask tips: cover the whole visible defect plus 1–3 px; consistent style matters more than pixel perfection. **20–50 masks** of one type is a good start (5–10 can work but expect less variety).

---

## Step 2 — Zero-shot sanity test (no training)

```bash
python scripts/zero_shot_inpaint.py --good-dir data/metal_part/good \
    --prompt "a thin scratch on a metal surface" --out outputs/zero_shot --num 10
python scripts/make_grid.py --images outputs/zero_shot/images --masks outputs/zero_shot/masks \
    --overlay --boxes --out outputs/zero_shot_grid.png
```
Expect unrealistic defects — that's the point: it confirms the pipeline (window, mask, paste-back, labels) works and gives a "before" picture to compare the LoRA against.

---

## Step 3 — Training crops

```bash
python -m src.data.prepare_crops \
    --images data/metal_part/defect/scratch --masks data/metal_part/masks/scratch \
    --out data/metal_part/crops/scratch --caption "a photo of sks scratch" --crops-per-image 4
```
- Cuts 512×512 windows at **native resolution** around each defect at random offsets (small defects stay visible).
- `sks` is a rare token the model will associate with *your* defect. Use a different rare word per defect type (`sks scratch`, `xqz dent`...) or one LoRA per type.
- If your images are much bigger than the defects (e.g. 4000 px image, 20 px defect), 512 native crops are right. If images are smaller than 512, the crop is the whole image resized.

---

## Step 4 — Train the LoRA (GPU)

```bash
accelerate launch -m src.diffusion.train_lora \
    --data data/metal_part/crops/scratch --output outputs/lora_scratch \
    --val-good-dir data/metal_part/good \
    --max-steps 2000 --rank 8 --lr 1e-4 \
    --mixed-precision fp16 --gradient-checkpointing
```
(Run `accelerate config` once first, or just use `python -m src.diffusion.train_lora ...` on a single GPU.)

What it does: freezes SD, adds LoRA adapters to the UNet, and trains on *[noisy latent | defect mask | image with defect masked out]* so the model learns "paint this defect inside this region, matching the surrounding surface". Loss inside the mask is weighted ×2 (`--mask-loss-weight`).

**Pick the checkpoint by eye:** `outputs/lora_scratch/samples/step_*.png` shows good-image crops (left, mask in red) and the painted defect (right).

| You see | Meaning | Try |
|---|---|---|
| Defect doesn't look like yours | Under-trained | More steps, higher rank (16), check captions |
| Exact copies of training defects / weird color patches | Over-trained | Earlier checkpoint (`checkpoint-N`), lower lr (5e-5), fewer steps |
| Out of memory | | `--use-8bit-adam`, batch 1, keep `--gradient-checkpointing` |
| Seam/halo around defect | | Larger `--mask-dilate` in generation, `--feather` |

---

## Step 5 — Generate defects

```bash
python -m src.diffusion.inpaint_generate \
    --good-dir data/metal_part/good --lora outputs/lora_scratch \
    --prompt "a photo of sks scratch" \
    --mask-mode mixed --real-mask-dir data/metal_part/crops/scratch/masks \
    --num 200 --out outputs/gen_scratch --class-id 0
```
Per sample it: picks a 512 window on a good image → makes a mask (`real` = deformed real defect shapes, `stroke` = scratches/cracks, `blob` = stains/dents; `mixed` = all) → inpaints → blends back (pixels outside the mask stay identical) → **refines the label** to the pixels that actually changed → drops samples where nothing changed.

Output:
```
outputs/gen_scratch/images/*.png    full-resolution synthetic image
outputs/gen_scratch/masks/*.png     segmentation mask
outputs/gen_scratch/labels/*.txt    YOLO boxes
outputs/gen_scratch/metadata.jsonl  source image, window, mask mode, boxes
```
COCO JSON (boxes + polygons) if you need it:
```bash
python -m src.export --images outputs/gen_scratch/images --masks outputs/gen_scratch/masks \
    --class-id 0 --class-name scratch --coco outputs/gen_scratch/coco.json
```

**ROI (needed when images show background):** without it, defects can be painted onto the empty background. In labelme, draw one polygon labelled `roi` around the part on each good image, then:
```bash
python scripts/labelme_to_masks.py --json-dir data/metal_part/good --out-dir data/metal_part/roi_masks
# then add to the generate command:  --roi-dir data/metal_part/roi_masks/roi
```
If parts are always in the same position, you can draw it once and copy the mask PNG for every good image name.

Useful options:
- `--mask-mode stroke` for scratches/cracks, `blob` for stains/dents.
- `--lora-scale 0.6–1.0` — lower = more "generic", higher = more like training defects.
- `--diff-threshold` — raise it if labels include faint background changes, lower it for subtle defects.

---

## Step 6 — Check, then prove it helps

1. Look at it: `python scripts/make_grid.py --images outputs/gen_scratch/images --masks outputs/gen_scratch/masks --overlay --boxes --out outputs/gen_grid.png`, next to a grid of real defects. Delete obvious failures.
2. **A/B test** with your detector/segmenter, same settings, evaluated on the **real test set from Step 1b**:
   - A: real training images only
   - B: real + synthetic (start around 1:1, then try more/less)
3. Keep the synthetic data only if B beats A on the real test set (recall / mAP / IoU).

**PatchCore:** train it on real good images only (as usual). Use the synthetic defects + masks as an extra validation set to tune its threshold, then confirm the threshold on real defects.

---

## Step 7 — Compare with AnomalyDiffusion

Read *AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model* (Hu et al., AAAI 2024). Its official code is linked from the paper. It learns defect appearance **and** mask distribution jointly; running it on the same data (or on MVTec AD first) gives a reference point for how good this pipeline's output is.

---

## Repo map

| File | Step | Needs GPU |
|---|---|---|
| `scripts/labelme_to_masks.py` | 1 labelme JSON → masks | no |
| `scripts/sam_box_to_mask.py` | 1 boxes → masks (SAM) | optional |
| `scripts/make_grid.py` | review anything | no |
| `scripts/zero_shot_inpaint.py` | 2 sanity test | yes (CPU is very slow) |
| `src/data/prepare_crops.py` | 3 training crops | no |
| `src/diffusion/train_lora.py` | 4 LoRA training | yes |
| `src/mask_gen.py` | 5 new defect masks | no |
| `src/diffusion/inpaint_generate.py` | 5 generation | yes |
| `src/export.py` | 5 YOLO / COCO export | no |
| `ui/app.py` | browser UI for all steps | for train/generate |
| `tests/` | self-test with a tiny random model | no |

## Honest limits
- The code is tested end-to-end only with a tiny random model on CPU (tests/); `sam_box_to_mask.py` is tested only with a tiny random SAM model, so its mask quality on your defects is unknown until you run it. Training time and result quality on real data have **not** been measured yet — Step 4 hyper-parameters are common starting points, expect to tune them.
- The refined mask is "pixels that changed", which can include slight shading around the defect; check grids.
- Synthetic data supplements real data; it can't fix a test set that the model has never seen anything similar to.
