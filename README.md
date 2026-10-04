# Artificial Defect Generation

Generate realistic synthetic **defect images** (with masks and bounding boxes) from a few real defect images plus good images, to train computer-vision inspection models: object detection, segmentation, and anomaly detection (PatchCore).

Full research write-up: **[docs/RESEARCH.md](docs/RESEARCH.md)**

## TL;DR

| Approach | Realism | Effort | Use it for |
|---|---|---|---|
| Classical augmentation (Albumentations) | Real pixels, little new variety | Very low | Always, as a base |
| Copy-paste + Poisson blending (`cv2.seamlessClone`) | Medium | Low, CPU | Fast baseline |
| Procedural anomalies (CutPaste, DRAEM, NSA) | Low–medium | Low | When you have no defect images |
| GANs (StyleGAN2-ADA, DFMGAN) | Medium–high | High | Mostly superseded by diffusion |
| **Diffusion inpainting** (Stable Diffusion + DreamBooth/LoRA, AnomalyDiffusion) | **High** | Medium, GPU | Main approach |

**Recommended pipeline:** inpaint/paste defects into **good images inside a known mask** → the mask is the label for free (segmentation mask + boxes via connected components) → quality filter → train → evaluate on a **real-only** held-out test set.

**PatchCore:** trains on good images only; use the synthetic defects to validate it and tune its threshold, not to train it.

## Quick start
**Browser UI** (all steps; import YOLO / VOC / COCO / labelme labels or draw boxes):
```bash
pip install -r requirements.txt   # install PyTorch for your CUDA first
python -m ui.app                  # open http://127.0.0.1:7860
```

**Command line:**
Step-by-step guide: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**

```bash
pip install -r requirements.txt                       # install PyTorch for your CUDA first
python scripts/labelme_to_masks.py ...                # 1. annotations -> masks
python scripts/zero_shot_inpaint.py ...               # 2. sanity test, no training
python -m src.data.prepare_crops ...                  # 3. 512px crops around defects
accelerate launch -m src.diffusion.train_lora ...     # 4. LoRA on SD inpainting (GPU)
python -m src.diffusion.inpaint_generate ...          # 5. synthetic images + masks + YOLO boxes
python -m pytest -q tests                             # self-test (tiny random model, CPU)
```

## Status
Starter code tested end-to-end on CPU with a tiny random model only; not yet validated on real defect data.
