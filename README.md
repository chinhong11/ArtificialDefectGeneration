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

## Status
Research stage — no code yet. See the proposed structure in [docs/RESEARCH.md §8](docs/RESEARCH.md#8-proposed-code-structure-for-a-future-implementation).
