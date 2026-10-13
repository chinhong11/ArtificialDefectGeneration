# Using NVIDIA Cosmos-AnomalyGen (second generator)

AnomalyGen is NVIDIA's defect-image model (Cosmos-Predict2-2B fine-tuned per use case).
It runs inside NVIDIA's Docker image `nvcr.io/nvidia/paidf-anomalygen:1.0.1`.
Commands below follow NVIDIA's skill `skills/data/tao-generate-anomalies` in
[NVIDIA-TAO/tao-skill-bank](https://github.com/NVIDIA-TAO/tao-skill-bank).
Engine source: [NVIDIA/paidf-anomalygen](https://github.com/NVIDIA/paidf-anomalygen).

Verified so far on Windows 11 + WSL2 Ubuntu + Docker Desktop + RTX 3090 24 GB:
setup and model download. Training and generation are not verified yet.

## 1. One-time setup (Windows)
1. NVIDIA Windows driver **570 or newer** (the image needs CUDA >= 12.8). Do not install a driver inside WSL.
2. `wsl --install`; keep the Ubuntu distro on a drive with 100 GB+ free
   (`wsl --manage Ubuntu --move D:\WSL\Ubuntu`).
3. Docker Desktop with the WSL 2 engine and Ubuntu integration on; disk image on a large drive.
4. Test: `docker run --rm --gpus all ubuntu nvidia-smi` shows the GPU.
5. Hugging Face read token, and accept the licence on **each** gated model page:
   `nvidia/Cosmos-Predict2-2B-Text2Image`, `nvidia/Cosmos-Guardrail1`.

## 2. Container and models (Ubuntu)
```bash
mkdir -p ~/anomalygen && cd ~/anomalygen
nano .env                    # HF_TOKEN=hf_...      then: chmod 600 .env
docker pull nvcr.io/nvidia/paidf-anomalygen:1.0.1      # ~32 GB
```
Start the container with `start.sh` (creates it on first use; full `docker run` flags in NVIDIA's
`references/docker.md`: run as your uid, mount /etc/passwd + /etc/group, writable caches):
the work folder `~/anomalygen/run/{home,checkpoints,results,ag_inference,ag_configs}` is mounted;
`~/anomalygen/run/home` appears as `/work/home` inside.

```bash
docker exec -w /workspace/paidf-anomalygen agrun bash -lc \
  '${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh --model-sizes 2B'    # ~40 GB
docker exec -w /workspace/paidf-anomalygen agrun bash -lc \
  '${ANOMALYGEN_SCRIPTS}/check.sh --model-sizes 2B'
```

## 3. Build the dataset from your images
AnomalyGen expects:
```
<dataset>/<Texture>/anomaly_image/<type>/image_001.png
<dataset>/<Texture>/mask/<type>/image_001_mask.png     binary, same size
<dataset>/<Texture>/clean_image/*.png                  good images
<dataset>/defect_spec.jsonl
```
`scripts/export_anomalygen.py` builds this from defect images + labels (YOLO box / YOLO-seg /
labelme, or ready-made masks) and good images. It needs only numpy + Pillow, so run it inside
the container:

```bash
# Ubuntu: copy the script and your images into the shared folder
mkdir -p ~/anomalygen/run/home/raw
cp export_anomalygen.py ~/anomalygen/run/home/
cp -r /mnt/d/my_data/defect /mnt/d/my_data/good ~/anomalygen/run/home/raw/

docker exec agrun python3 /work/home/export_anomalygen.py \
  --defects /work/home/raw/defect --good /work/home/raw/good \
  --out /work/home/datasets/connector --texture Connector \
  --roi-prompt "the transparent plastic connector part, excluding the background"

docker exec -w /workspace/paidf-anomalygen agrun \
  python3 -m scripts.utilities.validate_dataset /work/home/datasets/connector
```
Notes:
- Boxes become filled rectangles (include background); polygons or SAM masks are better.
- NVIDIA recommends >= 3 defect images per type for a meaningful `nn_score`.
- `--spatial text` (default) lets AnomalyGen find the part from `--roi-prompt` (Qwen-VL + SAM2),
  so defects are placed on the part, not the background.
