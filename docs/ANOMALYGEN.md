# Using NVIDIA Cosmos-AnomalyGen (second generator)

AnomalyGen is NVIDIA's defect-image model (Cosmos-Predict2-2B fine-tuned per use case).
It runs inside NVIDIA's Docker image `nvcr.io/nvidia/paidf-anomalygen:1.0.1`.
Commands below follow NVIDIA's skill `skills/data/tao-generate-anomalies` in
[NVIDIA-TAO/tao-skill-bank](https://github.com/NVIDIA-TAO/tao-skill-bank).
Engine source: [NVIDIA/paidf-anomalygen](https://github.com/NVIDIA/paidf-anomalygen).

Verified so far on Windows 11 + WSL2 Ubuntu + Docker Desktop + RTX 3090 24 GB, 32 GB RAM:
setup, model download, a 500-step fine-tune (~48 s/step) and generating 3 images (~5 s each).
Not yet checked: longer training, quality at higher steps.

Lessons from that run:
- WSL's default RAM (half of the PC) was too small; the container died during validation.
  Fix: `C:\Users\<you>\.wslconfig` with `[wsl2]` / `memory=24GB` / `swap=16GB`, then `wsl --shutdown`.
- Keep the Ubuntu distro on an SSD if possible; on an HDD, model loading and swap are slow.

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

## 4. Train a model (inside the container: `agshell`)
Each run gets its own `NAME`, so runs never overwrite each other. Training always starts from
NVIDIA's base model (the 1.0.1 scripts cannot resume a previous run).

```bash
NAME=disc_v2                       # new name per run
DATASET_DIR=/work/home/datasets/disc
DEFECT_DESC=$DATASET_DIR/defect_spec.jsonl
MAX_ITER=5000; EVERY=1000          # NVIDIA's own example configs use 15000; default 75000

# 4a. validation cases (9 = 3 per defect type; fewer images per check = less memory)
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh --name validation_$NAME --num-sdg 9 \
  --dataset-dir $DATASET_DIR --clean-dir $DATASET_DIR --defect-spec $DEFECT_DESC \
  --amp-output-dir ag_inference/validation_$NAME/amp \
  --output-jsonl ag_inference/validation_$NAME/testcase.jsonl --mode validation

# 4b. training settings
python3 -m scripts.utilities.generate_config --name $NAME --dataset-dir $DATASET_DIR \
  --defect-spec $DEFECT_DESC --validation-jsonl ag_inference/validation_$NAME/testcase.jsonl \
  --output ag_configs/$NAME.yaml --model-size 2b --max-iter $MAX_ITER \
  --save-iter $EVERY --validation-iter $EVERY --lr 0.02 --batch-size 2 --image-size 512

# 4c. start (keeps running after you close the window)
nohup ${ANOMALYGEN_SCRIPTS}/launch_training.sh --ag-config ag_configs/$NAME.yaml \
  --num-gpus 1 --model-size 2b > results/train_$NAME.log 2>&1 &
```
Watch from Ubuntu: `tail -f ~/anomalygen/run/results/train_disc_v2.log`.
Stop early: `pkill -f ag_train` (inside the container). The last saved step stays usable.

## 5. Where everything is saved
Ubuntu path `~/anomalygen/run/...` = container path `/workspace/paidf-anomalygen/...`.

| What | Path |
|---|---|
| Validation cases | `ag_inference/validation_<NAME>/` |
| Training settings | `ag_configs/<NAME>.yaml` |
| Training log | `results/train_<NAME>.log` |
| Model checkpoints | `results/anomaly_gen/<NAME>/<NAME>_training_FP32_lr0.02_bs=2_2B_512x512/checkpoints/` (`latest_checkpoint.txt` names the newest, e.g. `iter_000001000.pt`) |
| Validation images per step | same folder, `valid/<step>/` |
| Validation scores (`nn_score`) | same folder, `stdout.log` |
| Generated images | `results/<NAME>_s<STEP>/original/` (+ `original_daft_v3/`) |
| Scores of generated images | `results/<NAME>_s<STEP>/original/{per_sample.csv, eval.log, SDG_result.csv}` |

Scores during training: `grep -A9 "nn_score" .../stdout.log | grep Average`. Higher is better;
pick the step with the highest average `nn_score`. NVIDIA's default keep-threshold is 0.4.

## 6. Generate and test one checkpoint
```bash
NAME=disc_v2; STEP=3000; NUM=30    # STEP must be a saved step
DATASET_DIR=/work/home/datasets/disc
DEFECT_DESC=$DATASET_DIR/defect_spec.jsonl
CKPT=$(ls -d results/anomaly_gen/$NAME/${NAME}_training_*)
GEN=${NAME}_s$STEP

${ANOMALYGEN_SCRIPTS}/prep_testcase.sh --name $GEN --num-sdg $NUM \
  --dataset-dir $DATASET_DIR --clean-dir $DATASET_DIR --defect-spec $DEFECT_DESC \
  --amp-output-dir ag_inference/$GEN/amp --output-jsonl ag_inference/$GEN/testcase.jsonl
${ANOMALYGEN_SCRIPTS}/run_sdg.sh --checkpoint_dir $CKPT --step $STEP \
  --input_jsonl ag_inference/$GEN/testcase.jsonl --output_dir results/$GEN/original \
  --model_size 2b --num_gpus 1
${ANOMALYGEN_SCRIPTS}/run_eval.sh --real-path $DATASET_DIR --generated-path results/$GEN/original \
  --anomaly-types Disc+Additional Disc+ChippedOff Disc+ForeignM
```
Check: images in Explorer (`\\wsl.localhost\Ubuntu\home\<you>\anomalygen\run\results\<GEN>\original`),
`eval.log` averages, and `SDG_result.csv` — drop rows with `guardrail_pass = 0` (blacked-out images).
NVIDIA's optional extra phases (per-sample search, filter + regenerate below `nn_threshold`) are in
the skill's `SKILL.md`, phases 5-7.
