#!/bin/bash
#SBATCH --job-name=sam3-ft-verify
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/%x-%j.out

# Check that moving the fine-tune configs out of sam3/ and stashing sam3's local edits changed nothing:
# re-run zero-shot / smoke box / smoke seg on the 200 smoke val windows exactly as sam3_ft_smoke.sbatch did
# (threshold 0.05) and compare with the saved predictions in outputs/sam3_ft/smoke/preds.
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_verify.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
S=outputs/sam3_ft/smoke
J=$S/data/val/_annotations.coco.json
V=outputs/sam3_ft/verify
nvidia-smi --query-gpu=name --format=csv,noheader

python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.05 --images-json "$J" --out "$V/zeroshot_val.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/box/checkpoints/checkpoint.pt" --mask-mode box --threshold 0.05 \
    --images-json "$J" --out "$V/box_val.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/seg/checkpoints/checkpoint.pt" --mask-mode head --threshold 0.05 \
    --images-json "$J" --out "$V/seg_val.json"

python - <<'PY'
import json
import numpy as np
from pycocotools import mask as mu
for m in ("zeroshot", "box", "seg"):
    a = json.load(open(f"outputs/sam3_ft/smoke/preds/{m}_val.json"))
    b = json.load(open(f"outputs/sam3_ft/verify/{m}_val.json"))
    if len(a) != len(b):
        print(f"{m}: DIFFERENT detection count {len(a)} -> {len(b)}")
        continue
    key = lambda d: (d["image_id"], -d["score"], d["bbox"])
    a, b = sorted(a, key=key), sorted(b, key=key)
    db = max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p["bbox"], q["bbox"]))
    ds = max(abs(p["score"] - q["score"]) for p, q in zip(a, b))
    same_mask = sum(p["segmentation"]["counts"] == q["segmentation"]["counts"] for p, q in zip(a, b))
    print(f"{m}: {len(a)} detections, max |bbox diff| {db:.4g} px, max |score diff| {ds:.4g}, "
          f"identical masks {same_mask}/{len(a)}")
PY
