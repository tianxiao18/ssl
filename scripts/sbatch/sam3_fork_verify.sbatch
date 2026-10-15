#!/bin/bash
#SBATCH --job-name=sam3-fork-verify
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/%x-%j.out

# Check that switching sam3/ to the tianxiao18 fork changed nothing: re-run zero-shot / smoke box / smoke seg
# on the 200 smoke val windows on an H200 (same GPU as the originals) and compare with outputs/sam3_ft/smoke/preds.
# Submit from the repo root: sbatch scripts/sbatch/sam3_fork_verify.sbatch
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
V=outputs/sam3_ft/verify_fork
nvidia-smi --query-gpu=name --format=csv,noheader
git -C sam3 log --oneline -1
python -c "import sam3, sam3.device_utils as d; print(sam3.__file__, d.DEVICE)"

python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.05 --images-json "$J" --out "$V/zeroshot_val.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/box/checkpoints/checkpoint.pt" --mask-mode box --threshold 0.05 \
    --images-json "$J" --out "$V/box_val.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/seg/checkpoints/checkpoint.pt" --mask-mode head --threshold 0.05 \
    --images-json "$J" --out "$V/seg_val.json"

python - <<'PY'
import json
from collections import defaultdict

def iou(p, q):
    ix = max(0, min(p[0] + p[2], q[0] + q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[1] + p[3], q[1] + q[3]) - max(p[1], q[1]))
    i = ix * iy
    return i / (p[2] * p[3] + q[2] * q[3] - i + 1e-9)

for m in ("zeroshot", "box", "seg"):
    a = json.load(open(f"outputs/sam3_ft/smoke/preds/{m}_val.json"))
    b = json.load(open(f"outputs/sam3_ft/verify_fork/{m}_val.json"))
    ga, gb = defaultdict(list), defaultdict(list)
    for d in a: ga[d["image_id"]].append(d)
    for d in b: gb[d["image_id"]].append(d)
    ds, db, same, only_a, only_b = [], [], 0, [], []
    for k in set(ga) | set(gb):
        rest = sorted(gb[k], key=lambda d: -d["score"])
        for d in sorted(ga[k], key=lambda d: -d["score"]):
            j = max(range(len(rest)), key=lambda j: iou(d["bbox"], rest[j]["bbox"]) - abs(d["score"] - rest[j]["score"]), default=None)
            if j is None or iou(d["bbox"], rest[j]["bbox"]) < 0.5:
                only_a.append(round(d["score"], 3)); continue
            e = rest.pop(j)
            ds.append(abs(d["score"] - e["score"]))
            db.append(max(abs(x - y) for x, y in zip(d["bbox"], e["bbox"])))
            same += d["segmentation"]["counts"] == e["segmentation"]["counts"]
        only_b += [round(e["score"], 3) for e in rest]
    at = lambda L, t: sum(d["score"] >= t for d in L)
    print(f"{m}: count {len(a)} -> {len(b)} | >=0.2 {at(a, .2)} -> {at(b, .2)} | >=0.5 {at(a, .5)} -> {at(b, .5)}")
    print(f"   matched {len(ds)}: max |score diff| {max(ds, default=0):.4g}, max |bbox diff| {max(db, default=0):.4g} px, "
          f"identical masks {same}/{len(ds)}")
    print(f"   only in original (scores): {only_a}   only in fork: {only_b}")
PY
