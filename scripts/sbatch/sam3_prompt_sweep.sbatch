#!/bin/bash
#SBATCH --job-name=sam3-prompt-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/%x-%j.out

# Zero-shot SAM3: how much do the text prompt and the score threshold matter? Each prompt runs once at
# threshold 0.05 on the gerbil_ssl val windows; the summary then scores every threshold against the val boxes
# (window-level, greedy box match at IoU >= 0.3, as eval_sam3_ft.py). Resumable per prompt.
# Submit from the repo root: sbatch scripts/sbatch/sam3_prompt_sweep.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
J=outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json
V=outputs/sam3_prompt_sweep
PROMPTS=("trace" "traces" "white trace" "white traces" "line" "white line" "curve" "white curve"
         "squiggle" "streak" "chirp" "call" "vocalization" "ultrasonic vocalization" "whistle")
nvidia-smi --query-gpu=name --format=csv,noheader
git -C sam3 log --oneline -1

for p in "${PROMPTS[@]}"; do
    out="$V/${p// /_}.json"
    [ -f "$out" ] && continue
    python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.05 --prompt "$p" \
        --images-json "$J" --out "$out"
done

python - "$J" "$V" "${PROMPTS[@]}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/evaluation")
from eval_sam3_ft import match_stats

gt_path, v, prompts = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
gt = json.load(open(gt_path))
thrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
print(f"{len(gt['images'])} windows, {len(gt['annotations'])} GT boxes; F1 (P/R) at IoU>=0.3 per score threshold")
print(f"{'prompt':26s} " + " ".join(f"{t:>15.1f}" for t in thrs) + "   best F1 @thr")
rows = []
for p in prompts:
    preds = json.load(open(v / f"{p.replace(' ', '_')}.json"))
    cells, best = [], (0, 0)
    for t in [x / 100 for x in range(5, 96)]:
        tp, fp, fn, _, _ = match_stats(gt, preds, t, 0.3)
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        best = max(best, (f1, t))
        if round(t, 2) in thrs:
            cells.append(f"{f1:.2f} ({pr:.2f}/{rc:.2f})")
    rows.append((best, p, cells))
for best, p, cells in sorted(rows, reverse=True):
    print(f"{p:26s} " + " ".join(f"{c:>15s}" for c in cells) + f"   {best[0]:.3f} @{best[1]:.2f}")
PY
