#!/bin/bash
#SBATCH --job-name=sam3-ft-runpy-check
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/%x-%j.out

# Check that `run.py sam3_ft_seg|sam3_ft_box2mask` (weights-only exports in checkpoints/) reproduce
# run_sam3_ft.py's dryad_gerbil_full outputs on one recording.
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_runpy_check.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
REC=cohort2_formatted/2020_08_06_21_38_40_073129_merged
V=outputs/sam3_ft_runpy_check
nvidia-smi --query-gpu=name --format=csv,noheader
git -C sam3 log --oneline -1

python scripts/run.py sam3_ft_seg "$V/seg" --dataset dryad_gerbil_full --prefix mic --recording "$REC" --overwrite
python scripts/run.py sam3_ft_box2mask "$V/box2mask" --dataset dryad_gerbil_full --prefix mic --recording "$REC" --overwrite

python - "$REC" "$V" <<'PY'
import json, sys
from collections import Counter
rec, v = sys.argv[1:]
for new, old in (("seg", "sam3_text_maskft_dryad"), ("box2mask", "sam3_text_boxft_dryad")):
    a = json.load(open(f"outputs/{old}/dryad_gerbil_full/{rec}/coco_ch_0.json"))
    b = json.load(open(f"{v}/{new}/dryad_gerbil_full/{rec}/coco_ch_0.json"))
    ka = Counter((x["image_id"], round(x["score"], 2)) for x in a["annotations"])
    kb = Counter((x["image_id"], round(x["score"], 2)) for x in b["annotations"])
    print(f"{new}: windows {len(a['images'])} -> {len(b['images'])}, detections {len(a['annotations'])} -> "
          f"{len(b['annotations'])}, (window, score) pairs differing: {sum(((ka - kb) + (kb - ka)).values())}")
PY
