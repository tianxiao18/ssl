#!/bin/bash
#SBATCH --job-name=sam3-ft-smoke-test
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/%x-%j.out

# Zero-shot ("trace", score 0.2) and the smoke box/seg models (score 0.5) on the 31 test (val) recordings:
# full recordings for detection, test windows for box/mask, then compare_sam3_ft.py with sam3_best
# -> outputs/eval/sam3_ft_smoke_test. The zero-shot outputs are reused by sam3_ft_compare.sbatch.
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_smoke_test.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
J=outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json
S=outputs/sam3_ft/smoke
P=outputs/sam3_ft/preds_val

python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.2 \
    --split-json "$J" --out outputs/sam3_zeroshot_trace
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/box/checkpoints/checkpoint.pt" --mask-mode box \
    --split-json "$J" --out outputs/sam3_ft_smoke_box2mask
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/seg/checkpoints/checkpoint.pt" --mask-mode head \
    --split-json "$J" --out outputs/sam3_ft_smoke_head

python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.2 \
    --images-json "$J" --out "$P/zeroshot.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/box/checkpoints/checkpoint.pt" --mask-mode box \
    --images-json "$J" --out "$S/preds_test/box2mask.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/seg/checkpoints/checkpoint.pt" --mask-mode head \
    --images-json "$J" --out "$S/preds_test/seg.json"

python scripts/evaluation/compare_sam3_ft.py --split-json "$J" --out outputs/eval/sam3_ft_smoke_test \
    --det zero-shot=outputs/sam3_zeroshot_trace/gerbil_ssl@0.2 box2mask=outputs/sam3_ft_smoke_box2mask/gerbil_ssl \
          seg=outputs/sam3_ft_smoke_head/gerbil_ssl sam3_best=outputs/sam3_best/gerbil_ssl@0 \
    --win zero-shot="$P/zeroshot.json@0.2" box2mask="$S/preds_test/box2mask.json" seg="$S/preds_test/seg.json"
