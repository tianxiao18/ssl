#!/bin/bash
#SBATCH --job-name=sam3-ft-compare
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --dependency=afterany:7240250
#SBATCH --output=/tmp/%x-%j.out

# Zero-shot ("trace", score 0.2) vs box2mask vs seg (score 0.5) vs sam3_best on the val recordings (test set):
#   zero-shot on the full val recordings (box2mask/seg full-recording runs come from their own jobs),
#   all three on the val windows for box/mask scoring, then compare_sam3_ft.py -> outputs/eval/sam3_ft_val
# Waits for 7240250 (sam3-ft-seg). Submit from the repo root: sbatch scripts/sbatch/sam3_ft_compare.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
J=outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json
BOX=outputs/sam3_ft/runs/trace_frozen_v1/checkpoints/best.pt
SEG=outputs/sam3_ft/runs/trace_seg_frozen_v1/checkpoints/best.pt
P=outputs/sam3_ft/preds_val

[ -d outputs/sam3_zeroshot_trace/gerbil_ssl ] ||  # made by sam3_ft_smoke_test.sbatch
    python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.2 \
        --split-json "$J" --out outputs/sam3_zeroshot_trace

[ -f "$P/zeroshot.json" ] ||
    python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.2 \
        --images-json "$J" --out "$P/zeroshot.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$BOX" --mask-mode box --images-json "$J" --out "$P/box2mask.json"
python scripts/segmentation/run_sam3_ft.py --ckpt "$SEG" --mask-mode head --images-json "$J" --out "$P/seg.json"

python scripts/evaluation/compare_sam3_ft.py --split-json "$J" --out outputs/eval/sam3_ft_val \
    --det sam3-box=outputs/sam3_best/gerbil_ssl@0 sam3-text=outputs/sam3_zeroshot_trace/gerbil_ssl@0.2 \
          sam3-text-boxft=outputs/sam3_ft_box2mask/gerbil_ssl sam3-text-maskft=outputs/sam3_ft_head/gerbil_ssl \
    --win sam3-text="$P/zeroshot.json@0.2" sam3-text-boxft="$P/box2mask.json" sam3-text-maskft="$P/seg.json"
