#!/bin/bash
#SBATCH --job-name=sam3-ft-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/%x-%j.out

# Smoke test of the SAM3 "trace" fine-tuning: 2 epochs on 300 train windows (box-only, then box+mask),
# then zero-shot SAM3 vs both fine-tuned models scored on the same 300 train / 200 val windows.
# Dataset (already built): build_sam3_ft_dataset.py --out outputs/sam3_ft/smoke/data --min-labeled 0.5
#   --max-windows 300 200. Results table at the end of the log.
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_smoke.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
S=outputs/sam3_ft/smoke
nvidia-smi --query-gpu=name,memory.total --format=csv

# a run whose checkpoint already exists is not retrained
[ -f "$S/runs/box/checkpoints/checkpoint.pt" ] ||
    python sam3/sam3/train/train.py -c configs/gerbil/gerbil_trace_ft_smoke.yaml --use-cluster 0 --num-gpus 1
[ -f "$S/runs/seg/checkpoints/checkpoint.pt" ] ||
    python sam3/sam3/train/train.py -c configs/gerbil/gerbil_trace_seg_ft_smoke.yaml --use-cluster 0 --num-gpus 1

for split in train val; do
    J="$S/data/$split/_annotations.coco.json"
    python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.05 \
        --images-json "$J" --out "$S/preds/zeroshot_$split.json"
    python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/box/checkpoints/checkpoint.pt" --mask-mode box \
        --threshold 0.05 --images-json "$J" --out "$S/preds/box_$split.json"
    python scripts/segmentation/run_sam3_ft.py --ckpt "$S/runs/seg/checkpoints/checkpoint.pt" --mask-mode head \
        --threshold 0.05 --images-json "$J" --out "$S/preds/seg_$split.json"
done

for split in train val; do
    python scripts/evaluation/eval_sam3_ft.py --gt "$S/data/$split/_annotations.coco.json" \
        --pred zeroshot="$S/preds/zeroshot_$split.json" box="$S/preds/box_$split.json" seg="$S/preds/seg_$split.json"
done
