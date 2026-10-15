#!/bin/bash
#SBATCH --job-name=sam3-ft-seg
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out

# Seg variant of sam3_ft_trace.sbatch: box + mask loss fine-tune (configs/gerbil/gerbil_trace_seg_ft.yaml),
# then val-set inference with its own mask head -> outputs/sam3_ft_head/gerbil_ssl
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_seg.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"  # train.py adds it to sys.path, for vox_tracer.sam3_ft
DATA=outputs/sam3_ft/gerbil_ssl
RUNS=outputs/sam3_ft/runs

python sam3/sam3/train/train.py -c configs/gerbil/gerbil_trace_seg_ft.yaml --use-cluster 0 --num-gpus 1 || true
python scripts/segmentation/run_sam3_ft.py --ckpt "$RUNS/trace_seg_frozen_v1/checkpoints/best.pt" \
    --mask-mode head --split-json "$DATA/val/_annotations.coco.json" --out outputs/sam3_ft_head
