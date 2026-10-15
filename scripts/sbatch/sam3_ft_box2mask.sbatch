#!/bin/bash
#SBATCH --job-name=sam3-ft-box2mask
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/%x-%j.out

# Val-set inference with the box fine-tune from sam3_ft_trace.sbatch (trace_frozen_v1/best.pt):
# its boxes prompted into SAM3's box->mask predictor -> outputs/sam3_ft_box2mask/gerbil_ssl
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_box2mask.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
DATA=outputs/sam3_ft/gerbil_ssl
RUNS=outputs/sam3_ft/runs

python scripts/segmentation/run_sam3_ft.py --ckpt "$RUNS/trace_frozen_v1/checkpoints/best.pt" \
    --mask-mode box --split-json "$DATA/val/_annotations.coco.json" --out outputs/sam3_ft_box2mask
