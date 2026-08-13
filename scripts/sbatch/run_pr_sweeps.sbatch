#!/bin/bash
#SBATCH --job-name=pr-sweeps
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/tmp/%x-%j.out
#
# Permissive regeneration of all three detectors for the precision-recall sweep.
# Each method is run ONCE at a permissive threshold and exports a per-detection
# `score`; scripts/pr_curves.py then traces the curve post-hoc (no re-running).
#
# Writes to *_pr dirs — the deployed outputs/{ridge,sam3,squeakout} are left
# untouched (their tuned operating point can't be recovered from the permissive
# run for the per-pixel-threshold methods, so both must coexist).
#
# Submit:  sbatch scripts/run_pr_sweeps.sbatch

CHECKPOINT="sam3/sam3.pt"                 # leave empty to download from HuggingFace
GT_CSV="data/gerbil_ssl/sampled_contiguous_detections.csv"

# permissive thresholds (low → large superset → high-recall arm of the curve)
RIDGE_THRESHOLD_PCT=90                     # vs deployed 99
SQUEAKOUT_MASK_THRESHOLD=0.1              # vs deployed ~0.51
# sam3: open the confidence gate fully
SAM3_SCORE_THRESHOLD=0.0                   # vs deployed 0.5

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# echo "=== squeakout (GPU, mask_threshold=${SQUEAKOUT_MASK_THRESHOLD}) → outputs/squeakout_pr ==="
python scripts/run.py squeakout outputs/squeakout_pr \
    --dataset gerbil_ssl \
    --all --workers 1 \
    --mask-threshold "$SQUEAKOUT_MASK_THRESHOLD"

# echo "=== sam3 (GPU, score_threshold=${SAM3_SCORE_THRESHOLD}) → outputs/sam3_pr ==="
# python scripts/run.py sam3 outputs/sam3_pr \
#     --dataset gerbil_ssl \
#     --all --workers 1 \
#     ${CHECKPOINT:+--sam3-checkpoint "$CHECKPOINT"} \
#     --score-threshold "$SAM3_SCORE_THRESHOLD" \
#     --overwrite

echo "=== DONE ==="
