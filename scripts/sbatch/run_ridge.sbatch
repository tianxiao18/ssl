#!/bin/bash
#SBATCH --job-name=ridge-run
##SBATCH --partition=gen          # uncomment/set to your cluster's CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/%x-%j.out

# Ridge is CPU-only (no GPU). Unlike the old image-only pipeline it now reads each
# recording's headmic wavs for the stage-2 spectral gates (centroid/flatness), so the
# data/ tree (audio) must be reachable from the compute node.

OUT_BASE="outputs/ridge_flatness"  # flatness-only run; use ridge_spectral for the centroid config
WORKERS=16                        # keep <= --cpus-per-task

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# Stage-2 uses spectral flatness as the SOLE filter (flatness<=0.21, ~0.82 F1 in
# the from-scratch filter search): the geometric column gate and the centroid gate
# are disabled so flatness alone decides. Stage-1 keeps the CV-best min_area=50.
# python scripts/run.py ridge \
#     "$OUT_BASE" \
#     --dataset gerbil_ssl \
#     --all \
#     --workers "$WORKERS" \
#     --min-area 50 \
#     --min-freq-sweep-frac 0.0 \
#     --min-mask-cols 0 \
#     --min-centroid-hz 0 \
#     --max-flatness 0.21

# Alternative: the previous CV-best n_cols>=9 & centroid>=25 kHz config (~0.89 F1).
# These are also the ridge.py defaults, so --all alone would use them.
python scripts/run.py ridge \
    "$OUT_BASE" \
    --dataset gerbil_ssl \
    --all \
    --workers "$WORKERS" \
    --min-area 50 \
    --min-mask-cols 9 \
    --min-freq-sweep-frac 0.0 \
    --min-centroid-hz 25000
