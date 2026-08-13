#!/bin/bash
#SBATCH --job-name=squeakout-run
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out

OUT_BASE="outputs/squeakout"
CHECKPOINT=""                   # leave empty to use squeakout/squeakout_weights.ckpt
OVERWRITE="--overwrite"         # set to "--overwrite" to reprocess existing outputs

# Stage-2 now applies the cross-validated ridge gate (min_mask_cols>=9, sweep off,
# band-limited spectral centroid>=25 kHz). The centroid gate reads each recording's
# headmic wavs, so the data/ tree (audio) must be reachable from the compute node;
# these are also the run.py defaults, so no extra flags are needed. To reproduce the
# old geometry-only gate, add: --min-centroid-hz 0 --min-freq-sweep-frac 0.04 --min-mask-cols 5

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/run.py squeakout \
    "$OUT_BASE" \
    --dataset gerbil_ssl \
    --all \
    --workers 1 \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    $OVERWRITE
