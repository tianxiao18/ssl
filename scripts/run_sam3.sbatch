#!/bin/bash
#SBATCH --job-name=sam3-run
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out

SPEC_BASE="outputs/spectrograms"
OUT_BASE="outputs/sam3"
CHECKPOINT="sam3/sam3.pt"        # leave empty to download from HuggingFace
OVERWRITE="--overwrite"         # set to "--overwrite" to reprocess existing outputs

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/run.py sam3 \
    "$SPEC_BASE" \
    "$OUT_BASE" \
    --all \
    --workers 1 \
    ${CHECKPOINT:+--sam3-checkpoint "$CHECKPOINT"} \
    $OVERWRITE
