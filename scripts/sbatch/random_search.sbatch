#!/bin/bash
#SBATCH --job-name=sam3-search
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out

SPEC_BASE="outputs/spectrograms/gerbil_ssl"
RECORDING_BASE="data/gerbil_ssl"      # holds <rel>/*annotations_gt.csv
OUT_DIR="outputs/random_search"
CHECKPOINT="sam3/sam3.pt"             # leave empty to download from HuggingFace

N_SESSIONS=20                         # random sessions across ALL experiments (seeded)
N_TRIALS=40                           # stage-1 GPU passes
STAGE2_PER_TRIAL=10                   # cheap stage-2 samples per pass (incl. score_threshold)
OBJECTIVE="f1"                        # f1 | precision | recall
SEED=0

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/hparam_search/random_search.py \
    "$SPEC_BASE" \
    "$RECORDING_BASE" \
    "$OUT_DIR" \
    --n-sessions "$N_SESSIONS" \
    --n-trials "$N_TRIALS" \
    --stage2-per-trial "$STAGE2_PER_TRIAL" \
    --objective "$OBJECTIVE" \
    --seed "$SEED" \
    ${CHECKPOINT:+--sam3-checkpoint "$CHECKPOINT"}
