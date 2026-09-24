#!/bin/bash
#SBATCH --job-name=time-sam3-throughput
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=/tmp/%x-%j.out

# Diagnostic: measure real SAM3 steady-state throughput (imports+model-load cost
# separated from per-window inference cost, reported in rolling batches) on a
# full 1200-window recording, before sizing the full dryad_gerbil_full run.
# The earlier smoke test's "1.14s/window" was total-job-wall-time / window-count,
# which conflates one-time overhead with per-window cost -- this isolates them.
#
# Submit:  sbatch scripts/sbatch/time_sam3_throughput.sbatch

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/experimental/time_sam3_throughput.py \
    outputs/spectrograms_h5/dryad_gerbil_full/cohort2_formatted/2020_07_19_16_32_55_857727_merged \
    --channels 0 --prefix mic --freq-min 500 --min-centroid-hz 0 \
    --sam3-checkpoint sam3/sam3.pt --batch 100
