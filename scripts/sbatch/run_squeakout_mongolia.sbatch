#!/bin/bash
#SBATCH --job-name=squeakout-mongolia
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100"
# (constraint: venv torch cu126 has no Blackwell sm_120 kernels)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/tmp/%x-%j.out

# SqueakOut over mongolia_wild_data (12 recordings, 1601 windows, HDF5-backed).
# Centroid gate off, as in run_sam3_mongolia.sbatch. SqueakOut has no freq-min, so
# unlike ridge/SAM3 it is not told to ignore the <3 kHz wind band.
#
# Submit:  sbatch scripts/sbatch/run_squeakout_mongolia.sbatch

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/run.py squeakout outputs/squeakout --dataset mongolia_wild_data \
    --all --workers 1 --channels 0 --prefix mic \
    --min-centroid-hz 0

echo "Done."
