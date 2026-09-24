#!/bin/bash
#SBATCH --job-name=ridge-mongolia
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/tmp/%x-%j.out

# Ridge over mongolia_wild_data (12 recordings, 1601 windows, HDF5-backed), one worker
# per recording. Same settings as run_sam3_mongolia.sbatch: 192 kHz audio, freq-min
# 15000 to drop wind, centroid gate off (no annotations to tune it against).
#
# Submit:  sbatch scripts/sbatch/run_ridge_mongolia.sbatch

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

OMP_NUM_THREADS=1 python scripts/run.py ridge outputs/ridge --dataset mongolia_wild_data \
    --all --workers "$SLURM_CPUS_PER_TASK" --channels 0 --prefix mic \
    --sample-rate 192000 --freq-min 15000 --min-centroid-hz 0

echo "Done."
