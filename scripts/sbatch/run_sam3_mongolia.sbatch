#!/bin/bash
#SBATCH --job-name=sam3-mongolia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/%x-%j.out

# SAM3 over mongolia_wild_data (12 recordings, 1601 one-second windows, HDF5-backed).
# No job array: at the 0.311s/window rate measured in time_sam3_throughput.py this is
# ~8min of inference + ~8min of per-recording model load, so one GPU covers it.
#
# --sample-rate 192000 is required (default 125000) -- these are 192kHz recordings and
# the stage-2 spectral gates map mask rows to Hz with it.
#
# --freq-min 15000 --min-centroid-hz 0: calls sit ~18-40kHz with wind/handling noise
# below ~3kHz. freq-min excludes the wind; the centroid gate is DISABLED for this first
# pass because the dataset has no annotations to tune it against -- look at the raw
# detections before turning it back on.
#
# Submit:  sbatch scripts/sbatch/run_sam3_mongolia.sbatch
# Monitor: squeue -u $USER
# Resume:  just resubmit (run_sam3 skips recordings whose coco_ch_0.json exists)

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/run.py sam3 outputs/sam3_mongolia --dataset mongolia_wild_data \
    --all --channels 0 --prefix mic \
    --sample-rate 192000 --freq-min 15000 --min-centroid-hz 0 \
    --sam3-checkpoint sam3/sam3.pt

echo "Done."
