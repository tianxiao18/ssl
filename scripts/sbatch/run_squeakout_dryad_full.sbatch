#!/bin/bash
#SBATCH --job-name=squeakout-dryad-full
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100"
# (constraint: venv torch cu126 has no Blackwell sm_120 kernels)
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/tmp/%x-%A_%a.out

# SqueakOut over dryad_gerbil_full (1881 recordings, ~3.88M windows, HDF5-backed), one
# array task per SAM3 shard (outputs/sam3_shards/shard_NN.txt).
#
# --gpus=1 on A100 (MIG slice is fine) or H100: SqueakOut is a small UNet, unlike SAM3.
# rtx_pro_6000_blackwell nodes fail with "no kernel image" (job 7097272 shards 5,6,7,15).
# Sizing: gerbil_ssl ran at ~0.014 s/window on one GPU incl. stage-2 gating ->
# ~242k windows/shard ~= 1h, plus a few s of model load per recording.
#
# Stage-2 matches the SAM3 dryad runs: min-centroid-hz 0 (gerbil calls span
# ~500 Hz-62.5 kHz), min-mask-cols 9 (default).
#
# Submit:  sbatch scripts/sbatch/run_squeakout_dryad_full.sbatch
# Resume:  resubmit (run_squeakout skips recordings whose coco_ch_0.json exists)

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

SHARD_FILE=$(printf "outputs/sam3_shards/shard_%02d.txt" "$SLURM_ARRAY_TASK_ID")
echo "Shard $SLURM_ARRAY_TASK_ID: $(wc -l < "$SHARD_FILE") recordings from $SHARD_FILE"

while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    echo "=== $rel ==="
    python scripts/run.py squeakout outputs/squeakout --dataset dryad_gerbil_full \
        --recording "$rel" --channels 0 --prefix mic \
        --min-centroid-hz 0 \
        || echo "FAILED $rel"
done < "$SHARD_FILE"

echo "Shard $SLURM_ARRAY_TASK_ID done."
