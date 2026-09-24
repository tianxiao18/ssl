#!/bin/bash
#SBATCH --job-name=ridge-dryad-full
#SBATCH --partition=genx
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/tmp/%x-%A_%a.out

# Ridge over dryad_gerbil_full (1881 recordings, ~3.88M windows, HDF5-backed), one
# array task per SAM3 shard (outputs/sam3_shards/shard_NN.txt, balanced by window
# count), 16 recordings in parallel per task.
#
# Config matches run_ridge_dryad.sbatch (flatness-only stage-2: min-mask-cols 0,
# min-centroid-hz 0, max-flatness 0.21, freq-min 500 for the ~500 Hz-62.5 kHz band).
#
# Sizing: 0.085 s/window measured on one CPU (incl. audio load) -> ~242k windows *
# 0.085 / 16 ~= 22 min per task. ~0.6 GB float32 audio per worker.
#
# Submit:  sbatch scripts/sbatch/run_ridge_dryad_full.sbatch
# Resume:  resubmit (recordings with coco_ch_0.json are skipped), e.g. --array=3,7

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

OUT_ROOT=outputs/ridge
SHARD_FILE=$(printf "outputs/sam3_shards/shard_%02d.txt" "$SLURM_ARRAY_TASK_ID")
echo "Shard $SLURM_ARRAY_TASK_ID: $(wc -l < "$SHARD_FILE") recordings from $SHARD_FILE"

export OUT_ROOT
grep -v '^$' "$SHARD_FILE" | xargs -P "$SLURM_CPUS_PER_TASK" -I{} bash -c '
    rel="$1"
    if [ -f "$OUT_ROOT/dryad_gerbil_full/$rel/coco_ch_0.json" ]; then
        echo "skip $rel"; exit 0
    fi
    OMP_NUM_THREADS=1 python scripts/run.py ridge "$OUT_ROOT" --dataset dryad_gerbil_full \
        --recording "$rel" --channels 0 --prefix mic \
        --freq-min 500 --min-mask-cols 0 --min-centroid-hz 0 --max-flatness 0.21 \
        || echo "FAILED $rel"
' _ {}

echo "Shard $SLURM_ARRAY_TASK_ID done."
