#!/bin/bash
#SBATCH --job-name=ridge-raw-dryad-full
#SBATCH --partition=genx
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/tmp/%x-%A_%a.out

# Ridge on dryad_gerbil_full WITHOUT the flatness gate, with the same emit-time gates the sam3_h5 and squeakout
# runs used (run.py defaults: min_mask_cols 9, max area 0.15; centroid off) -> outputs/ridge_raw.
# Same 16 balanced shards as run_sam3_dryad_full.sbatch. Submit: sbatch scripts/sbatch/ridge_raw_dryad_full.sbatch
LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

OUT_ROOT=outputs/ridge_raw
# SHARD_FMT picks the recording lists, e.g. the 100-recording subset (4 shards):
#   sbatch --array=0-3 --export=ALL,SHARD_FMT=outputs/dryad_full_eval/shard_%d.txt scripts/sbatch/ridge_raw_dryad_full.sbatch
SHARD_FILE=$(printf "${SHARD_FMT:-outputs/sam3_shards/shard_%02d.txt}" "$SLURM_ARRAY_TASK_ID")
echo "Shard $SLURM_ARRAY_TASK_ID: $(wc -l < "$SHARD_FILE") recordings from $SHARD_FILE"

export OUT_ROOT
grep -v '^$' "$SHARD_FILE" | xargs -P "$SLURM_CPUS_PER_TASK" -I{} bash -c '
    rel="$1"
    if [ -f "$OUT_ROOT/dryad_gerbil_full/$rel/coco_ch_0.json" ]; then
        echo "skip $rel"; exit 0
    fi
    OMP_NUM_THREADS=1 python scripts/run.py ridge "$OUT_ROOT" --dataset dryad_gerbil_full \
        --recording "$rel" --channels 0 --prefix mic \
        --freq-min 500 --min-centroid-hz 0 \
        || echo "FAILED $rel"
' _ {}

echo "Shard $SLURM_ARRAY_TASK_ID done."
