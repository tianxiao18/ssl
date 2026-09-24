#!/bin/bash
#SBATCH --job-name=sam3-dryad-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --output=/tmp/%x-%A_%a.out

# Full run across dryad_gerbil_full (1881 recordings, ~3.88M one-second windows)
# via a 16-way Slurm job array, each array task processing a pre-balanced shard
# sequentially on its own single GPU (scripts/experimental/make_sam3_shards.py
# split the corpus by real per-recording window count, LPT bin-packing, <1%
# imbalance across shards -- outputs/sam3_shards/shard_NN.txt).
#
# Chose a job array over disBatch: disBatch doesn't bind GPUs itself (only
# exposes a per-task DISBATCH_CYLINDER_RANK; the task would have to set
# CUDA_VISIBLE_DEVICES itself), and this workload's tasks (one recording each)
# have predictable, near-uniform cost, so disBatch's dynamic work-stealing
# doesn't buy us anything a static balanced split doesn't already get for free.
# Each array task's own Slurm allocation exposes exactly one GPU -- zero
# device-binding ambiguity. Confirmed via `scontrol show partition gpu`
# (Exclusive=NO) and a real node from this session's own smoke-test jobs
# (workergpu067, shared across multiple jobs/users already) that a 1-GPU
# request does NOT reserve a whole node -- Slurm packs these onto shared nodes.
#
# --gres=gpu:a100-sxm4-80gb:1 (a specific full GPU type, not generic --gpus=1):
# this session's two prior GPU jobs both landed on workergpu067, whose only
# GRES are MIG slices (a100_2g.20gb / a100_1g.20gb) -- generic --gpus=1 is
# satisfiable by a ~2/7-of-a-card MIG fraction, not a whole GPU. The measured
# throughput this sizing is based on (0.311s/window, from time_sam3_throughput.py)
# was on exactly such a MIG slice, so a full A100 should be faster than this,
# not slower -- --time is sized off the conservative (slower) MIG-slice number.
#
# Sizing (conservative, MIG-slice-rate basis): ~242,282 windows/shard *
# 0.311s/window + ~118 recordings/shard * 38.3s one-time model-load overhead
# each ~= 22.2h/shard. --time=30:00:00 leaves real margin; watch the first
# shard's pace in the log and adjust if a full GPU turns out not to be faster.
#
# Submit:  sbatch scripts/sbatch/run_sam3_dryad_full.sbatch
# Monitor: squeue -u $USER  (array shows as sam3-dryad-full_0 .. _15)
# Resume a failed/timed-out shard: just resubmit (run_sam3 skips recordings
# whose coco_ch_0.json already exists) -- e.g.
#   sbatch --array=3,7 scripts/sbatch/run_sam3_dryad_full.sbatch

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

SHARD_FILE=$(printf "outputs/sam3_shards/shard_%02d.txt" "$SLURM_ARRAY_TASK_ID")
echo "Shard $SLURM_ARRAY_TASK_ID: $(wc -l < "$SHARD_FILE") recordings from $SHARD_FILE"

# Gerbil vocalizations span ~500 Hz-62.5 kHz -- override the mouse/gerbil_ssl-tuned
# ultrasonic-only defaults (freq_min=20000, min_centroid_hz=25000), same as every
# other dryad_gerbil* run in this repo.
while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    echo "=== $rel ==="
    python scripts/run.py sam3 outputs/sam3_h5 --dataset dryad_gerbil_full \
        --recording "$rel" --channels 0 --prefix mic \
        --freq-min 500 --min-centroid-hz 0 \
        --sam3-checkpoint sam3/sam3.pt
done < "$SHARD_FILE"

echo "Shard $SLURM_ARRAY_TASK_ID done."
