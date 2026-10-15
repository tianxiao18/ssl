#!/bin/bash
#SBATCH --job-name=sam3-text-dryad-full
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-47
#SBATCH --output=/tmp/%x-%A_%a.out

# Text-prompt ("trace") SAM3 on all of dryad_gerbil_full (HDF5 spectrograms), over the same 16 balanced shards
# as run_sam3_dryad_full.sbatch (outputs/sam3_shards). Array index = 16 * model + shard:
#   0-15   sam3-text         zero-shot, score 0.2      -> outputs/sam3_text_dryad
#   16-31  sam3-text-boxft   full box fine-tune, 0.5   -> outputs/sam3_text_boxft_dryad
#   32-47  sam3-text-maskft  full mask fine-tune, 0.5  -> outputs/sam3_text_maskft_dryad
# ~242k windows/shard at ~0.08 s/window ~= 5.5 h per task. Resumable (recordings with coco_ch_0.json are skipped,
# so the 100 already done are not redone). One model only: sbatch --array=16-31 ...
# Submit from the repo root: sbatch scripts/sbatch/sam3_text_dryad_full.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
RUNS=outputs/sam3_ft/runs
MODEL=$((SLURM_ARRAY_TASK_ID / 16))
SHARD=$(printf "outputs/sam3_shards/shard_%02d.txt" $((SLURM_ARRAY_TASK_ID % 16)))
case $MODEL in
    0) ARGS=(--ckpt none --mask-mode head --threshold 0.2 --out outputs/sam3_text_dryad) ;;
    1) ARGS=(--ckpt "$RUNS/trace_frozen_v1/checkpoints/best.pt" --mask-mode box --out outputs/sam3_text_boxft_dryad) ;;
    2) ARGS=(--ckpt "$RUNS/trace_seg_frozen_v1/checkpoints/best.pt" --mask-mode head --out outputs/sam3_text_maskft_dryad) ;;
esac
echo "model $MODEL, $SHARD ($(grep -c . "$SHARD") recordings)"

python scripts/segmentation/run_sam3_ft.py "${ARGS[@]}" --dataset dryad_gerbil_full --channels 0 --prefix mic \
    --recordings-file "$SHARD"
