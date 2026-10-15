#!/bin/bash
#SBATCH --job-name=sam3-text-dryad
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11
#SBATCH --output=/tmp/%x-%A_%a.out

# Text-prompt ("trace") SAM3 on the 100-recording dryad_gerbil_full eval subset (outputs/dryad_full_eval,
# 4 shards), reading the HDF5 spectrograms. Array index = 4 * model + shard:
#   0-3   sam3-text         zero-shot, score 0.2         -> outputs/sam3_text_dryad
#   4-7   sam3-text-boxft   full box fine-tune, 0.5      -> outputs/sam3_text_boxft_dryad
#   8-11  sam3-text-maskft  full mask fine-tune, 0.5     -> outputs/sam3_text_maskft_dryad
# maskft needs trace_seg_frozen_v1/checkpoints/best.pt (sam3-ft-seg, 7240250). Submit from the repo root:
#   sbatch --array=0-7 scripts/sbatch/sam3_text_dryad.sbatch
#   sbatch --array=8-11 --dependency=afterok:7240250 scripts/sbatch/sam3_text_dryad.sbatch
# Resumable: recordings that already have coco_ch_0.json are skipped.
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
RUNS=outputs/sam3_ft/runs
MODEL=$((SLURM_ARRAY_TASK_ID / 4))
SHARD=outputs/dryad_full_eval/shard_$((SLURM_ARRAY_TASK_ID % 4)).txt
case $MODEL in
    0) ARGS=(--ckpt none --mask-mode head --threshold 0.2 --out outputs/sam3_text_dryad) ;;
    1) ARGS=(--ckpt "$RUNS/trace_frozen_v1/checkpoints/best.pt" --mask-mode box --out outputs/sam3_text_boxft_dryad) ;;
    2) ARGS=(--ckpt "$RUNS/trace_seg_frozen_v1/checkpoints/best.pt" --mask-mode head --out outputs/sam3_text_maskft_dryad) ;;
esac
echo "model $MODEL, $SHARD ($(wc -l < "$SHARD") recordings)"

python scripts/segmentation/run_sam3_ft.py "${ARGS[@]}" --dataset dryad_gerbil_full --channels 0 --prefix mic \
    --recordings-file "$SHARD"
