#!/bin/bash
#SBATCH --job-name=sam3-ft
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/tmp/%x-%j.out

# Fine-tuned SAM3 via run.py; all arguments are passed through.
#   sbatch scripts/sbatch/sam3_ft_run.sbatch sam3_ft_seg --dataset mongolia_wild_data --prefix mic
# As an array, task i only runs the recordings in $SHARDS formatted with i (default outputs/sam3_shards/shard_XX.txt):
#   sbatch --array=0-15 scripts/sbatch/sam3_ft_run.sbatch sam3_ft_seg outputs/sam3_text_maskft_dryad \
#       --dataset dryad_gerbil_full --prefix mic
# Resumable: recordings that already have output are skipped.
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ARGS=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS+=(--recordings-file "$(printf "${SHARDS:-outputs/sam3_shards/shard_%02d.txt}" "$SLURM_ARRAY_TASK_ID")")
fi
echo "run.py ${ARGS[*]}"
python scripts/run.py "${ARGS[@]}"
