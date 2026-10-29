#!/bin/bash
#SBATCH --job-name=vae-dryad-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100|a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=/tmp/%x-%A_%a.out

# Step 2 (GPU): one stock AVA VAE per arm, identical seed / architecture / 100 epochs (train_ava_ab.py), ~4 h each.
#   task 0 = das, task 1 = sam3   -> outputs/ava/dryad_das_vs_sam3/{<arm>/, latent_<arm>.npy, metrics_<arm>.json}
# Then step 3: sbatch --dependency=afterok:<this job> scripts/sbatch/vae_dryad_compare.sbatch
set -euo pipefail
ARMS=(das sam3)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${ARM}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/vae/train_ava_ab.py --data /mnt/ceph/users/the10/ava/dryad_das_vs_sam3 --arms das,sam3 --arm "$ARM" \
    --out outputs/ava/dryad_das_vs_sam3 --epochs 100
echo "Done."
