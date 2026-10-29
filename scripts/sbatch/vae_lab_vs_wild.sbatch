#!/bin/bash
#SBATCH --job-name=vae-lab-vs-wild
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/%x-%j.out

# Lab (dryad, paper latents + z_70) vs wild (Mongolia) calls in the paper's pretrained VAE (checkpoint_050.tar).
# CPU only: ~56k wild specs + 840 lab exemplar specs, encode, UMAP/kNN/GMM.
# Per-frequency noise floor (option E, window start calibrated on lab), filtered split at 25 kHz -> outputs/ava/lab_vs_wild_floor
set -euo pipefail
LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
for stage in specs encode compare; do
    python scripts/vae/lab_vs_wild_vae.py "$stage" --out outputs/ava/lab_vs_wild_floor
done
echo "Done."
