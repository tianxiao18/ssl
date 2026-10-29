#!/bin/bash
#SBATCH --job-name=vae-dryad-compare
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/%x-%j.out

# Step 3 (CPU): side-by-side UMAP + cluster metrics (train_ava_ab.py --arm compare), then the human-label comparison
# against the sub100_k7 campaign (compare_label_sources.py: share of real calls, latent AUC real vs not, durations).
# Outputs in outputs/ava/dryad_das_vs_sam3/: latent_umap_ab.png, metrics.json, label_compare.png, label_compare.json
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
DATA=/mnt/ceph/users/the10/ava/dryad_das_vs_sam3
OUT=outputs/ava/dryad_das_vs_sam3
python scripts/vae/train_ava_ab.py --data "$DATA" --arms das,sam3 --arm compare --out "$OUT"
python scripts/vae/compare_label_sources.py --data "$DATA" --compare "$OUT"
echo "Done."
