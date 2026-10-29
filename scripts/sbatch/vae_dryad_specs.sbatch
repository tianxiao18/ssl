#!/bin/bash
#SBATCH --job-name=vae-dryad-specs
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/tmp/%x-%A_%a.out

# Step 1 of the DAS vs SAM3 label-source VAE comparison (CPU): AVA spectrograms for each arm's segments on the
# 100-recording sub100_k7 subset, 50k segments per arm, identical paper preprocessing.
#   task 0 = das, task 1 = sam3 (sam3 gold ft)   -> /mnt/ceph/users/the10/ava/dryad_das_vs_sam3/<arm>
# Then step 2: sbatch --dependency=afterok:<this job> scripts/sbatch/vae_dryad_train.sbatch
set -euo pipefail
ARMS=(das sam3)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${ARM}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python scripts/vae/labels_to_ava_specs.py --source "$ARM" --out "/mnt/ceph/users/the10/ava/dryad_das_vs_sam3/$ARM"
echo "Done."
