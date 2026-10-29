#!/bin/bash
#SBATCH --job-name=specs-h5-mongolia-all
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/%x-%A_%a.out

# HDF5 spectrograms for the picks of sample_mongolia_all.py that lack one: task i = $DIR/specs_<i>.txt (~24 h
# audio, ~15 min, as mongolia_am_specs_h5.sbatch). Same --lo/--hi as the first 131 recordings. Resumable.
# Submit with the --array / --export printed by the sampler.
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
LIST="$DIR/specs_${SLURM_ARRAY_TASK_ID}.txt"
[ -f "$LIST" ] || { echo "no $LIST, nothing to do"; exit 0; }
while IFS= read -r rel; do
    python scripts/gen_spectrograms_h5.py --dataset mongolia_audiomoth --recording "$rel" --channels 0 --prefix mic \
        --lo -95 --hi -50 --workers 1
done < "$LIST"
echo "Done."
