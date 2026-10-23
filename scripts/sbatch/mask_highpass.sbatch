#!/bin/bash
#SBATCH --job-name=mask-highpass
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/%x-%j.out

# Effect of a CUT kHz high-pass (default 15) on the 6 detectors' masks over the mongolia_audiomoth_k6 recordings, plus which
# labeled k6 candidates keep a mask -> outputs/eval/mongolia_highpass[_<CUT>k]/{mask_stats.csv,labeled_candidates.csv,
# mask_highpass.png}. CPU only. Submit from the repo root:
#   sbatch scripts/sbatch/mask_highpass.sbatch
#   sbatch --export=ALL,CUT=24 scripts/sbatch/mask_highpass.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CUT=${CUT:-15}
OUT=outputs/eval/mongolia_highpass; [ "$CUT" = 15 ] || OUT=${OUT}_${CUT}k
python scripts/evaluation/mask_highpass.py --cut-khz "$CUT" --out "$OUT"
echo "Done."
