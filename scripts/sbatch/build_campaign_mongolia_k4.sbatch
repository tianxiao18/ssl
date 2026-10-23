#!/bin/bash
#SBATCH --job-name=campaign-mongolia-k4
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/%x-%j.out

# mongolia_audiomoth_k4: mongolia_audiomoth_k6 without ridge and squeakout, on the same 36 recordings (--clips pins the
# sample; the detector trees now also hold all-colony outputs). Settings as k6: alpha 0.05 per rule, J 12,
# seed 20260911, n_max 10000. Skipped if campaign.json exists. Submit from the repo root:
#   sbatch scripts/sbatch/build_campaign_mongolia_k4.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DS=mongolia_audiomoth
OUT=outputs/label_campaigns/${DS}_k4
DETS=sam3_best,sam3_traces_t037,sam3_ft_traces_gold_v2,sam3_ft_traces_dasyolo_labeled_v2

if [ -e "$OUT/campaign.json" ]; then echo "$OUT exists, skipped"; exit 0; fi
mkdir -p "$OUT"
python scripts/build_candidate_pool.py --dataset "$DS" --detectors "$DETS" \
    --clips outputs/mongolia_audiomoth/campaign_sample.txt --out "$OUT/candidates.csv" | tee "$OUT/build_pool.log"
python scripts/make_campaign.py --pool "$OUT/candidates.csv" --detectors "$DETS" --out "$OUT" --dataset "$DS" \
    --n-max 10000 --recall-stream-max 10000 --checkpoints 12 --alpha 0.05 --alpha-per-rule --seed 20260911
echo "Done."
