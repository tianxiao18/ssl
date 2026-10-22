#!/bin/bash
#SBATCH --job-name=campaign-mongolia
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/%x-%j.out

# Candidate pools + frozen campaigns for Mongolia, after mongolia_detectors.sbatch and mongolia_ridge.sbatch:
#   mongolia_audiomoth_k6   the 36-recording sample (the pool keeps clips every detector covers)
#   mongolia_wild_data_k6   all 12 recordings
# 6 single-detector rules (das / das_yolo cannot run on Mongolia), order and settings as gerbil_ssl_test_k7:
# alpha 0.05 per rule, J 12, seed 20260911, n_max 10000. Zero-shot "traces" is thresholded to 0.37 first.
# Each campaign is skipped if its campaign.json exists. Submit from the repo root, after both jobs:
#   sbatch --dependency=afterok:<detectors job>:<ridge job> scripts/sbatch/build_campaign_mongolia.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DETS=sam3_best,squeakout,ridge,sam3_traces_t037,sam3_ft_traces_gold_v2,sam3_ft_traces_dasyolo_labeled_v2

for pair in mongolia_audiomoth:outputs/sam3_traces_mongolia_am mongolia_wild_data:outputs/sam3_traces_mongolia; do
    DS=${pair%%:*}; ZS=${pair#*:}; OUT=outputs/label_campaigns/${DS}_k6
    if [ -e "$OUT/campaign.json" ]; then echo "$OUT exists, skipped"; continue; fi
    [ -d "outputs/sam3_traces_t037/$DS" ] || \
        python scripts/dataset_prep/threshold_preds.py --src "$ZS" --dst outputs/sam3_traces_t037 --dataset "$DS" --threshold 0.37
    mkdir -p "$OUT"
    python scripts/build_candidate_pool.py --dataset "$DS" --detectors "$DETS" --out "$OUT/candidates.csv" \
        | tee "$OUT/build_pool.log"
    python scripts/make_campaign.py --pool "$OUT/candidates.csv" --detectors "$DETS" --out "$OUT" --dataset "$DS" \
        --n-max 10000 --recall-stream-max 10000 --checkpoints 12 --alpha 0.05 --alpha-per-rule --seed 20260911
done
echo "Done."
