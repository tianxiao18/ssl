#!/bin/bash
#SBATCH --job-name=campaign-mongolia-k4-hp
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/%x-%j.out

# mongolia_audiomoth_k4_hp<CUT>k: the 4 SAM3 detectors (no ridge, no squeakout) with every mask high-passed at
# CUT kHz (default 15) (pixels below dropped; masks left empty removed), on the same 36 recordings as k6. Settings as k6:
# alpha 0.05 per rule, J 12, seed 20260911, n_max 10000. Skipped if campaign.json exists. Submit from the repo root:
#   sbatch --export=ALL,CUT=10 scripts/sbatch/build_campaign_mongolia_k4_hp.sbatch
#   sbatch --export=ALL,CUT=24 scripts/sbatch/build_campaign_mongolia_k4_hp.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DS=mongolia_audiomoth
CLIPS=outputs/mongolia_audiomoth/campaign_sample.txt
CUT=${CUT:-15}
OUT=outputs/label_campaigns/${DS}_k4_hp${CUT}k
SRC=sam3_best,sam3_traces_t037,sam3_ft_traces_gold_v2,sam3_ft_traces_dasyolo_labeled_v2
DETS=$(echo "$SRC" | sed "s/,/_hp${CUT}k,/g; s/\$/_hp${CUT}k/")

if [ -e "$OUT/campaign.json" ]; then echo "$OUT exists, skipped"; exit 0; fi
[ -d "outputs/sam3_best_hp${CUT}k/$DS" ] || \
    python scripts/dataset_prep/highpass_preds.py --dataset "$DS" --detectors "$SRC" --clips "$CLIPS" --cut-khz "$CUT"
mkdir -p "$OUT"
python scripts/build_candidate_pool.py --dataset "$DS" --detectors "$DETS" --clips "$CLIPS" \
    --out "$OUT/candidates.csv" | tee "$OUT/build_pool.log"
python scripts/make_campaign.py --pool "$OUT/candidates.csv" --detectors "$DETS" --out "$OUT" --dataset "$DS" \
    --n-max 10000 --recall-stream-max 10000 --checkpoints 12 --alpha 0.05 --alpha-per-rule --seed 20260911
echo "Done."
