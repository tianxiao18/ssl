#!/bin/bash
#SBATCH --job-name=mongolia-full-filter
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/%x-%j.out

# Gold-only candidate filter on every Mongolia AudioMoth recording: 15 kHz high-pass of the E8 gold ft masks,
# gold-only candidate pool (time-overlap merge), then candidate_filter.py --apply with the filter trained on the
# k4_hp15k labels transferred to gold-only candidates (outputs/eval/candidate_filter/gold_only_k4_hp15k).
# Run after mongolia_all_detectors.sbatch with DIR=outputs/mongolia_audiomoth/full. Submit from the repo root:
#   sbatch --dependency=afterok:<detectors> scripts/sbatch/mongolia_full_filter.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
DS=mongolia_audiomoth
CLIPS=outputs/mongolia_audiomoth/full/sample.txt
DET=sam3_ft_traces_gold_v2_hp15k
OUT=outputs/eval/candidate_filter/full_gold_hp15k
TRAIN=outputs/eval/candidate_filter/gold_only_k4_hp15k

python scripts/dataset_prep/highpass_preds.py --dataset "$DS" --detectors sam3_ft_traces_gold_v2 --clips "$CLIPS" --cut-khz 15
mkdir -p "$OUT"
python scripts/build_candidate_pool.py --dataset "$DS" --detectors "$DET" --clips "$CLIPS" --out "$OUT/candidates.csv" \
    | tee "$OUT/build_pool.log"
cat > "$OUT/campaign.json" <<EOF
{"name": "full_gold_hp15k", "dataset": "$DS", "detectors": ["$DET"], "pool_csv": "$OUT/candidates.csv"}
EOF
python scripts/evaluation/candidate_filter.py --campaign "$TRAIN" --apply "$OUT"
echo "Done."
