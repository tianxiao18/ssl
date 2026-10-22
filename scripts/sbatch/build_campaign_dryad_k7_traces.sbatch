#!/bin/bash
#SBATCH --job-name=campaign-dryad-k7
#SBATCH --partition=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/%x-%j.out

# Candidate pool + frozen campaign dryad_gerbil_full_k7_traces: the 7 single-detector rules on all 1881 dryad
# recordings, detector order mirroring dryad_gerbil_full_sub100_k7. Same merge (gap 0), alpha 0.05, J 12,
# seed 20260911 and n_max 10000 as sub100_k7, plus --alpha-per-rule so the 7 bars hold jointly.
# The k4 full-dryad pool was 8.6M candidates, so this runs here rather than on the 8 GiB workstation session.
# Refuses to overwrite an existing campaign. Submit from the repo root: sbatch scripts/sbatch/build_campaign_dryad_k7_traces.sbatch
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OUT=outputs/label_campaigns/dryad_gerbil_full_k7_traces
DETS=sam3_h5,squeakout,ridge_raw,sam3_traces_dryad_full,sam3_ft_traces_gold_v2_dryad,sam3_ft_traces_dasyolo_labeled_v2_dryad,das
[ -e "$OUT/campaign.json" ] && { echo "$OUT/campaign.json exists; not touching it"; exit 1; }
mkdir -p "$OUT"

python scripts/build_candidate_pool.py --dataset dryad_gerbil_full --detectors "$DETS" --out "$OUT/candidates.csv" \
    | tee "$OUT/build_pool.log"
python scripts/make_campaign.py --pool "$OUT/candidates.csv" --detectors "$DETS" --out "$OUT" --dataset dryad_gerbil_full \
    --n-max 10000 --recall-stream-max 10000 --checkpoints 12 --alpha 0.05 --alpha-per-rule --seed 20260911
echo "Done."
