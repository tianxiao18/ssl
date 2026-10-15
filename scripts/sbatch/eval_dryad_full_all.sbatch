#!/bin/bash
#SBATCH --job-name=eval-dryad-full-all
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/tmp/%x-%j.out

# scripts/evaluate.py on all of dryad_gerbil_full for every method (no flatness filter anywhere),
# GT = the release segments (data/dryad_gerbil_full/<rel>/*_annotations_gt.csv,
# evaluate.py's default). -> outputs/eval/dryad_full_seg/<method>/metrics.csv + comparison.png.
# Methods whose predictions are missing are skipped. Submit after the runs and filters have finished.
set -uo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1
# SUBSET=1: only the 100-recording subset (outputs/dryad_full_eval; evaluate.py discovers recordings from
# spec_dir, so point it at the subset's GT tree) -> outputs/eval/dryad_subset_seg
#   sbatch --export=ALL,SUBSET=1 scripts/sbatch/eval_dryad_full_all.sbatch
if [ "${SUBSET:-0}" = 1 ]; then
    SPEC=outputs/dryad_full_eval/gt OUT=outputs/eval/dryad_subset_seg
else
    SPEC=outputs/spectrograms_h5/dryad_gerbil_full OUT=outputs/eval/dryad_full_seg
fi
declare -A PRED=([sam3-box]=sam3_h5 [squeakout]=squeakout [ridge]=ridge_raw [sam3-text]=sam3_text_dryad
                 [sam3-text-boxft]=sam3_text_boxft_dryad [sam3-text-maskft]=sam3_text_maskft_dryad)
ORDER=(sam3-box squeakout ridge sam3-text sam3-text-boxft sam3-text-maskft)
DONE=()
for m in "${ORDER[@]}"; do
    P=outputs/${PRED[$m]}/dryad_gerbil_full
    n=$(find "$P" -maxdepth 3 -name coco_ch_0.json 2>/dev/null | wc -l)
    echo "=== $m: $n recordings with predictions"
    [ "$n" -eq 0 ] && continue
    python scripts/evaluate.py "$SPEC" outputs/das_yolo/dryad_gerbil_full \
        data/dryad_gerbil_full "$P" "$OUT/$m" --channels 0 --all --workers "$SLURM_CPUS_PER_TASK" | grep -E "Overall|ERROR"
    DONE+=("$m")
done
python scripts/compare_methods.py "$OUT" --methods "$(IFS=,; echo "${DONE[*]}")" --out "$OUT/comparison.png"
