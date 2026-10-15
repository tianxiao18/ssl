#!/bin/bash
#SBATCH --job-name=dryad-full-eval
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/%x-%j.out

# dryad_gerbil_full subset (outputs/dryad_full_eval): sam3-box = the existing box-prompt run (outputs/sam3_h5)
# with the pilot's best post-hoc filter (sam3_single_candidate_best: flatness <= 0.21, no column/sweep gates),
# then scripts/evaluate.py for all four models vs the paper and segments GT
# -> outputs/eval/dryad_full_{paper,segments}/<model>/metrics.csv + comparison.png
# CPU only. Submit after sam3_text_dryad.sbatch finishes: sbatch --dependency=afterok:<array ids> ...
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=4
E=outputs/dryad_full_eval
BOX=$E/sam3_box_flat

if [ ! -d "$BOX" ]; then
    # apply_mask_filter.py's --flat-layout expects <base>/<recording>/coco_ch_*.json: stage the subset per cohort
    while IFS= read -r rel; do
        mkdir -p "$E/sam3_box_raw/$(dirname "$rel")"
        ln -sfn "$PWD/outputs/sam3_h5/dryad_gerbil_full/$rel" "$E/sam3_box_raw/$rel"
    done < "$E/recordings.txt"
    for co in $(cut -d/ -f1 "$E/recordings.txt" | sort -u); do
        python scripts/segmentation/apply_mask_filter.py "$E/sam3_box_raw/$co" "$BOX/$co" \
            --channels 0 --flat-layout --recording-dir-base "data/dryad_gerbil_full/$co" --prefix mic \
            --min-mask-cols 0 --min-freq-sweep-frac 0 --max-flatness 0.21
    done
fi

# evaluate.py: spec_dir = the subset's GT tree (only used to discover the 100 recordings), recording_dir = GT
#   paper     $E/gt (vocalization_df.csv)      segments  data/dryad_gerbil_full (release cohort*_segments)
# Predictions were thresholded at inference (sam3-text 0.2, fine-tuned 0.5; sam3-box its own 0.5 + flatness).
declare -A PRED=([sam3-box]=$BOX [sam3-text]=outputs/sam3_text_dryad/dryad_gerbil_full
                 [sam3-text-boxft]=outputs/sam3_text_boxft_dryad/dryad_gerbil_full
                 [sam3-text-maskft]=outputs/sam3_text_maskft_dryad/dryad_gerbil_full)
declare -A GT=([paper]=$E/gt [segments]=data/dryad_gerbil_full)
for g in paper segments; do
    for m in sam3-box sam3-text sam3-text-boxft sam3-text-maskft; do
        echo "=== $g / $m"
        python scripts/evaluate.py "$E/gt" outputs/das_yolo/dryad_gerbil_full "${GT[$g]}" "${PRED[$m]}" \
            "outputs/eval/dryad_full_$g/$m" --channels 0 --all --workers 4
    done
    python scripts/compare_methods.py "outputs/eval/dryad_full_$g" \
        --methods sam3-box,sam3-text,sam3-text-boxft,sam3-text-maskft --out "outputs/eval/dryad_full_$g/comparison.png"
done
