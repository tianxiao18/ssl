#!/bin/bash
#SBATCH --job-name=ridge-dryad
#SBATCH --partition=gen          # CPU partition; ridge has no GPU step
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/%x-%j.out

# outputs/ridge/dryad_gerbil (generated 2026-08-10) used ridge.py's defaults
# (freq_min=20000, min_centroid_hz=25000) -- tuned for the ultrasonic-only
# mouse/gerbil_ssl data. Every one of its 4185 detections has centroid_hz >=
# ~25 kHz, meaning the whole sub-25kHz half of the dryad_gerbil call range
# (~500 Hz-62.5 kHz, see scripts/dryad_prep.py) was silently discarded by the
# stage-2 gate.
#
# Just disabling min-centroid-hz (as the sam3 dryad runs do) isn't enough --
# it leaves ridge with only the geometric min-mask-cols gate and no spectral
# gate at all. ridge.py's OTHER cross-validated stage-2 config, flatness-only
# (max_flatness<=0.21, tonal-vs-noise-like), doesn't assume anything about
# *where* in the spectrum a call sits, so unlike centroid it's still valid
# for dryad's wider band. Uses that config here (same one run_ridge.sbatch
# uses for gerbil_ssl: min-mask-cols 0, min-centroid-hz 0, max-flatness 0.21).
#
# Submit:  sbatch scripts/run_ridge_dryad.sbatch

SPEC_ROOT="outputs/spectrograms/dryad_gerbil"
OUT_ROOT="outputs/ridge"
WORKERS=16

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

for rec_dir in "$SPEC_ROOT"/*/; do
    stem="$(basename "$rec_dir")"
    out="$OUT_ROOT/dryad_gerbil/$stem/coco_ch_0.json"
    if [ -f "$out" ]; then
        echo "skip $stem (already done)"
        continue
    fi
    python scripts/run.py ridge \
        "$OUT_ROOT" \
        --dataset dryad_gerbil \
        --recording "$stem" \
        --channels 0 --prefix mic \
        --freq-min 500 \
        --min-mask-cols 0 --min-centroid-hz 0 --max-flatness 0.21 \
        --workers "$WORKERS"
done
