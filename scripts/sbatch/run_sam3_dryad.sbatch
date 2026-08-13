#!/bin/bash
#SBATCH --job-name=sam3-dryad
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out

# Full SAM3 run across all 5 dryad_gerbil recordings (~8400 chunks total). Run
# scripts/run_sam3_dryad_smoketest.sbatch first to confirm the GPU path works
# before spending a full allocation on this.
#
# Submit:  sbatch scripts/run_sam3_dryad.sbatch

SPEC_ROOT="outputs/spectrograms/dryad_gerbil"
OUT_ROOT="outputs/sam3/dryad_gerbil"
DATA_ROOT="data/dryad_gerbil"
CHECKPOINT="sam3/sam3.pt"        # leave empty to download from HuggingFace

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# Gerbil vocalizations span ~500 Hz-62.5 kHz -- override the mouse/gerbil_ssl-tuned
# ultrasonic-only defaults (freq_min=20000, min_centroid_hz=25000), which would
# otherwise filter out real dryad calls.
for rec_dir in "$SPEC_ROOT"/*/; do
    stem="$(basename "$rec_dir")"
    out="$OUT_ROOT/$stem/coco_ch_0.json"
    if [ -f "$out" ]; then
        echo "skip $stem (already done)"
        continue
    fi
    python scripts/run.py sam3 \
        "$rec_dir" \
        "$OUT_ROOT/$stem" \
        --channels 0 --prefix mic \
        --recording-dir "$DATA_ROOT/$stem" \
        --freq-min 500 --min-centroid-hz 0 \
        ${CHECKPOINT:+--sam3-checkpoint "$CHECKPOINT"}
done
