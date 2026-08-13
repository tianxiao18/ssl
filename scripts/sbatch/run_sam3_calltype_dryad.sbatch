#!/bin/bash
#SBATCH --job-name=sam3-calltype-dryad
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/tmp/%x-%j.out

# Full call-type-gated SAM3 run across all 5 dryad_gerbil recordings (~7200 chunks
# total, from outputs/spectrograms/dryad_gerbil). Same recordings as
# scripts/run_sam3_dryad.sbatch, but through scripts/run_sam3_calltype.py: every
# window prompts SAM3 with every ridge candidate that matches a known dryad_gerbil
# call-type cluster (outputs/dryad_holdout), not just the single loudest one.
#
# On the 10-chunk smoke test (scripts/run_sam3_calltype_smoketest.sbatch) this
# took ~2x as long as the single-candidate run (multiple SAM3 prompts + a VAE
# encode per surviving candidate, per window) while recall went 0.575 -> 0.850
# at ~unchanged precision. --time is set to 2x scripts/run_sam3_dryad.sbatch's
# 24h accordingly; that's an extrapolation from 10 chunks, not a measurement at
# scale, so watch the first recording's pace in the log and adjust if needed.
#
# Submit:  sbatch scripts/run_sam3_calltype_dryad.sbatch

SPEC_ROOT="outputs/spectrograms/dryad_gerbil"
OUT_ROOT="outputs/sam3_calltype/dryad_gerbil"
DATA_ROOT="data/dryad_gerbil"
SAM3_CHECKPOINT="sam3/sam3.pt"        # leave empty to download from HuggingFace
CALL_TYPE_SPECS="outputs/dryad_holdout"
CALL_TYPE_CHECKPOINT="/mnt/home/the10/ceph/dataset/dryad_gerbil/checkpoint_050.tar"

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
    python scripts/segmentation/run_sam3_calltype.py \
        "$rec_dir" \
        "$OUT_ROOT/$stem" \
        --recording-dir "$DATA_ROOT/$stem" \
        --call-type-specs "$CALL_TYPE_SPECS" \
        --call-type-checkpoint "$CALL_TYPE_CHECKPOINT" \
        --channels 0 --prefix mic \
        --freq-min 500 --min-centroid-hz 0 \
        ${SAM3_CHECKPOINT:+--sam3-checkpoint "$SAM3_CHECKPOINT"}
done
