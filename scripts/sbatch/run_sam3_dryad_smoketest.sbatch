#!/bin/bash
#SBATCH --job-name=sam3-dryad-smoketest
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/tmp/%x-%j.out

# 10-chunk smoke test before the full 5-recording SAM3 run (scripts/run_sam3_dryad.sbatch).
# Confirms the GPU path (model load, ridge-exemplar prompt, mask inference) works on
# real dryad_gerbil spectrograms before committing a full GPU allocation to all ~8400
# chunks. Points at a small persistent subset (outputs/spectrograms/dryad_gerbil_smoketest,
# symlinks into the first 10 chunks of 2020_07_22_..., which visibly has clear calls).
#
# Submit:  sbatch scripts/run_sam3_dryad_smoketest.sbatch

SPEC_DIR="outputs/spectrograms/dryad_gerbil_smoketest/2020_07_22_15_52_33_369348_merged"
OUT_DIR="outputs/sam3/dryad_gerbil_smoketest/2020_07_22_15_52_33_369348_merged"
RECORDING_DIR="data/dryad_gerbil/2020_07_22_15_52_33_369348_merged"
CHECKPOINT="sam3/sam3.pt"        # leave empty to download from HuggingFace

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# Gerbil vocalizations span ~500 Hz-62.5 kHz (see scripts/dryad_prep.py) -- override
# the mouse/gerbil_ssl-tuned ultrasonic-only defaults (freq_min=20000, min_centroid_hz=25000).
python scripts/run.py sam3 \
    "$SPEC_DIR" \
    "$OUT_DIR" \
    --channels 0 --prefix mic \
    --recording-dir "$RECORDING_DIR" \
    --freq-min 500 --min-centroid-hz 0 \
    ${CHECKPOINT:+--sam3-checkpoint "$CHECKPOINT"}

echo "Done. Inspect: $OUT_DIR/coco_ch_0.json"
