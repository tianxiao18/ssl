#!/bin/bash
#SBATCH --job-name=sam3-calltype-smoketest
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/tmp/%x-%j.out

# Same 10-chunk smoke-test subset as run_sam3_dryad_smoketest.sbatch, but through
# scripts/run_sam3_calltype.py: every window prompts SAM3 with every ridge candidate
# that matches a known dryad_gerbil call-type cluster (outputs/dryad_holdout), instead
# of only the single loudest ridge candidate. Compare coco_ch_0.json recall/precision
# against outputs/sam3/dryad_gerbil_smoketest/.../coco_ch_0.json (scripts/evaluate.py)
# before committing a full GPU allocation to the 5-recording run.
#
# Submit:  sbatch scripts/run_sam3_calltype_smoketest.sbatch

SPEC_DIR="outputs/spectrograms/dryad_gerbil_smoketest/2020_07_22_15_52_33_369348_merged"
OUT_DIR="outputs/sam3_calltype/dryad_gerbil_smoketest/2020_07_22_15_52_33_369348_merged"
RECORDING_DIR="data/dryad_gerbil/2020_07_22_15_52_33_369348_merged"
SAM3_CHECKPOINT="sam3/sam3.pt"        # leave empty to download from HuggingFace
CALL_TYPE_SPECS="outputs/dryad_holdout"
CALL_TYPE_CHECKPOINT="/mnt/home/the10/ceph/dataset/dryad_gerbil/checkpoint_050.tar"

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# Gerbil vocalizations span ~500 Hz-62.5 kHz (see scripts/dryad_prep.py) -- override
# the mouse/gerbil_ssl-tuned ultrasonic-only defaults (freq_min=20000, min_centroid_hz=25000).
python scripts/segmentation/run_sam3_calltype.py \
    "$SPEC_DIR" \
    "$OUT_DIR" \
    --recording-dir "$RECORDING_DIR" \
    --call-type-specs "$CALL_TYPE_SPECS" \
    --call-type-checkpoint "$CALL_TYPE_CHECKPOINT" \
    --channels 0 --prefix mic \
    --freq-min 500 --min-centroid-hz 0 \
    ${SAM3_CHECKPOINT:+--sam3-checkpoint "$SAM3_CHECKPOINT"}

echo "Done. Inspect: $OUT_DIR/coco_ch_0.json"
