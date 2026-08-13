#!/bin/bash
#SBATCH --job-name=sam3-gerbil-family-smoketest
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/tmp/%x-%j.out

# 20-chunk-per-recording smoke test on the new gerbil_family dataset (experiment_497,
# experiment_552), mirroring run_sam3_dryad_smoketest.sbatch. Confirms the GPU path
# works on gerbil_family spectrograms before committing a full GPU allocation to all
# ~718 chunks. Points at outputs/spectrograms/gerbil_family_smoketest, symlinks into
# the first 20 chunks of each recording (both contain annotated calls in that span).
#
# Ridge smoke test already ran on the full recordings (CPU, no GPU needed):
#   experiment_497: Recall=0.716 Precision=0.539 F1=0.615 (GT=289)
#   experiment_552: Recall=0.896 Precision=0.593 F1=0.713 (GT=741)
# Compare this SAM3 run's coco_ch_30.json against the same GT via scripts/evaluate.py
# before committing a full GPU allocation to the 2-recording run.
#
# Submit:  sbatch scripts/run_sam3_gerbil_family_smoketest.sbatch

CHECKPOINT="sam3/sam3.pt"        # leave empty to download from HuggingFace

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# gerbil_family spans alarm (sonic) through newborn/high-freq (ultrasonic) calls --
# override the mouse/gerbil_ssl-tuned ultrasonic-only defaults (freq_min=20000,
# min_centroid_hz=25000), same as dryad_gerbil.
for exp in experiment_497 experiment_552; do
    python scripts/run.py sam3 \
        "outputs/spectrograms/gerbil_family_smoketest/$exp" \
        "outputs/sam3/gerbil_family_smoketest/$exp" \
        --channels 30 --prefix channel \
        --recording-dir "data/gerbil_family/$exp" \
        --freq-min 500 --min-centroid-hz 0 \
        ${CHECKPOINT:+--sam3-checkpoint "$CHECKPOINT"}
done

echo "Done. Inspect: outputs/sam3/gerbil_family_smoketest/{experiment_497,experiment_552}/coco_ch_30.json"
