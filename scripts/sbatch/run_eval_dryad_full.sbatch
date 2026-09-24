#!/bin/bash
#SBATCH --job-name=eval-dryad-full
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/tmp/%x-%j.out

# Event-level scoring of the SAM3 dryad_gerbil_full predictions against the
# per-recording mic_0_*_annotations_gt.csv files.  CPU-only: the cost is JSON
# parsing (~7.5 MB per recording x 1881) plus 1-D interval matching, no GPU.
#
# Why this is a batch job and not a notebook cell: the jupyterhub session this
# was first run from is a 1-CPU / 16 GB allocation, which loses on both counts.
#   - 1 CPU: --workers N gives no parallelism, so the run takes ~4.7h serialized.
#     Measured per recording: 10.1s (cohort2, 1199 windows) / 7.2s (cohort4).
#     At 8 cpus that is ~40 min.
#   - 16 GB: evaluate.py's all_examples pool retains ~12 MB of SAM3 polygon
#     segmentation per recording and never frees it, so the parent crosses 16 GB
#     around recording ~1300-1700 of 1881 and gets OOM-killed -- after which
#     nothing is written at all, since the CSV lands only once the pool drains.
#     evaluate.py now guards that pool behind COLLECT_EXAMPLES (built only when
#     --montage-samples > 0), so the real peak here is a few hundred MB.
#     --mem=48G is kept deliberately as headroom: it still covers the ~23 GB
#     unguarded peak, so this script also runs correctly against an older
#     checkout of evaluate.py or a future --montage-samples run.
#
# --montage-samples is deliberately not passed: dryad_gerbil_full is HDF5-backed
# (outputs/spectrograms_h5/...), and _render_column reads PNGs via cv2.imread,
# so montage pages cannot be rendered for this dataset in any case.
#
# GT source: no --gt-csv, so evaluate.py reads each recording's own
# *_annotations_gt.csv.  Note that with the current working-tree diff a recording
# whose GT file is missing is skipped silently -- compare the metrics.csv row
# count against the coverage number printed below.
#
# Submit:  sbatch scripts/sbatch/run_eval_dryad_full.sbatch
# Result:  outputs/eval/sam3_dryad_full/metrics.csv (one row per recording)

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

PRED=outputs/sam3_h5/dryad_gerbil_full
OUT=outputs/eval/sam3_dryad_full

# Record what fraction of the corpus actually has predictions, so the metrics
# are interpretable later (shard 15 of job 7000134 lost its GPU and produced
# none of its 118 recordings; a shard-15 rerun brings this to 1881).
n_pred=$(find "$PRED" -maxdepth 3 -name 'coco_ch_0.json' | wc -l)
echo "predictions present: ${n_pred} / 1881 recordings"

python scripts/evaluate.py \
    outputs/spectrograms_h5/dryad_gerbil_full \
    outputs/das_yolo/dryad_gerbil_full \
    data/dryad_gerbil_full \
    "$PRED" \
    "$OUT" \
    --channels 0 --all --workers "${SLURM_CPUS_PER_TASK}"

echo "eval done. rows: $(( $(wc -l < "$OUT/metrics.csv") - 1 ))"
