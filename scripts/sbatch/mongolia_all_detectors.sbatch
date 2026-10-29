#!/bin/bash
#SBATCH --job-name=mongolia-all-detectors
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100|a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/%x-%A_%a.out

# GPU detectors on the sample_mongolia_all.py picks that still lack them: task i = line i of $DIR/gpu_tasks.tsv
# (detector, chunk of ~3 GPU h). Settings identical to mongolia_detectors.sbatch (sam3_best, E8 gold / E7 das_yolo
# ft @0.5, squeakout) and mongolia_am_sam3.sbatch (zero-shot "traces" >= 0.2). The ft and zero-shot runs skip
# recordings already done; sam3_best / squeakout lists hold only missing ones. Submit with the sampler's command.
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
RUNS=outputs/sam3_ft/runs
IFS=$'\t' read -r DET LIST < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$DIR/gpu_tasks.tsv")
echo "$DET $LIST ($(wc -l < "$LIST") recordings)"
ft() {
    python scripts/segmentation/run_sam3_ft.py --ckpt "$RUNS/traces_$1_v2/checkpoints/best.pt" --mask-mode box \
        --prompt traces --threshold 0.5 --dataset mongolia_audiomoth --channels 0 --prefix mic \
        --out "outputs/sam3_ft_traces_$1_v2" --recordings-file "$LIST"
}
case "$DET" in
    sam3_best)
        python scripts/run.py sam3 outputs/sam3_best --dataset mongolia_audiomoth --all --recordings-file "$LIST" \
            --channels 0 --prefix mic --workers 1 --sam3-checkpoint sam3/sam3.pt --sigmas 2,3,4 --threshold-pct 99.39 \
            --score-threshold 0.48 --min-area 30 --vert-aspect 3.47 --horiz-aspect 0.2 --close-kernel 7,3 \
            --max-mask-area-frac 0.15 --min-freq-sweep-frac 0.031 --min-mask-cols 1 --sample-rate 192000 \
            --freq-min 15000 --min-centroid-hz 0 ;;
    zs_traces)
        python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.2 --prompt traces \
            --dataset mongolia_audiomoth --channels 0 --prefix mic --out outputs/sam3_traces_mongolia_am \
            --recordings-file "$LIST" ;;
    gold_ft) ft gold ;;
    dasyolo_ft) ft dasyolo_labeled ;;
    squeakout)
        python scripts/run.py squeakout outputs/squeakout --dataset mongolia_audiomoth --all --workers 1 \
            --channels 0 --prefix mic --min-centroid-hz 0 --recordings-file "$LIST" ;;
    *) echo "unknown detector $DET"; exit 1 ;;
esac
echo "Done."
