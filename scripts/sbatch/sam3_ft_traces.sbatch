#!/bin/bash
#SBATCH --job-name=sam3-ft-traces
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=h100|a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=/tmp/%x-%j.out

# Box-only fine-tune with the text prompt "traces", then inference (threshold 0.5) on the 31 test recordings
# (full length, both mics) -> outputs/sam3_ft_traces_$VARIANT/gerbil_ssl. Configs: configs/sam3_ft/gerbil_traces_*.
#   VARIANT=gold             gold boxes, same windows as trace_frozen_v1 (~75 min)
#   VARIANT=dasyolo_labeled  das_yolo boxes on the gold run's windows; best.pt picked on das_yolo val (~75 min)
#   VARIANT=dasyolo_all      das_yolo boxes on every window of the 538 non-val recordings, 5 epochs (~8 h)
# GPU constraint: the venv torch is cu126, which has no kernels for the rtx_pro_6000_blackwell nodes.
# Submit from the repo root, e.g.: sbatch --export=ALL,VARIANT=gold --time=03:00:00 scripts/sbatch/sam3_ft_traces.sbatch
set -euo pipefail
VARIANT=${VARIANT:?set VARIANT=gold, dasyolo_labeled or dasyolo_all}

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${VARIANT}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"  # train.py adds it to sys.path, for vox_tracer.sam3_ft
TEST=outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json  # its recordings = the 31 test recordings

python scripts/segmentation/train_sam3_ft.py -c "gerbil_traces_${VARIANT}_ft" --use-cluster 0 --num-gpus 1
python scripts/segmentation/run_sam3_ft.py --ckpt "outputs/sam3_ft/runs/traces_${VARIANT}_v1/checkpoints/best.pt" \
    --mask-mode box --prompt traces --threshold 0.5 --split-json "$TEST" --out "outputs/sam3_ft_traces_$VARIANT"
