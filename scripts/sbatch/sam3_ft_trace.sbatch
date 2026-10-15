#!/bin/bash
#SBATCH --job-name=sam3-ft-trace
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --dependency=afterok:7234244
#SBATCH --output=/tmp/%x-%j.out

# Fine-tune SAM3 with the text prompt "trace" on the corrected-GT pseudo boxes, vision trunk and
# text encoder frozen, using SAM3's own trainer via scripts/segmentation/train_sam3_ft.py (configs in configs/sam3_ft;
# --use-cluster 0: runs inside this allocation, never submits jobs itself). Two variants, then val-set inference:
#   box  configs/sam3_ft/gerbil_trace_ft.yaml      box loss only; masks = its boxes prompted into
#        SAM3's box->mask predictor            -> outputs/sam3_ft_box2mask/gerbil_ssl
#   seg  configs/sam3_ft/gerbil_trace_seg_ft.yaml  box + mask loss; masks from its own mask head
#                                              -> outputs/sam3_ft_head/gerbil_ssl
# Submit from the repo root: sbatch scripts/sbatch/sam3_ft_trace.sbatch
# Waits for 7234244 (tail windows + pseudo-box rebuild); drop the --dependency line if it's done.
set -euo pipefail

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"  # train.py adds it to sys.path, for vox_tracer.sam3_ft
DATA=outputs/sam3_ft/gerbil_ssl
RUNS=outputs/sam3_ft/runs

python scripts/dataset_prep/build_sam3_ft_dataset.py --out "$DATA" --prompt trace --min-labeled 0.5

python scripts/segmentation/train_sam3_ft.py -c gerbil_trace_ft --use-cluster 0 --num-gpus 1
python scripts/segmentation/run_sam3_ft.py --ckpt "$RUNS/trace_frozen_v1/checkpoints/best.pt" \
    --mask-mode box --split-json "$DATA/val/_annotations.coco.json" --out outputs/sam3_ft_box2mask

python scripts/segmentation/train_sam3_ft.py -c gerbil_trace_seg_ft --use-cluster 0 --num-gpus 1
python scripts/segmentation/run_sam3_ft.py --ckpt "$RUNS/trace_seg_frozen_v1/checkpoints/best.pt" \
    --mask-mode head --split-json "$DATA/val/_annotations.coco.json" --out outputs/sam3_ft_head
