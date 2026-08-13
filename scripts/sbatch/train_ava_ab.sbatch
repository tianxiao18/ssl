#!/bin/bash
#SBATCH --job-name=ava-ab
#SBATCH --partition=eval
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/%x-%j.out
#
# Step 2 of the SAM3-mask denoising A/B: train one arm's AVA VAE (raw or
# masked). GPU job. Submit both arms as separate jobs so they train in
# parallel on two GPUs (~2x wall-clock speedup vs. training them sequentially
# in one job), then run the CPU-only compare step once both finish.
#
# Prereq: run step 1 (CPU, no GPU) first to build the paired HDF5 datasets:
#     python scripts/vae/coco_to_ava_specs.py \
#         --coco 'outputs/sam3_best/experiment_*/idx_*/coco_ch_*.json' \
#         --out  "$DATA"
#
# Submit (one job per arm, run in parallel):
#     sbatch --job-name=ava-ab-raw    --export=ALL,ARM=raw    scripts/train_ava_ab.sbatch
#     sbatch --job-name=ava-ab-masked --export=ALL,ARM=masked scripts/train_ava_ab.sbatch
#
# Then, once BOTH jobs have finished (check with: squeue -u $USER),
# combine them into the UMAP figure + metrics.json. This is CPU-only and
# takes under a minute, so just run it directly (no sbatch needed):
#     source venv/bin/activate
#     python scripts/vae/train_ava_ab.py --data "$DATA" --out "$OUT" --arm compare
#
# Outputs: $OUT/{latent_raw.npy,metrics_raw.json,
#                latent_masked.npy,metrics_masked.json,
#                latent_umap_ab.png,metrics.json}   (last two from --arm compare)
# ($OUT defaults to outputs/compare on home -- small artifacts (checkpoints,
#  latents, figures); $DATA/{raw,masked} stay on ceph -- the ~20GB spec dataset.)

# --- config -----------------------------------------------------------------
: "${ARM:?set ARM=raw or ARM=masked, e.g. sbatch --export=ALL,ARM=raw train_ava_ab.sbatch}"
case "$ARM" in
    raw|masked) ;;
    *) echo "ERROR: ARM must be 'raw' or 'masked', got '$ARM'" >&2; exit 1 ;;
esac
DATA="/mnt/ceph/users/the10/ava/all_recordings"
OUT="/mnt/home/the10/ssl/outputs/compare"
EPOCHS=150  # 30 was undertrained (loss still dropping); 150 matches other converged runs, ~2.7h per arm in parallel
Z_DIM=32
BATCH_SIZE=64
SEED=0
# ----------------------------------------------------------------------------

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/ava-ab-${ARM}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# Fail early (before burning a GPU slot) if step 1 hasn't produced the datasets.
if [[ ! -d "$DATA/raw" || ! -d "$DATA/masked" ]]; then
    echo "ERROR: $DATA/{raw,masked} not found. Run step 1 (coco_to_ava_specs.py) first." >&2
    exit 1
fi

echo "=== GPU visible to this job (arm=$ARM) ==="
nvidia-smi -L || echo "(no nvidia-smi)"

mkdir -p "$OUT"
python scripts/vae/train_ava_ab.py \
    --data "$DATA" \
    --out "$OUT" \
    --arm "$ARM" \
    --epochs "$EPOCHS" \
    --z-dim "$Z_DIM" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED"

echo "=== DONE ($ARM) ==="
