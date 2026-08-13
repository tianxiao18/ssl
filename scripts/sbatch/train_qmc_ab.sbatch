#!/bin/bash
#SBATCH --job-name=qmc-ab
#SBATCH --partition=eval
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/%x-%j.out
#
# Train a QMC-LVM (decoder-only, RQMC-lattice inference, 2D torus latent) on one arm
# (raw or masked) of the AVA spectrogram data, and compare it against that arm's
# already-trained AVA VAE checkpoint (train_ava_ab.py / train_ava_ab.sbatch). GPU job.
# Submit both arms as separate jobs so they train in parallel on two GPUs (~2x
# wall-clock speedup vs. training them sequentially in one job), then merge the
# per-arm results once both finish.
#
# QMC-LVM's per-epoch cost is much higher than the VAE's -- every step forwards the
# decoder over the WHOLE training lattice (610 points at the default --train-grid-m
# 15), not just the data batch -- so epoch count isn't directly comparable to the
# VAE's. At ~3.45 min/epoch/arm (measured on an A100), 60 epochs is ~3.4h/arm; the
# time budget above leaves headroom for a slower GPU.
#
# Prereq: both AVA VAE arms must already be trained (train_ava_ab.sbatch), since this
# script loads $VAE_DIR/{raw,masked}/checkpoint.tar for the comparison.
#
# Submit (one job per arm, run in parallel):
#     sbatch --job-name=qmc-ab-raw    --export=ALL,ARM=raw    scripts/train_qmc_ab.sbatch
#     sbatch --job-name=qmc-ab-masked --export=ALL,ARM=masked scripts/train_qmc_ab.sbatch
#
# Then, once BOTH jobs have finished (check with: squeue -u $USER), merge their
# per-arm metrics.json into one combined file. This is CPU-only and takes under a
# second, so just run it directly (no sbatch needed):
#     source venv/bin/activate
#     python scripts/vae/train_qmc_ab.py --data "$DATA" --out "$OUT" --arm compare
#
# Outputs: $OUT/{raw,masked}/{qmc_checkpoint.tar, qmc_train_curve.png,
#                             qmc_vs_vae_reconstructions.png, metrics.json}
#          $OUT/metrics.json   (combined, from --arm compare)
# ($OUT defaults to outputs/compare_qmc on home -- small artifacts only; $DATA stays
#  on ceph -- the ~20GB spec dataset, same one train_ava_ab.sbatch reads.)

# --- config -----------------------------------------------------------------
: "${ARM:?set ARM=raw or ARM=masked, e.g. sbatch --export=ALL,ARM=raw train_qmc_ab.sbatch}"
case "$ARM" in
    raw|masked) ;;
    *) echo "ERROR: ARM must be 'raw' or 'masked', got '$ARM'" >&2; exit 1 ;;
esac
DATA="/mnt/ceph/users/the10/ava/all_recordings"
VAE_DIR="/mnt/home/the10/ssl/outputs/compare"
OUT="/mnt/home/the10/ssl/outputs/compare_qmc"
EPOCHS=60
BATCH_SIZE=64
LATENT_DIM=2
TRAIN_GRID_M=15
SEED=0
# ----------------------------------------------------------------------------

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/qmc-ab-${ARM}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

# Fail early (before burning a GPU slot) if the data or the VAE checkpoint aren't there.
if [[ ! -d "$DATA/raw" || ! -d "$DATA/masked" ]]; then
    echo "ERROR: $DATA/{raw,masked} not found." >&2
    exit 1
fi
if [[ ! -f "$VAE_DIR/$ARM/checkpoint.tar" ]]; then
    echo "ERROR: $VAE_DIR/$ARM/checkpoint.tar not found. Run train_ava_ab.sbatch for" \
         "this arm first." >&2
    exit 1
fi

echo "=== GPU visible to this job (arm=$ARM) ==="
nvidia-smi -L || echo "(no nvidia-smi)"

mkdir -p "$OUT"
python scripts/vae/train_qmc_ab.py \
    --data "$DATA" \
    --vae-dir "$VAE_DIR" \
    --out "$OUT" \
    --arm "$ARM" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --latent-dim "$LATENT_DIM" \
    --train-grid-m "$TRAIN_GRID_M" \
    --seed "$SEED"

echo "=== DONE ($ARM) ==="
