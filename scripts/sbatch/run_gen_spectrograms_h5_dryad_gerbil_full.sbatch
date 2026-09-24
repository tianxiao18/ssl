#!/bin/bash
#SBATCH --job-name=specs-h5-dryad-full
#SBATCH --partition=gen          # CPU partition; STFT+HDF5-write has no GPU step
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=4:00:00
#SBATCH --output=/tmp/%x-%j.out

# Full run across dryad_gerbil_full (cohort2_formatted: 1206 recordings x ~20min,
# cohort4_formatted: 675 recordings x ~60min -- ~1881 recordings total), writing
# ONE continuous-spectrogram HDF5 file per recording via scripts/gen_spectrograms_h5.py
# instead of scripts/gen_spectrograms.py's one-PNG-per-second (which would be
# ~3.88M files / ~487GB -- an inode/small-file-IO problem regardless of which
# shared filesystem it lands on). See vox_tracer.spec.write_recording_spectrogram_h5.
#
# Sizing, measured directly (not extrapolated) on one recording from each cohort
# in an interactive session, after switching the STFT pipeline to float32 +
# in-place ops (vox_tracer/spec.py) to cut peak memory roughly in half:
#   cohort2 sample (1200s):  15.2s compute,  57.5 MB output, ~7GB peak RSS
#   cohort4 sample (3600s):  31.9s compute,  86.3 MB output, ~12.4GB peak RSS
# Extrapolated: ~11.1h serial total, ~128GB total output across 1881 files.
# --workers 16 x ~13GB/worker headroom -> --mem=220G; wall time optimistically
# ~40min but likely more once 16 workers contend for reads against the same
# source filesystem -- watch the first hour of the log and adjust --time if
# this is running much slower than that.
#
# --lo/--hi inherited from the dryad_gerbil pilot's calibration (-95/-57); NOT
# re-derived for this cohort -- see the calibration discussion in this session
# (some recurring broadband non-vocal noise saturates under this window, but
# real calls looked comparable in quality to dryad_gerbil; a downstream
# flatness/centroid gate is expected to reject the saturated noise, same as it
# does for dryad_gerbil).
#
# Output lives on ceph (not home -- ~128GB is bulk data), symlinked into the
# repo at outputs/spectrograms_h5/dryad_gerbil_full ->
# /mnt/ceph/users/the10/dryad_gerbil_full_specs_h5
#
# Submit:  sbatch scripts/sbatch/run_gen_spectrograms_h5_dryad_gerbil_full.sbatch

LOGDIR="/mnt/home/the10/ssl/logs"
mkdir -p "$LOGDIR"
exec > "${LOGDIR}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>&1

cd /mnt/home/the10/ssl
source venv/bin/activate

python scripts/gen_spectrograms_h5.py --dataset dryad_gerbil_full --all --workers 16 \
    --channels 0 --prefix mic --lo -95 --hi -57
