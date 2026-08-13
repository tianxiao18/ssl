"""Regenerate AVA spectrograms for the dryad_gerbil vocalizations whose raw audio
we actually have on disk, paired with the paper's own official latents/labels.

The Dryad download (README.md) only ships 5 sample .wav files out of the full
20-day, 3-cohort recording -- the rest needs the ~15TB Google Drive from the
author. But `vocalization_df` has onset/offset/latent_means/z_70 for every one
of the 583,237 vocalizations in the paper, including whichever ones happen to
fall in those 5 wavs. This script:

  1. filters vocalization_df to rows whose audio_filename is one of the 5
     present wavs,
  2. regenerates each vocalization's spectrogram from the raw audio using the
     *exact* preprocessing params from autoencoded-vocal-analysis/notebooks/
     figure2-vae.ipynb (ava.preprocessing.utils.get_spec, called the same way
     ava.preprocessing.preprocess.get_syll_specs calls it),
  3. writes one AVA-format HDF5 (drop-in for ava.models.vae_dataset, same
     contract as coco_to_ava_specs.py) with the official latent_means/z_70
     carried along as provenance, so a downstream script can compare "my
     pipeline's spectrogram -> paper's checkpoint -> latent" against "paper's
     own latent for that exact vocalization" with a plain paired join (no
     re-parsing the 1GB CSV).

    python scripts/vae/dryad_to_ava_specs.py \
        --data /mnt/home/the10/ceph/dataset/dryad_gerbil \
        --out outputs/dryad_holdout
"""
import argparse
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

from ava.models.vae import X_SHAPE
import ava.preprocessing.utils as ava_pp_utils
from ava.preprocessing.utils import get_spec


class _Interp2DCompat:
    """Drop-in replacement for scipy.interpolate.interp2d (removed in SciPy
    1.14+) covering exactly the call pattern ava.preprocessing.utils.get_spec
    uses: 1-D x/y grids, __call__(x_new, y_new) -> z on the outer product
    grid, out-of-domain queries filled with `fill_value` instead of
    extrapolated (interp2d's bounds_error=False behavior)."""

    def __init__(self, x, y, z, copy=False, bounds_error=False, fill_value=None):
        from scipy.interpolate import RectBivariateSpline
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        self._xlim = (x.min(), x.max())
        self._ylim = (y.min(), y.max())
        self._fill_value = fill_value
        self._spline = RectBivariateSpline(y, x, z)  # z is (len(y), len(x)), matching interp2d's convention

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=False):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        z = self._spline(y, x)
        if self._fill_value is not None:
            out_x = (x < self._xlim[0]) | (x > self._xlim[1])
            out_y = (y < self._ylim[0]) | (y > self._ylim[1])
            z[:, out_x] = self._fill_value
            z[out_y, :] = self._fill_value
        return z


ava_pp_utils.interp2d = _Interp2DCompat

# Exact preprocess_params from figure2-vae.ipynb (autoencoded-vocal-analysis
# notebooks), the params the paper's checkpoint_050.tar was trained under.
PREPROCESS_PARAMS = {
    'max_dur': 0.3,
    'min_freq': 500,
    'max_freq': 62500,
    'num_freq_bins': X_SHAPE[0],
    'num_time_bins': X_SHAPE[1],
    'nperseg': 512,
    'noverlap': 256,
    'spec_min_val': -8,
    'spec_max_val': -5,
    'mel': False,
    'time_stretch': True,
    'within_syll_normalize': False,
}


def parse_vec(s):
    """Parse a numpy.array2string-formatted column value, e.g. '[ 1.2 -3.4\n  5.6]'."""
    return np.fromstring(s.strip().strip('[]'), sep=' ')


def load_vocalization_df(data_dir, wav_names):
    """Rows of vocalization_df.csv whose audio_filename is one of `wav_names`,
    with latent_means/z_70 parsed to arrays. Reads the whole CSV (no pyarrow
    available for the .feather) but only keeps matching rows."""
    usecols = ['audio_filename', 'onset', 'offset', 'cohort', 'z_70', 'latent_means', 'prob_z_70']
    chunks = []
    for chunk in pd.read_csv(data_dir / 'vocalization_df.csv', usecols=usecols,
                             chunksize=100_000):
        sub = chunk[chunk['audio_filename'].isin(wav_names)]
        if len(sub):
            chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    df['latent_means'] = df['latent_means'].apply(parse_vec)
    df['prob_z_70'] = df['prob_z_70'].apply(parse_vec)
    # z_70 = argmax(prob_z_70) by construction (README); this is the paper's own
    # confidence that a vocalization belongs to its assigned cluster, used to
    # pick cluster-representative exemplars downstream.
    df['z70_confidence'] = df.apply(lambda r: r['prob_z_70'][r['z_70']], axis=1)
    return df


def build_specs(data_dir, out_dir):
    wav_paths = sorted(data_dir.glob('*.wav'))
    if not wav_paths:
        raise SystemExit(f"no .wav files found in {data_dir}")
    wav_names = [p.name for p in wav_paths]
    print(f"{len(wav_paths)} wav file(s): {wav_names}")

    df = load_vocalization_df(data_dir, wav_names)
    print(f"{len(df)} vocalizations with audio present (of 583,237 in the full paper)")
    if len(df) == 0:
        raise SystemExit("no matching rows in vocalization_df.csv for the available wavs")

    target_freqs = np.linspace(PREPROCESS_PARAMS['min_freq'], PREPROCESS_PARAMS['max_freq'],
                               PREPROCESS_PARAMS['num_freq_bins'])

    specs, onsets, offsets, audio_filenames, latent_means, z_70, cohort, z70_conf = ([] for _ in range(8))
    n_dropped = 0
    for wav_path in wav_paths:
        sub = df[df['audio_filename'] == wav_path.name]
        if len(sub) == 0:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=WavFileWarning)
            fs, audio = wavfile.read(str(wav_path))
        print(f"  {wav_path.name}: fs={fs}, {len(sub)} vocalizations")
        for row in sub.itertuples(index=False):
            spec, valid = get_spec(row.onset, row.offset, audio, PREPROCESS_PARAMS, fs,
                                   target_freqs=target_freqs)
            if not valid or not np.any(spec):
                n_dropped += 1
                continue
            specs.append(np.clip(spec, 0.0, 1.0).astype(np.float32))
            onsets.append(row.onset)
            offsets.append(row.offset)
            audio_filenames.append(row.audio_filename)
            latent_means.append(row.latent_means.astype(np.float32))
            z_70.append(row.z_70)
            cohort.append(row.cohort)
            z70_conf.append(row.z70_confidence)

    if not specs:
        raise SystemExit("no valid spectrograms produced")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "syllables_0000.hdf5"
    with h5py.File(out_file, "w") as f:
        f.create_dataset("specs", data=np.stack(specs))
        f.create_dataset("onset", data=np.array(onsets, dtype=np.float32))
        f.create_dataset("offset", data=np.array(offsets, dtype=np.float32))
        f.create_dataset("audio_filename", data=np.array(audio_filenames, dtype="S128"))
        f.create_dataset("latent_means_official", data=np.stack(latent_means))
        f.create_dataset("z_70", data=np.array(z_70, dtype=np.int64))
        f.create_dataset("cohort", data=np.array(cohort, dtype="S8"))
        f.create_dataset("z70_confidence", data=np.array(z70_conf, dtype=np.float32))
    print(f"wrote {len(specs)} specs -> {out_file} (dropped {n_dropped} invalid/empty segments)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/mnt/home/the10/ceph/dataset/dryad_gerbil"))
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir; writes {out}/syllables_0000.hdf5")
    args = ap.parse_args()
    build_specs(args.data, args.out)


if __name__ == "__main__":
    main()
