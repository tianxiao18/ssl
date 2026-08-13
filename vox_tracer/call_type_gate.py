"""Gate SAM3 candidate exemplars by whether they resemble a known dryad_gerbil
call type, using the paper's own VAE latent space and cluster assignments.

Candidates come from vox_tracer.sam3_runner's ridge-filter connected components
-- one per window can be a real call, a broadband noise streak, or a speckle.
Each candidate's pixel bbox is mapped to (onset, offset) in seconds, re-spectrogrammed
with AVA's own preprocessing (the exact recipe checkpoint_050.tar was trained under),
encoded to a 32-d latent, and compared to the 70 cluster centroids computed from the
paper's own local vocalizations (outputs/dryad_holdout, from scripts/vae/dryad_to_ava_specs.py).
A candidate too far from every centroid isn't a recognized call type and is dropped
before it reaches SAM3 -- this is what keeps a quiet window from prompting SAM3 with
noise, and what lets a busy window offer several distinct real call shapes instead of
just whichever ridge blob happened to be loudest (see pick_best_candidate in sam3_runner.py).
"""
import sys
import warnings
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch

# train_ava_ab.py (StableVAE) lives in scripts/vae/, not a package -- add it
# to sys.path the same way running any scripts/*.py entrypoint would.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "vae")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from ava.models.vae import X_SHAPE
import ava.preprocessing.utils as ava_pp_utils
from ava.preprocessing.utils import get_spec


class _Interp2DCompat:
    """scipy.interpolate.interp2d was removed in SciPy 1.14+; ava.preprocessing.utils.get_spec
    still calls it. Same shim as scripts/vae/dryad_to_ava_specs.py, duplicated here so this
    module doesn't depend on that script's import-time side effects."""

    def __init__(self, x, y, z, copy=False, bounds_error=False, fill_value=None):
        from scipy.interpolate import RectBivariateSpline
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        self._xlim = (x.min(), x.max())
        self._ylim = (y.min(), y.max())
        self._fill_value = fill_value
        self._spline = RectBivariateSpline(y, x, z)

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

Z_DIM = 32

# Exact preprocess_params checkpoint_050.tar was trained under (see
# scripts/vae/dryad_to_ava_specs.py); must match so this VAE sees the same kind of
# input it was trained on.
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
TARGET_FREQS = np.linspace(PREPROCESS_PARAMS['min_freq'], PREPROCESS_PARAMS['max_freq'],
                           PREPROCESS_PARAMS['num_freq_bins'])


def load_cluster_centroids(specs_dir, pct=95.0):
    """(centroid_matrix [K,32], cluster_ids [K], distance_threshold) from local
    *.hdf5 files written by scripts/vae/dryad_to_ava_specs.py (z_70 + latent_means_official).

    distance_threshold is the `pct`-th percentile of each vocalization's distance
    to its OWN assigned cluster's centroid -- one global cutoff standing in for
    "how far a real call can be from its nearest cluster", not a per-cluster
    covariance model. Simple on purpose.
    """
    z70, latents = [], []
    for fn in sorted(glob(str(Path(specs_dir) / "*.hdf5"))):
        with h5py.File(fn, "r") as f:
            z70.append(f["z_70"][:])
            latents.append(f["latent_means_official"][:])
    if not z70:
        raise FileNotFoundError(f"no *.hdf5 with z_70/latent_means_official in {specs_dir}")
    z70 = np.concatenate(z70)
    latents = np.concatenate(latents)

    cluster_ids = np.unique(z70)
    centroids = np.stack([latents[z70 == k].mean(axis=0) for k in cluster_ids])

    own_centroid = centroids[np.searchsorted(cluster_ids, z70)]
    own_dist = np.linalg.norm(latents - own_centroid, axis=1)
    threshold = float(np.percentile(own_dist, pct))
    return centroids, cluster_ids, threshold


def build_vae(checkpoint, save_dir):
    """Load the paper's VAE (checkpoint_050.tar) in eval mode for encoding candidates."""
    from train_ava_ab import StableVAE
    vae = StableVAE(save_dir=str(save_dir), z_dim=Z_DIM)
    vae.load_state(str(checkpoint))
    vae.eval()
    return vae


def encode_bbox(vae, audio, sr, bbox, t0, t1, W):
    """Latent z for one candidate bbox's audio span, or None if the segment is empty.

    bbox is (x0, y0, x1, y1) in image pixels; only the x-span is used -- get_spec
    re-derives its own frequency axis over the full species band (500-62500 Hz),
    same as every other AVA spectrogram this VAE was trained on.
    """
    x0, _, x1, _ = bbox
    onset  = t0 + (x0 / W) * (t1 - t0)
    offset = t0 + (x1 / W) * (t1 - t0)
    if offset <= onset:
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # get_spec warns on segments > max_dur; harmless here
        spec, valid = get_spec(onset, offset, audio, PREPROCESS_PARAMS, sr, target_freqs=TARGET_FREQS)
    if not valid or not np.any(spec):
        return None
    x = torch.from_numpy(spec).float().unsqueeze(0).to(vae.device)
    with torch.no_grad():
        mu, _, _ = vae.encode(x)
    return mu[0].cpu().numpy()


def nearest_cluster(z, centroids, cluster_ids):
    """(cluster_id, distance) of the centroid nearest to z."""
    d = np.linalg.norm(centroids - z, axis=1)
    i = int(np.argmin(d))
    return int(cluster_ids[i]), float(d[i])
