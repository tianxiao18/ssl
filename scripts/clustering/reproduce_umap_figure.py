"""Reproduce autoencoded-vocal-analysis/notebooks/figure2-umap-vocalization-examples.ipynb's
layout (12 border example spectrograms around a central UMAP scatter, colored
by z_70), built two ways for comparison:

  A. official latents : the paper's own saved 32-d latent_means for our
     local subset. Since these are literally the checkpoint's own
     precomputed outputs for these exact vocalizations, this is the closest
     thing to a "1-to-1" version -- the input latents are identical to what
     the paper used.
  B. checkpoint latents : latents produced BY US by running checkpoint_050.tar
     over our local spectrograms (dryad_to_ava_specs.py's regenerated specs
     -> StableVAE.encode).

UMAP is fit ONCE on the two latent sets stacked together (not fit
independently per source), so the resulting 2-D layout is the SAME
coordinate frame for both panels -- unlike fitting separately, where each
side gets its own arbitrarily-oriented embedding (UMAP layouts are only
defined up to rotation/reflection, so two independent fits on different
input matrices will not line up even with a fixed random_state). This makes
the two panels directly, visually comparable: a cluster landing "in the same
place" in both actually means something.

Both are colored by the paper's own saved z_70 (no GMM refit here). Border
examples are the same 12 states used in the notebook
([8, 66, 38, 17, 19, 30, 51, 59, 31, 36, 57, 5]); state 8 has zero local
candidates and gets an empty panel. Per state, the exemplar is the local
vocalization with the highest z70_confidence (= paper's own prob_z_70[z_70]).

Border spectrograms are rendered the same way the notebook does -- raw,
zero-padded waveform through ax.specgram(NFFT=512, noverlap=256, magma) --
but with a per-panel percentile-based clim instead of the notebook's fixed
clim=(-10,20): our wavs are float64 at a very different absolute amplitude
scale (~1e-2) than whatever the notebook's original pydub-loaded audio was
on, so the fixed clim renders everything black. Adaptive clim keeps the same
visual style without depending on that absolute scale.

Population is restricted to cohort c2 (what the notebook itself used, and
the cohort for which most of the 12 states have local candidates).

    python scripts/clustering/reproduce_umap_figure.py --specs outputs/dryad_holdout \
        --data /mnt/home/the10/ceph/dataset/dryad_gerbil \
        --out-official outputs/dryad_holdout/umap_repro_official.png \
        --out-checkpoint outputs/dryad_holdout/umap_repro_checkpoint.png
"""
import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vae"))
from dryad_to_ava_specs import parse_vec

PAPER_STATES = [8, 66, 38, 17, 19, 30, 51, 59, 31, 36, 57, 5]
BORDER_POSITIONS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
                    (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0)]
COLORS = ['#ff0000', '#ff8000', '#ffff00', '#80ff00', '#08eb8c', '#029221',
          '#00ffff', '#0080ff', '#0000ff', '#8000ff', '#ff00ff', '#ff0080']
FS = 125000
MAX_DUR = 0.3


def load_local(specs_dir, source, cohort=b'c2'):
    files = sorted(Path(specs_dir).glob("*.hdf5"))
    fields = {}
    for fn in files:
        with h5py.File(fn, "r") as f:
            for key in ["z_70", "z70_confidence", "cohort", "onset", "offset", "audio_filename"]:
                fields.setdefault(key, []).append(f[key][:])
    fields = {k: np.concatenate(v) for k, v in fields.items()}

    npy = {"official": "latent_official.npy", "checkpoint": "latent_mine.npy"}[source]
    latents = np.load(specs_dir / "confirm" / npy) if (specs_dir / "confirm" / npy).exists() \
        else np.load(specs_dir / npy)
    mask = fields["cohort"] == cohort
    fields = {k: v[mask] for k, v in fields.items()}
    return fields, latents[mask]


def pick_examples(fields, states):
    picks = {}
    for s in states:
        idx = np.flatnonzero(fields["z_70"] == s)
        picks[s] = idx[np.argmax(fields["z70_confidence"][idx])] if len(idx) else None
    return picks


def find_substitute_state(data_dir, cohort, missing_state, local_states):
    """Nearest z_70 cluster to `missing_state`, among clusters we actually
    have a local example for. "Nearest" = smallest Euclidean distance between
    per-cluster centroids of the paper's own official latent_means, computed
    over the full cohort population (a geometry lookup on data already
    released in the CSV -- not audio, not something re-derived by us, so it's
    not part of the verified Track A/B latents, just used to pick a
    stand-in)."""
    chunks = []
    for chunk in pd.read_csv(data_dir / 'vocalization_df.csv',
                             usecols=['cohort', 'z_70', 'latent_means'], chunksize=100_000):
        sub = chunk[chunk['cohort'] == cohort]
        if len(sub):
            chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True)
    df['latent_means'] = df['latent_means'].apply(parse_vec)
    centroids = {k: np.mean(np.stack(g['latent_means'].values), axis=0)
                for k, g in df.groupby('z_70')}
    target = centroids[missing_state]
    candidates = [s for s in local_states if s != missing_state and s in centroids]
    dists = {s: float(np.linalg.norm(centroids[s] - target)) for s in candidates}
    best = min(dists, key=dists.get)
    print(f"z_70={missing_state}: no local example. Nearest cluster with a local "
          f"example is z_70={best} (centroid distance {dists[best]:.3f}); "
          f"next closest: {sorted(dists.items(), key=lambda kv: kv[1])[1:4]}")
    return best


def load_context_audio(data_dir, audio_filename, onset, offset, fs=FS, max_dur=MAX_DUR):
    """max_dur-long window of REAL audio centered on the vocalization, rather
    than isolating just the onset:offset samples and zero-padding around them.
    Zero-padding renders as flat, uniform black (literal silence -> -inf dB),
    which is visibly darker than the recording's own actual background noise
    floor (real mic hiss, never exactly zero) -- creating a hard rectangular
    seam in the specgram. Using real neighboring audio for context keeps the
    noise floor continuous across the whole panel, so it blends."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=WavFileWarning)
        file_fs, audio = wavfile.read(str(data_dir / audio_filename))
    assert file_fs == fs, f"expected fs={fs}, got {file_fs}"
    n = int(max_dur * fs)
    mid_sample = int(round((onset + offset) / 2 * fs))
    start_i = mid_sample - n // 2
    end_i = start_i + n
    v = np.zeros(n)
    src_lo, src_hi = max(0, start_i), min(len(audio), end_i)
    dst_lo = src_lo - start_i
    v[dst_lo:dst_lo + (src_hi - src_lo)] = audio[src_lo:src_hi]
    return v


def specgram_clim(v, fs, nfft, noverlap):
    """Percentile-based clim in place of the notebook's fixed (-10,20), which
    assumed a different absolute audio amplitude scale than our wavs are on.
    vmin at the 95th percentile clips the noise floor to black (most
    time-frequency bins are just background hiss); vmax at 99.8 keeps the
    call's own bright trace visible without saturating -- matches the
    notebook's "call glows against black" look instead of a fully-populated
    noise texture."""
    import matplotlib.mlab as mlab
    Pxx, _, _ = mlab.specgram(v, NFFT=nfft, noverlap=noverlap, Fs=fs)
    dB = 10 * np.log10(Pxx[Pxx > 0])
    return float(np.percentile(dB, 95)), float(np.percentile(dB, 99.8))


def render(fields, emb, states, picks, substituted, title, out_path, data_dir):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 4)

    for (r, c), color, state in zip(BORDER_POSITIONS, COLORS, states):
        ax = fig.add_subplot(gs[r, c])
        idx = picks[state]
        label = f"z70={state}" + (f"\n(near {substituted[state]})" if state in substituted else "")
        if idx is None:
            ax.set_facecolor('#dddddd')
            ax.text(0.5, 0.5, f"{label}\nno local\nexample", ha='center', va='center', fontsize=8)
        else:
            audio_filename = fields["audio_filename"][idx].decode()
            onset, offset = float(fields["onset"][idx]), float(fields["offset"][idx])
            v = load_context_audio(data_dir, audio_filename, onset, offset)
            clim = specgram_clim(v, FS, 512, 256)
            ax.specgram(v, NFFT=512, noverlap=256, Fs=FS, clim=clim, cmap='magma')
            ax.set_facecolor('k')
        ax.set_title(label, fontsize=8, color=color)
        ax.set_xticks([]); ax.set_yticks([])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_color(color)
            ax.spines[axis].set_linewidth(5)

    ax = fig.add_subplot(gs[1:3, 1:3])
    ax.plot(emb[:, 0], emb[:, 1], '.k', markersize=2, alpha=0.3)
    for state, color in zip(states, COLORS):
        idx = picks[state]
        if idx is None:
            continue
        ax.plot(emb[idx, 0], emb[idx, 1], 'o', markersize=10,
               markeredgecolor='k', markeredgewidth=2, c=color)
    ax.set_title(title, fontsize=10)
    ax.axis('equal')
    ax.axis('off')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("/mnt/home/the10/ceph/dataset/dryad_gerbil"))
    ap.add_argument("--specs", type=Path, required=True, help="dir from dryad_to_ava_specs.py")
    ap.add_argument("--out-official", type=Path, required=True)
    ap.add_argument("--out-checkpoint", type=Path, required=True)
    ap.add_argument("--cohort", default="c2")
    args = ap.parse_args()

    import umap
    import matplotlib
    matplotlib.use("Agg")

    fields_official, latents_official = load_local(args.specs, "official", args.cohort.encode())
    fields_checkpoint, latents_checkpoint = load_local(args.specs, "checkpoint", args.cohort.encode())
    assert len(latents_official) == len(latents_checkpoint), \
        "official/checkpoint latent counts disagree -- specs regenerated after one of them was computed?"
    fields = fields_official  # same underlying rows/cohort mask for both sources

    print(f"cohort {args.cohort}: {len(latents_official)} local vocalizations, "
          f"UMAP fit jointly on official+checkpoint (n={2 * len(latents_official)} stacked)")
    stacked = np.concatenate([latents_official, latents_checkpoint], axis=0)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(stacked)
    n = len(latents_official)
    emb_official, emb_checkpoint = emb[:n], emb[n:]

    local_states = set(np.unique(fields["z_70"]).tolist())
    states = list(PAPER_STATES)
    substituted = {}
    for i, s in enumerate(states):
        if s not in local_states:
            sub = find_substitute_state(args.data, args.cohort, s, local_states)
            substituted[sub] = s
            states[i] = sub

    picks = pick_examples(fields, states)
    missing = [s for s, v in picks.items() if v is None]
    print(f"{len(states) - len(missing)}/{len(states)} states have a local "
          f"{args.cohort} example (missing: {missing})")

    title = (f"UMAP jointly fit on official+checkpoint latents "
             f"(n={n}, cohort {args.cohort}) -- {{}}")
    render(fields, emb_official, states, picks, substituted,
           title.format("official"), args.out_official, args.data)
    render(fields, emb_checkpoint, states, picks, substituted,
           title.format("checkpoint"), args.out_checkpoint, args.data)


if __name__ == "__main__":
    main()
