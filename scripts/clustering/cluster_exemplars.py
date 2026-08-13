"""Visualize the most-representative spectrogram for each z_70 cluster (paper's
GMM, k=70), among the vocalizations we have local spectrograms for.

"Most representative" = the vocalization with the highest z70_confidence, i.e.
the paper's own prob_z_70[z_70] -- the GMM's posterior probability of the
vocalization belonging to its assigned cluster. Not every one of the 70
clusters necessarily has a member in this small (~2805-vocalization) local
subset; those are skipped and reported.

    python scripts/clustering/cluster_exemplars.py --specs outputs/dryad_holdout \
        --out outputs/dryad_holdout/cluster_exemplars.png
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def load(specs_dir):
    files = sorted(Path(specs_dir).glob("*.hdf5"))
    specs, z70, conf = [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            specs.append(f["specs"][:])
            z70.append(f["z_70"][:])
            conf.append(f["z70_confidence"][:])
    return np.concatenate(specs), np.concatenate(z70), np.concatenate(conf)


def pick_exemplars(specs, z70, conf, n_clusters):
    exemplars = {}
    for k in range(n_clusters):
        idx = np.flatnonzero(z70 == k)
        if len(idx) == 0:
            continue
        best = idx[np.argmax(conf[idx])]
        exemplars[k] = (best, len(idx), float(conf[best]))
    return exemplars


def plot_grid(specs, exemplars, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = sorted(exemplars)
    ncols = 10
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.9 * nrows))
    for ax, k in zip(axes.flat, ks):
        idx, n_local, conf = exemplars[k]
        ax.imshow(specs[idx], origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(f"z70={k}\nn={n_local} p={conf:.2f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(ks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", type=Path, required=True, help="dir from dryad_to_ava_specs.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-clusters", type=int, default=70)
    args = ap.parse_args()

    specs, z70, conf = load(args.specs)
    exemplars = pick_exemplars(specs, z70, conf, args.n_clusters)
    missing = sorted(set(range(args.n_clusters)) - set(exemplars))
    print(f"{len(exemplars)}/{args.n_clusters} clusters have a local exemplar"
          f" (missing: {missing})" if missing else f"all {args.n_clusters} clusters represented")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_grid(specs, exemplars, args.out)


if __name__ == "__main__":
    main()
