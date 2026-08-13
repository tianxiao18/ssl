"""Highlight VAE-training crops that overlap a "combined" (overlapping-call)
ground-truth annotation, on the raw-vs-masked latent UMAP.

Needs outputs/compare/is_combined.npy, a boolean mask over the latent rows
(built by cross-referencing window_start_sec/window_end_sec in the training
HDF5s against *_annotations_gt.csv rows where name == "combined").
"""
import argparse
from pathlib import Path

import numpy as np

ARMS = ["raw", "masked"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare-dir", type=Path, default=Path("outputs/compare"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.compare_dir / "latent_umap_ab_combined.png")

    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_combined = np.load(args.compare_dir / "is_combined.npy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, arm in zip(axes, ARMS):
        z = np.load(args.compare_dir / f"latent_{arm}.npy")
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(z)

        ax.scatter(emb[~is_combined, 0], emb[~is_combined, 1], s=6, alpha=0.35,
                   c="#4c72b0", label=f"other (n={(~is_combined).sum()})")
        ax.scatter(emb[is_combined, 0], emb[is_combined, 1], s=10, alpha=0.85,
                   c="#d62728", label=f'"combined" (n={is_combined.sum()})', zorder=3)
        ax.set_title(f"{arm}  (n={len(emb)})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
