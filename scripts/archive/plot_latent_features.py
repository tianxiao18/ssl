"""Color the raw-vs-masked latent UMAP by acoustic features and detector TP/FP.

Joins outputs/compare/latent_{arm}.npy (row-aligned VAE latents) against
outputs/spectral_features/sam3_best.csv (librosa features + GT-matched TP/FP
label per SAM3 detection, keyed by (recording, file_name, ann_id)) via
outputs/compare/crop_features.csv, built once by joining on those same keys.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["raw", "masked"]
CONTINUOUS_FEATURES = [
    ("centroid_mean", "mean frequency (Hz)"),
    ("bandwidth_mean", "bandwidth (Hz)"),
    ("duration_sec", "duration (s)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare-dir", type=Path, default=Path("outputs/compare"))
    ap.add_argument("--out-prefix", type=str, default="latent_umap_ab")
    args = ap.parse_args()

    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat = pd.read_csv(args.compare_dir / "crop_features.csv").sort_values("row")
    assert (feat["row"].values == np.arange(len(feat))).all()

    embeddings = {}
    for arm in ARMS:
        z = np.load(args.compare_dir / f"latent_{arm}.npy")
        embeddings[arm] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(z)

    # --- continuous spectral/geometric features ---
    fig, axes = plt.subplots(len(CONTINUOUS_FEATURES), len(ARMS),
                             figsize=(5.4 * len(ARMS), 4.4 * len(CONTINUOUS_FEATURES)))
    for row, (col, label) in enumerate(CONTINUOUS_FEATURES):
        vals = feat[col].values
        vmin, vmax = np.nanpercentile(vals, [2, 98])  # robust to outliers
        for c, arm in enumerate(ARMS):
            ax = axes[row, c]
            emb = embeddings[arm]
            sc = ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.7, c=vals,
                            cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{arm}: {label}" if row == 0 else arm, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.tight_layout()
    out = args.compare_dir / f"{args.out_prefix}_spectral.png"
    fig.savefig(out, dpi=120)
    print("wrote", out)

    # --- TP / FP (vs. GT vox events, from spectral_features.py) ---
    fig2, axes2 = plt.subplots(1, len(ARMS), figsize=(6 * len(ARMS), 5.4))
    label = feat["label"].values
    is_tp, is_fp, is_na = label == "tp", label == "fp", pd.isna(label)
    for ax, arm in zip(axes2, ARMS):
        emb = embeddings[arm]
        ax.scatter(emb[is_na, 0], emb[is_na, 1], s=6, alpha=0.3, c="lightgray",
                   label=f"no GT (n={is_na.sum()})")
        ax.scatter(emb[is_tp, 0], emb[is_tp, 1], s=6, alpha=0.5, c="#4c72b0",
                   label=f"TP (n={is_tp.sum()})")
        ax.scatter(emb[is_fp, 0], emb[is_fp, 1], s=10, alpha=0.85, c="#d62728",
                   label=f"FP (n={is_fp.sum()})", zorder=3)
        ax.set_title(arm)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, markerscale=2)
    fig2.tight_layout()
    out2 = args.compare_dir / f"{args.out_prefix}_tpfp.png"
    fig2.savefig(out2, dpi=120)
    print("wrote", out2)


if __name__ == "__main__":
    main()
