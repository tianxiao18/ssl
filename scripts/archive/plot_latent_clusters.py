"""Re-plot the AVA VAE latent UMAP (raw vs masked), colored by KMeans cluster,
and show input vs. reconstructed spectrogram for the most representative
latent in each cluster.

Reuses artifacts already written by train_ava_ab.py:
  outputs/compare/latent_{arm}.npy     -- latent means, row-aligned across arms
  outputs/compare/{arm}/checkpoint.tar -- trained VAE weights, for decoding
  {data}/{arm}/*.hdf5                  -- original spectrograms, for ground truth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vae"))  # for train_ava_ab
from train_ava_ab import StableVAE, make_loader
from ava.models.vae import X_SHAPE

ARMS = ["raw", "masked"]


def representative_indices(zc, labels, k):
    """Index of the point closest to each cluster centroid, in standardized space."""
    reps = []
    for c in range(k):
        members = np.where(labels == c)[0]
        centroid = zc[members].mean(0)
        d = np.linalg.norm(zc[members] - centroid, axis=1)
        reps.append(members[d.argmin()])
    return np.array(reps)


def decode(ckpt_dir, z_reps, z_dim):
    model = StableVAE(save_dir=str(ckpt_dir), z_dim=z_dim, device_name="cpu")
    model.load_state(str(ckpt_dir / "checkpoint.tar"))
    model.eval()
    with torch.no_grad():
        x = model.decode(torch.from_numpy(z_reps).float())
    return x.view(-1, *X_SHAPE).numpy()


def true_inputs(data_dir, arm, rep_idx):
    loader = make_loader(data_dir / arm, batch_size=64, shuffle=False)
    specs = loader.dataset[list(rep_idx)]
    return torch.stack(specs).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare-dir", type=Path, default=Path("outputs/compare"))
    ap.add_argument("--data", type=Path, default=None,
                     help="dir with raw/ and masked/ hdf5 (adds a ground-truth input row per arm)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--out-prefix", type=str, default=None,
                     help="output basename stem (default: latent_umap_ab_k{k})")
    args = ap.parse_args()
    stem = args.out_prefix or f"latent_umap_ab_k{args.k}"

    import umap
    from sklearn.cluster import KMeans
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=max(args.k - 1, 1))

    umap_fig, umap_axes = plt.subplots(1, 2, figsize=(12, 5.4))
    n_rows = 4 if args.data else 2
    rec_fig, rec_axes = plt.subplots(n_rows, args.k, figsize=(1.3 * args.k, 1.4 * n_rows))

    for row_arm, arm in enumerate(ARMS):
        z = np.load(args.compare_dir / f"latent_{arm}.npy")
        zc = (z - z.mean(0)) / (z.std(0) + 1e-9)
        labels = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(zc)
        rep_idx = representative_indices(zc, labels, args.k)

        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(z)
        ax = umap_axes[row_arm]
        ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.6, c=labels, cmap=cmap, norm=norm)
        ax.scatter(emb[rep_idx, 0], emb[rep_idx, 1], s=80, facecolors="none",
                   edgecolors="black", linewidths=1.2)
        ax.set_title(f"{arm}  (n={len(emb)}, k={args.k})")
        ax.set_xticks([]); ax.set_yticks([])

        recs = decode(args.compare_dir / arm, z[rep_idx], args.z_dim)
        counts = np.bincount(labels, minlength=args.k)

        if args.data:
            inputs = true_inputs(args.data, arm, rep_idx)
            row_pairs = [(f"{arm} input", inputs), (f"{arm} recon", recs)]
            base_row = row_arm * 2
        else:
            row_pairs = [(arm, recs)]
            base_row = row_arm

        for sub_row, (label, imgs) in enumerate(row_pairs):
            for c in range(args.k):
                a = rec_axes[base_row + sub_row, c]
                a.imshow(imgs[c], origin="lower", cmap="viridis")
                a.set_xticks([]); a.set_yticks([])
                color = cmap(norm(c))
                for spine in a.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                if sub_row == 0:
                    a.set_title(f"c{c} n={counts[c]}", fontsize=7)
            rec_axes[base_row + sub_row, 0].set_ylabel(label, fontsize=9)

    umap_fig.tight_layout()
    umap_out = args.compare_dir / f"{stem}.png"
    umap_fig.savefig(umap_out, dpi=120)
    print("wrote", umap_out)

    rec_fig.tight_layout()
    rec_out = args.compare_dir / f"{stem}_reps.png"
    rec_fig.savefig(rec_out, dpi=120)
    print("wrote", rec_out)


if __name__ == "__main__":
    main()
