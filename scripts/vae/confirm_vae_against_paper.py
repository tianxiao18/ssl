"""Track A: confirm the local VAE pipeline against the paper's own results.

Loads the paper's pretrained checkpoint (checkpoint_050.tar) into the *exact*
model class trained by train_ava_ab.py (StableVAE == stock ava.models.vae.VAE
plus numerics-only guards that don't touch the architecture or encode() math),
encodes the spectrograms produced by dryad_to_ava_specs.py, and compares the
resulting latent means against the paper's own official latent_means for the
same vocalizations (carried along as provenance in that hdf5).

If preprocessing is a faithful reproduction of the paper's pipeline and the
checkpoint loads correctly, "my spectrogram -> paper's weights -> latent"
should reproduce "paper's own latent" almost exactly -- encode() is a
deterministic function of the input pixels (mu, no sampling), so this is a
strict equality check up to preprocessing noise, not a statistical one.

    python scripts/vae/confirm_vae_against_paper.py \
        --specs outputs/dryad_holdout \
        --checkpoint /mnt/home/the10/ceph/dataset/dryad_gerbil/checkpoint_050.tar \
        --out outputs/dryad_holdout/confirm
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from train_ava_ab import StableVAE, make_loader

Z_DIM = 32


def load_official(specs_dir):
    """Official latents/labels from the single hdf5 written by
    dryad_to_ava_specs.py, in the same row order SyllableDataset reads them."""
    files = sorted(Path(specs_dir).glob("*.hdf5"))
    latents, z70, audio_filename, onset = [], [], [], []
    for fn in files:
        with h5py.File(fn, "r") as f:
            latents.append(f["latent_means_official"][:])
            z70.append(f["z_70"][:])
            audio_filename.append(f["audio_filename"][:])
            onset.append(f["onset"][:])
    return (np.concatenate(latents), np.concatenate(z70),
            np.concatenate(audio_filename), np.concatenate(onset))


def per_dim_pearson(a, b):
    return np.array([np.corrcoef(a[:, d], b[:, d])[0, 1] for d in range(a.shape[1])])


def scatter_grid(mine, official, r, var_official, order, out_png):
    """Panels ordered by descending official variance: AVA's rank-1-covariance
    VAE collapses most dims to the prior (near-zero per-sample variance), where
    Pearson r is numerically unstable/meaningless -- sorting puts the handful
    of informative ("active") dims first so agreement is visible at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = mine.shape[1]
    ncols = 8
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    for ax, i in zip(axes.flat, order):
        ax.scatter(official[:, i], mine[:, i], s=4, alpha=0.4, c="#4c72b0")
        lo = min(official[:, i].min(), mine[:, i].min())
        hi = max(official[:, i].max(), mine[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"z{i}  r={r[i]:.3f}  var={var_official[i]:.4f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[d:]:
        ax.axis("off")
    fig.supxlabel("official latent_means (paper)")
    fig.supylabel("mine (checkpoint reloaded)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print("wrote", out_png)


def umap_side_by_side(mine, official, z70, out_png):
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, z, title in [(axes[0], official, "official (paper)"), (axes[1], mine, "mine (checkpoint reloaded)")]:
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(z)
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.7, c=z70, cmap="tab20")
        ax.set_title(f"{title}  (n={len(emb)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label="z_70 cluster (paper)", fraction=0.03)
    fig.savefig(out_png, dpi=130)
    print("wrote", out_png)


def joint_umap_overlay(mine, official, z70, out_png):
    """Fit UMAP ONCE on official+mine stacked together, so both point clouds
    share literal x/y coordinates -- unlike umap_side_by_side, where each
    side gets its own independent, arbitrarily-oriented 2-D layout, so a
    cluster landing "in the same place" in both panels here actually means
    something. Same two-panel-by-z_70 layout as umap_side_by_side, just fit
    jointly instead of separately."""
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stacked = np.concatenate([official, mine], axis=0)
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(stacked)
    n = len(official)
    embs = {"official (paper)": emb[:n], "mine (checkpoint reloaded)": emb[n:]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (title, e) in zip(axes, embs.items()):
        sc = ax.scatter(e[:, 0], e[:, 1], s=8, alpha=0.7, c=z70, cmap="tab20")
        ax.set_title(f"{title}  (n={len(e)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label="z_70 cluster (paper)", fraction=0.03)
    fig.savefig(out_png, dpi=130)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", type=Path, required=True, help="dir from dryad_to_ava_specs.py")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/mnt/home/the10/ceph/dataset/dryad_gerbil/checkpoint_050.tar"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    official, z70, _, _ = load_official(args.specs)
    print(f"{len(official)} vocalizations with official latents")

    model = StableVAE(save_dir=str(args.out), z_dim=Z_DIM)
    model.load_state(str(args.checkpoint))
    model.eval()

    loader = make_loader(args.specs, args.batch_size, shuffle=False)
    mine = model.get_latent(loader)
    assert mine.shape == official.shape, f"{mine.shape} vs {official.shape}"

    r = per_dim_pearson(mine, official)
    cos = np.sum(mine * official, axis=1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(official, axis=1) + 1e-9)
    mse_per_row = np.mean((mine - official) ** 2, axis=1)

    # AVA's rank-1-covariance VAE typically collapses most z dims to the prior
    # (near-zero per-sample variance in mu); Pearson r on those is numerically
    # unstable and not informative, so report a variance-weighted r alongside
    # the naive mean, plus a breakdown restricted to the "active" dims (those
    # carrying most of the official latent's variance).
    var_official = official.var(axis=0)
    order = np.argsort(-var_official)
    weighted_r = float(np.average(r, weights=var_official))
    active = order[np.cumsum(var_official[order]) <= 0.99 * var_official.sum()]
    if len(active) == 0:
        active = order[:1]

    metrics = {
        "n": int(len(official)),
        "pearson_r_per_dim": r.tolist(),
        "official_variance_per_dim": var_official.tolist(),
        "pearson_r_mean_unweighted": float(np.nanmean(r)),
        "pearson_r_variance_weighted": weighted_r,
        "active_dims_99pct_variance": active.tolist(),
        "pearson_r_mean_active_dims": float(np.nanmean(r[active])),
        "cosine_sim_mean": float(np.mean(cos)),
        "cosine_sim_median": float(np.median(cos)),
        "mse_mean": float(np.mean(mse_per_row)),
        "mse_median": float(np.median(mse_per_row)),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    scatter_grid(mine, official, r, var_official, order, args.out / "latent_scatter_grid.png")
    umap_side_by_side(mine, official, z70, args.out / "latent_umap_ab.png")
    joint_umap_overlay(mine, official, z70, args.out / "latent_umap_joint.png")

    np.save(args.out / "latent_mine.npy", mine)
    np.save(args.out / "latent_official.npy", official)


if __name__ == "__main__":
    main()
