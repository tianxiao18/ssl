"""Post-hoc training diagnostics for the raw-vs-masked AVA VAE A/B run.

Everything here reuses artifacts already on disk (checkpoints, latents,
metrics.json) -- no retraining needed:

  A) train loss curve per arm, from checkpoint['loss']['train']
     (there's no validation curve: get_syllable_partition(split=1) puts 100%
     of data in 'train' and the 'test' loader is never evaluated -- confirmed
     empty checkpoint['loss']['test'] -- so this is train-only.)
  B) active latent dimensions: per-dimension variance of the encoder's mu
     across the dataset (classic posterior-collapse check -- a dim with near-
     zero variance across every input carries no information).
  C) reconstruction error distribution: per-sample MSE between input spectrogram
     and its decoded reconstruction, on a random sample of the training data.
  D) silhouette score vs k (from metrics_{arm}.json's silhouette_per_k sweep).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vae"))
from train_ava_ab import StableVAE, make_loader

ARMS = ["raw", "masked"]
COLORS = {"raw": "#4c72b0", "masked": "#dd8452"}


def load_metrics(compare_dir, arm):
    per_arm = compare_dir / f"metrics_{arm}.json"
    if per_arm.exists():
        return json.loads(per_arm.read_text())
    combined = json.loads((compare_dir / "metrics.json").read_text())
    return combined[arm]


def reconstruction_mse(compare_dir, data_dir, arm, z_dim, n_sample, seed=0):
    z = np.load(compare_dir / f"latent_{arm}.npy")
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(z), size=min(n_sample, len(z)), replace=False)
    idx.sort()  # SyllableDataset indexing wants ascending order

    loader = make_loader(data_dir / arm, batch_size=64, shuffle=False)
    inputs = torch.stack(loader.dataset[list(idx)]).numpy()

    model = StableVAE(save_dir=str(compare_dir / arm), z_dim=z_dim, device_name="cpu")
    model.load_state(str(compare_dir / arm / "checkpoint.tar"))
    model.eval()
    with torch.no_grad():
        recon = model.decode(torch.from_numpy(z[idx]).float()).view(-1, 128, 128).numpy()

    mse = ((inputs - recon) ** 2).reshape(len(idx), -1).mean(axis=1)
    return mse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare-dir", type=Path, default=Path("outputs/compare"))
    ap.add_argument("--data", type=Path, default=None,
                     help="dir with raw/ and masked/ hdf5 (needed for panel C; skipped if omitted)")
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--n-recon-sample", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.compare_dir / "training_diagnostics.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 4 if args.data else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2))
    ax_loss, ax_au, ax_sil = axes[0], axes[1], axes[-1]

    # A) train loss curve
    for arm in ARMS:
        ckpt = torch.load(args.compare_dir / arm / "checkpoint.tar", map_location="cpu")
        tl = ckpt["loss"]["train"]
        epochs = sorted(tl.keys())
        ax_loss.plot(epochs, [tl[e] for e in epochs], label=arm, color=COLORS[arm])
        assert not ckpt["loss"]["test"], f"expected empty test loss for {arm}, found data"
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("train loss (-ELBO)")
    ax_loss.set_title("train loss (no val split exists)")
    ax_loss.legend()

    # B) active latent dimensions (posterior-collapse check)
    au_threshold = 0.01
    for arm in ARMS:
        z = np.load(args.compare_dir / f"latent_{arm}.npy")
        var = z.var(axis=0)
        order = np.argsort(-var)
        n_active = int((var > au_threshold).sum())
        ax_au.plot(range(len(var)), var[order], label=f"{arm} ({n_active}/{len(var)} active)",
                  color=COLORS[arm], marker="o", markersize=3)
    ax_au.axhline(au_threshold, color="gray", linestyle="--", linewidth=1, label=f"AU threshold={au_threshold}")
    ax_au.set_yscale("log")
    ax_au.set_xlabel("latent dim (sorted by variance)"); ax_au.set_ylabel("Var[mu] across dataset")
    ax_au.set_title("active latent dimensions")
    ax_au.legend(fontsize=8)

    panel = 2
    if args.data:
        ax_mse = axes[2]
        for arm in ARMS:
            mse = reconstruction_mse(args.compare_dir, args.data, arm, args.z_dim, args.n_recon_sample)
            ax_mse.hist(mse, bins=40, alpha=0.55, label=f"{arm} (mean={mse.mean():.4f})", color=COLORS[arm])
        ax_mse.set_xlabel("per-crop reconstruction MSE"); ax_mse.set_ylabel("count")
        ax_mse.set_title(f"reconstruction error (n={args.n_recon_sample} sample)")
        ax_mse.legend(fontsize=8)
        panel = 3

    # D) silhouette vs k
    for arm in ARMS:
        m = load_metrics(args.compare_dir, arm)
        per_k = {int(k): v for k, v in m["silhouette_per_k"].items()}
        ks = sorted(per_k)
        ax_sil.plot(ks, [per_k[k] for k in ks], marker="o", label=arm, color=COLORS[arm])
    ax_sil.set_xlabel("k"); ax_sil.set_ylabel("silhouette")
    ax_sil.set_title("cluster separability vs k")
    ax_sil.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
