"""Train paired AVA VAEs on the raw vs masked datasets and compare their latents.

Consumes the output of ``coco_to_ava_specs.py`` (``{data}/raw`` and
``{data}/masked``). Trains one stock ``ava.models.vae.VAE`` per arm under an
*identical* seed / architecture / epoch budget, so any difference in the learned
latent space is attributable to SAM3 mask-denoising alone. Then extracts paired
latent means and writes a side-by-side UMAP figure plus label-free structure
metrics.

    python scripts/vae/train_ava_ab.py --data outputs/ava/experiment_384_idx000 \
        --out outputs/ava/experiment_384_idx000/compare --epochs 100

Notes
-----
* Latents are extracted with a NON-shuffled loader so row i lines up across arms
  and with the HDF5 provenance (ann_id/file_name), i.e. the comparison is paired.
* Without call-type labels the metrics are unsupervised cluster-separability
  scores (higher silhouette / Calinski-Harabasz, lower Davies-Bouldin = cleaner
  structure). If you have call labels, join them on (file_name, ann_id) and swap
  in a supervised metric (kNN accuracy) for a stronger claim.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ava.models.vae import VAE
from ava.models.vae_dataset import SyllableDataset, get_syllable_partition
from ava.models.utils import numpy_to_tensor, _get_sylls_per_file

ARMS = ("raw", "masked")


class StableVAE(VAE):
    """Stock AVA VAE hardened against the mid-training NaN blow-up.

    Three numerics-only guards, none of which touch the ELBO or the network
    architecture (so the raw-vs-masked contrast stays apples-to-apples):

    * ``encode`` clamps the covariance-diagonal logit before ``exp`` so
      ``cov_diag`` can never overflow to ``inf`` (the usual ignition point).
    * ``train_epoch`` clips the gradient norm, capping any single oversized step.
    * ``train_epoch`` skips a batch whose loss is non-finite, so one bad step
      can't write NaNs into the weights and poison every later forward pass.
    """

    D_LOGIT_CLAMP = 10.0   # exp(10) ~ 2.2e4: keeps cov_diag finite, still ample
    GRAD_CLIP = 5.0

    def encode(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(self.bn1(x)))
        x = F.relu(self.conv2(self.bn2(x)))
        x = F.relu(self.conv3(self.bn3(x)))
        x = F.relu(self.conv4(self.bn4(x)))
        x = F.relu(self.conv5(self.bn5(x)))
        x = F.relu(self.conv6(self.bn6(x)))
        x = F.relu(self.conv7(self.bn7(x)))
        x = x.view(-1, 8192)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc41(F.relu(self.fc31(x)))
        u = self.fc42(F.relu(self.fc32(x))).unsqueeze(-1)  # rank-1 cov factor
        d_logit = self.fc43(F.relu(self.fc33(x)))
        d = torch.exp(torch.clamp(d_logit, -self.D_LOGIT_CLAMP, self.D_LOGIT_CLAMP))
        return mu, u, d

    def train_epoch(self, train_loader):
        self.train()
        train_loss = 0.0
        n_skipped = 0
        for data in train_loader:
            self.optimizer.zero_grad()
            data = data.to(self.device)
            loss = self.forward(data)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue  # grads already zeroed; leave weights untouched
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.GRAD_CLIP)
            self.optimizer.step()
        train_loss /= len(train_loader.dataset)
        msg = 'Epoch: {} Average loss: {:.4f}'.format(self.epoch, train_loss)
        if n_skipped:
            msg += '  (skipped {} non-finite batch(es))'.format(n_skipped)
        print(msg)
        self.epoch += 1
        return train_loss


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(hdf5_dir, batch_size, shuffle):
    """A single-file SyllableDataset loader over hdf5_dir."""
    partition = get_syllable_partition([str(hdf5_dir)], split=1, shuffle=False)
    spf = _get_sylls_per_file(partition)
    ds = SyllableDataset(filenames=partition["train"], sylls_per_file=spf,
                         transform=numpy_to_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def train_arm(arm, data_dir, out_dir, epochs, z_dim, batch_size, seed):
    """Train one VAE and return (latent_means[N,z], final_train_loss)."""
    seed_everything(seed)  # identical init + data order across arms
    arm_out = out_dir / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    model = StableVAE(save_dir=str(arm_out), z_dim=z_dim)

    train_loader = make_loader(data_dir / arm, batch_size, shuffle=True)
    loaders = {"train": train_loader, "test": train_loader}  # tiny set: test==train
    # test_freq=None -> never runs test; save pushed past the end to avoid side effects.
    # NOTE: AVA's train_loop guards the save_freq branch with `epoch > 0` but not the
    # vis_freq branch, so `epoch % vis_freq == 0` is always true at epoch 0 regardless
    # of vis_freq -- self.visualize() fires once after just the first training epoch.
    # We can't suppress that call, so we explicitly re-run visualize() below, after
    # training, to overwrite reconstruction.pdf with output from the fully trained model.
    model.train_loop(loaders, epochs=epochs, test_freq=None,
                     save_freq=epochs + 1, vis_freq=epochs + 1)
    model.save_state(str(arm_out / "checkpoint.tar"))

    eval_loader = make_loader(data_dir / arm, batch_size, shuffle=False)  # ordered!
    model.visualize(eval_loader)  # overwrite the stale epoch-0 reconstruction.pdf
    latent = model.get_latent(eval_loader)
    return latent, float(model.loss["train"].get(model.epoch - 1, np.nan))


def structure_metrics(z, k_range=range(2, 9)):
    """Unsupervised cluster-separability metrics over a sweep of k (best-of)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                  calinski_harabasz_score)
    zc = (z - z.mean(0)) / (z.std(0) + 1e-9)
    best = {"silhouette": -1.0, "davies_bouldin": np.inf, "calinski_harabasz": -1.0, "k": None}
    per_k = {}
    for k in k_range:
        if k >= len(zc):
            continue
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(zc)
        sil = float(silhouette_score(zc, lab))
        per_k[k] = sil
        if sil > best["silhouette"]:
            best.update(silhouette=sil, k=int(k),
                        davies_bouldin=float(davies_bouldin_score(zc, lab)),
                        calinski_harabasz=float(calinski_harabasz_score(zc, lab)))
    best["silhouette_per_k"] = per_k
    return best


def umap_figure(latents, out_png):
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, arm in zip(axes, ARMS):
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(latents[arm])
        ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.6, c="#4c72b0")
        ax.set_title(f"{arm}  (n={len(emb)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("AVA VAE latent space — SAM3 mask denoising A/B", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png)


def finalize(out):
    """CPU-only: combine whatever per-arm latents/metrics are on disk into the
    paired UMAP figure + metrics.json. Requires both arms' latent_{arm}.npy
    (and, if present, metrics_{arm}.json — recomputed from the latents otherwise)."""
    missing = [arm for arm in ARMS if not (out / f"latent_{arm}.npy").exists()]
    if missing:
        raise SystemExit(f"finalize: missing latent_{{arm}}.npy for {missing} in {out}")

    latents, metrics = {}, {}
    for arm in ARMS:
        z = np.load(out / f"latent_{arm}.npy")
        latents[arm] = z
        per_arm_file = out / f"metrics_{arm}.json"
        if per_arm_file.exists():
            metrics[arm] = json.loads(per_arm_file.read_text())
        else:
            print(f"(no metrics_{arm}.json found, recomputing structure metrics from latents)")
            metrics[arm] = structure_metrics(z)

    umap_figure(latents, out / "latent_umap_ab.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== structure metrics (higher silhouette / CH, lower DB = cleaner) ===")
    for arm in ARMS:
        m = metrics[arm]
        loss_str = f"  loss={m['final_train_loss']:.1f}" if "final_train_loss" in m else ""
        print(f"{arm:7s} silhouette={m['silhouette']:.3f} (k={m['k']})  "
              f"DB={m['davies_bouldin']:.3f}  CH={m['calinski_harabasz']:.1f}{loss_str}")
    print("wrote", out / "metrics.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dir containing raw/ and masked/ from coco_to_ava_specs.py")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: {data}/compare)")
    ap.add_argument("--arm", choices=("raw", "masked", "both", "compare"), default="both",
                     help="'raw'/'masked' trains one arm only (submit both as separate, "
                          "parallel jobs to halve wall-clock time); 'both' trains "
                          "sequentially in one job (original behavior); 'compare' only "
                          "combines existing latent_{arm}.npy into the UMAP figure + "
                          "metrics.json (CPU-only, run after both arms finish)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out or (args.data / "compare")
    out.mkdir(parents=True, exist_ok=True)

    if args.arm == "compare":
        finalize(out)
        return

    arms = ARMS if args.arm == "both" else (args.arm,)
    for arm in arms:
        print(f"\n===== training arm: {arm} =====")
        z, final_loss = train_arm(arm, args.data, out, args.epochs, args.z_dim, args.batch_size, args.seed)
        np.save(out / f"latent_{arm}.npy", z)
        m = {"final_train_loss": final_loss, **structure_metrics(z)}
        (out / f"metrics_{arm}.json").write_text(json.dumps(m, indent=2))
        print(f"wrote latent_{arm}.npy, metrics_{arm}.json")

    if args.arm == "both":
        finalize(out)


if __name__ == "__main__":
    main()
