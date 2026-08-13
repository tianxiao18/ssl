"""Train a QMC latent-variable model (QMCLVM: decoder-only, RQMC-lattice inference,
no amortized encoder) on the raw/masked AVA spectrogram data, and compare it against
the already-trained AVA VAE checkpoints from train_ava_ab.py.

Consumes the same hdf5 layout as train_ava_ab.py (``{data}/raw`` and ``{data}/masked``,
each holding ``syllables_*.hdf5`` files with a ``specs`` dataset of shape
``(n_specs, 128, 128)``) via qmc_deep_gen's ``bird_data`` loader.

The QMC-LVM has no encoder: training maximizes a quasi-Monte Carlo estimate of the log
evidence log p(x), integrating the decoder over a randomly-shifted Fibonacci lattice on
a 2D torus latent space (see qmc_deep_gen/models/qmc_base.py). This is compared against
the 32-dim AVA VAE's ELBO on the same data.

    python scripts/vae/train_qmc_ab.py --data /mnt/home/the10/ceph/ava/all_recordings \
        --vae-dir outputs/compare --out outputs/compare_qmc --epochs 30
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
QMC_DIR = SCRIPTS_DIR.parents[1] / "qmc_deep_gen"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(QMC_DIR))

from train_ava_ab import StableVAE, seed_everything  # noqa: E402  (existing 32-dim AVA VAE)

from data.bird_data import bird_data  # noqa: E402
from models.qmc_base import QMCLVM  # noqa: E402
from models.sampling import gen_fib_basis  # noqa: E402
from models.utils import get_decoder_arch  # noqa: E402
from train.losses import gaussian_evidence, gaussian_lp  # noqa: E402
import train.train as train_qmc  # noqa: E402

ARMS = ("raw", "masked")


def make_dataset(hdf5_dir, specs_per_file=128):
    """A bird_data Dataset over every syllables_*.hdf5 file in hdf5_dir."""
    files = sorted(glob.glob(str(hdf5_dir / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"no hdf5 files in {hdf5_dir}")
    file_ids = list(range(len(files)))  # one id per file; unused (non-conditional training)
    return bird_data(files, file_ids, specs_per_file=specs_per_file,
                      transform=lambda x: torch.from_numpy(x).to(torch.float32).unsqueeze(0))


def train_qmc_arm(arm, data_dir, arm_out, epochs, batch_size, latent_dim, var,
                   train_grid_m, device, seed):
    seed_everything(seed)
    loader = torch.utils.data.DataLoader(make_dataset(data_dir / arm), batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    decoder = get_decoder_arch(dataset_name="gerbil_ava", latent_dim=latent_dim, arch="qmc")
    model = QMCLVM(latent_dim=latent_dim, device=device, decoder=decoder)
    lattice = gen_fib_basis(m=train_grid_m).to(device)
    loss_fn = lambda samples, data: gaussian_evidence(samples, data, var=var)

    model, opt, losses = train_qmc.train_loop(model, loader, lattice, loss_fn,
                                               nEpochs=epochs, verbose=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "losses": losses},
               arm_out / "qmc_checkpoint.tar")
    return model, losses


def load_vae_arm(arm, vae_dir, z_dim, device):
    model = StableVAE(save_dir=str(vae_dir / arm), z_dim=z_dim, device_name=device)
    model.load_state(str(vae_dir / arm / "checkpoint.tar"))
    model.eval()
    return model


def training_curve(losses, batches_per_epoch, out_path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    epoch_losses = -np.array(losses).reshape(-1, batches_per_epoch).mean(axis=1)
    ax.plot(epoch_losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("QMC-LVM log evidence (per sample)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compare_reconstructions(arm, qmc_model, vae_model, data_dir, arm_out, test_grid_m,
                            var, n_samples, device, seed):
    ds = make_dataset(data_dir / arm)
    idxs = np.random.RandomState(seed).choice(len(ds), n_samples, replace=False)
    samples = torch.stack([ds[int(i)][0] for i in idxs]).to(device)  # N x 1 x 128 x 128

    lp_fn = lambda s, d: gaussian_lp(s, d, var)
    test_lattice = gen_fib_basis(m=test_grid_m).to(device)

    qmc_model.eval()
    with torch.no_grad():
        qmc_recon = qmc_model.round_trip(test_lattice, samples, lp_fn, recon_type="posterior")
        mu, _, _ = vae_model.encode(samples.squeeze(1))
        vae_recon = vae_model.decode(mu).view(-1, 1, 128, 128)

    fig, axs = plt.subplots(3, n_samples, figsize=(2.2 * n_samples, 6.6))
    row_labels = ["input", "QMC-LVM", "AVA VAE"]
    rows = [samples, qmc_recon, vae_recon]
    for r, (row, label) in enumerate(zip(rows, row_labels)):
        for i in range(n_samples):
            axs[r, i].imshow(row[i, 0].detach().cpu(), origin="lower", cmap="viridis")
            axs[r, i].set_xticks([])
            axs[r, i].set_yticks([])
        axs[r, 0].set_ylabel(label, fontsize=9)
    fig.tight_layout()
    out_path = arm_out / "qmc_vs_vae_reconstructions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


def evidence_comparison(arm, qmc_model, vae_model, data_dir, var, eval_batch_size, device, seed):
    """Per-sample log-evidence estimates: QMC-LVM's lattice quadrature vs the VAE's
    single-sample ELBO (a lower bound), both approximating log p(x) for the same data."""
    ds = make_dataset(data_dir / arm)
    idxs = np.random.RandomState(seed + 1).choice(len(ds), min(eval_batch_size, len(ds)), replace=False)
    data = torch.stack([ds[int(i)][0] for i in idxs]).to(device)

    # Reuse the training lattice (not a finer test grid): a fine grid x large eval batch
    # blows up gaussian_lp's B x K x 1 x 128 x 128 intermediate (vmap over both axes).
    test_lattice = gen_fib_basis(m=15).to(device)
    qmc_model.eval()
    with torch.no_grad():
        qmc_samples = qmc_model.forward(test_lattice, random=True, mod=True)
        qmc_loss = gaussian_evidence(qmc_samples, data, var=var, batch_size=200)  # chunk over grid points
        qmc_log_evidence_per_sample = -qmc_loss.item()

        vae_loss = vae_model.forward(data.squeeze(1))  # sum over batch (-ELBO)
        vae_elbo_per_sample = -vae_loss.item() / data.shape[0]

    return {"qmc_log_evidence_per_sample": qmc_log_evidence_per_sample,
            "vae_elbo_per_sample": vae_elbo_per_sample}


def run_arm(arm, args, device):
    """Train one arm's QMC-LVM and compare it against that arm's AVA VAE checkpoint.
    Writes qmc_checkpoint.tar, qmc_train_curve.png, qmc_vs_vae_reconstructions.png, and
    metrics_{arm}.json under out/{arm}/ -- everything needed to run this arm as its own
    (GPU) job, in parallel with the other arm."""
    arm_out = args.out / arm
    arm_out.mkdir(parents=True, exist_ok=True)

    qmc_model, losses = train_qmc_arm(arm, args.data, arm_out, args.epochs, args.batch_size,
                                      args.latent_dim, args.var, args.train_grid_m, device, args.seed)
    n_files = len(glob.glob(str(args.data / arm / "*.hdf5")))
    n_specs = n_files * 128
    batches_per_epoch = len(losses) // args.epochs
    training_curve(losses, batches_per_epoch, arm_out / "qmc_train_curve.png")

    vae_model = load_vae_arm(arm, args.vae_dir, args.vae_z_dim, device)

    compare_reconstructions(arm, qmc_model, vae_model, args.data, arm_out,
                            test_grid_m=20, var=args.var, n_samples=args.n_recon_samples,
                            device=device, seed=args.seed)

    ev = evidence_comparison(arm, qmc_model, vae_model, args.data, args.var,
                             args.eval_batch_size, device, args.seed)
    ev["n_specs"] = n_specs
    ev["final_qmc_train_loss"] = float(np.mean(losses[-batches_per_epoch:]))
    (arm_out / "metrics.json").write_text(json.dumps(ev, indent=2))
    print(f"{arm}: {ev}")
    print("wrote", arm_out / "metrics.json")
    return ev


def finalize(out):
    """CPU-only: merge both arms' out/{arm}/metrics.json into a single combined
    out/metrics.json. Run after both per-arm jobs finish."""
    missing = [arm for arm in ARMS if not (out / arm / "metrics.json").exists()]
    if missing:
        raise SystemExit(f"finalize: missing {{arm}}/metrics.json for {missing} in {out}")
    results = {arm: json.loads((out / arm / "metrics.json").read_text()) for arm in ARMS}
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    for arm in ARMS:
        print(f"{arm}: {results[arm]}")
    print("wrote", out / "metrics.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                     help="dir containing raw/ and masked/ AVA hdf5 syllable files")
    ap.add_argument("--vae-dir", type=Path, default=Path("outputs/compare"),
                     help="dir with {arm}/checkpoint.tar from train_ava_ab.py")
    ap.add_argument("--out", type=Path, default=Path("outputs/compare_qmc"))
    ap.add_argument("--arm", choices=("raw", "masked", "both", "compare"), default="both",
                     help="'raw'/'masked' trains+compares one arm only (submit both as "
                          "separate, parallel GPU jobs to halve wall-clock time); 'both' "
                          "runs both sequentially in one job (original behavior); "
                          "'compare' only merges existing {arm}/metrics.json into a "
                          "combined metrics.json (CPU-only, run after both arms finish)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--vae-z-dim", type=int, default=32)
    ap.add_argument("--var", type=float, default=0.1,
                     help="observation noise variance; 0.1 matches the AVA VAE's "
                          "default model_precision=10.0 (var = 1/precision)")
    ap.add_argument("--train-grid-m", type=int, default=15, help="Fibonacci lattice index for training")
    ap.add_argument("--n-recon-samples", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.arm == "compare":
        finalize(args.out)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    arms = ARMS if args.arm == "both" else (args.arm,)
    for arm in arms:
        print(f"\n===== QMC-LVM: training arm {arm} =====")
        run_arm(arm, args, device)

    if args.arm == "both":
        finalize(args.out)


if __name__ == "__main__":
    main()
