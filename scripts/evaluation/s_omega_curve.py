"""S_omega against omega for every rule of a label campaign, with anytime-valid bands.

omega = 0 is relative recall, 0.5 is F1, 1 is precision (main.pdf eq. 6-8). Bands are
the single-rule confidence sequences of vox_label/ranking.py, pointwise in omega;
--simultaneous splits alpha over the omega grid instead.

Read-only: it never binds the campaign, so it does not freeze omega.

    python scripts/evaluation/s_omega_curve.py outputs/label_campaigns/gerbil_ssl_k4
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_label.campaign import Campaign
from vox_label.ranking import corpus_stats, rule_cs

# Reference categorical palette, fixed order (dataviz skill, light mode).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]


def curves(campaign, omegas, alpha, a_star):
    ann = campaign.annotations()
    x = np.array([a["label"] for a in ann], float)
    fns = {r: campaign.rule_fn(r) for r in campaign.rules}
    a_r = corpus_stats(list(campaign.pool.values()), campaign.rules, fns)["a_r"]
    out = {}
    for r in campaign.rules:
        dec = np.array([bool(fns[r](a["z"])) for a in ann], float)
        rows = [rule_cs(x, dec, w, alpha, a_r[r], a_star) for w in omegas]
        out[r] = {k: np.array([np.nan if row[k] is None else row[k] for row in rows])
                  for k in ("estimate", "lo", "hi")}
    return out, len(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--out", default=None, help="default <campaign>/s_omega_curve.png")
    ap.add_argument("--n-omega", type=int, default=41)
    ap.add_argument("--alpha", type=float, default=None,
                    help="default: the campaign's alpha")
    ap.add_argument("--simultaneous", action="store_true",
                    help="alpha / n_omega per point, so the bands hold jointly on the grid")
    ap.add_argument("--a-star", type=float, default=0.5,
                    help="prevalence guess for the prior variance (eq. 22)")
    ap.add_argument("--rules", nargs="*", default=None)
    args = ap.parse_args()

    c = Campaign(args.campaign)
    if args.rules:
        c.rules = args.rules
    omegas = np.linspace(0, 1, args.n_omega)
    alpha = args.alpha or c.alpha
    per_point = alpha / len(omegas) if args.simultaneous else alpha
    data, n = curves(c, omegas, per_point, args.a_star)

    # Legend in order of F1, so it reads top to bottom like the lines at the middle.
    mid = len(omegas) // 2
    order = sorted(data, key=lambda r: -np.nan_to_num(data[r]["estimate"][mid]))
    color = {r: COLORS[i % len(COLORS)] for i, r in enumerate(c.rules)}

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    for r in order:
        d = data[r]
        ax.fill_between(omegas, d["lo"], d["hi"], color=color[r], alpha=0.12, lw=0)
        ax.plot(omegas, d["estimate"], color=color[r], lw=2, label=r)
    for w, name in ((0, "recall"), (0.5, "F1"), (1, "precision")):
        ax.axvline(w, color="#d9d8d3", lw=1, zorder=0)
        ax.text(w, 1.01, name, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=9, color="#52514e")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\omega$  (weight on false positives)")
    ax.set_ylabel(r"$S_\omega$")
    band = f"{1 - alpha:.0%} {'simultaneous' if args.simultaneous else 'pointwise'} CS"
    ax.set_title(f"n = {n} annotations, bands: {band}", fontsize=10, color="#52514e",
                 loc="left", pad=16)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    out = Path(args.out or Path(args.campaign) / "s_omega_curve.png")
    fig.savefig(out, dpi=160, facecolor="#ffffff")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
