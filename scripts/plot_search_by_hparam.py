"""Plot the random-search PR cloud, one panel per hyperparameter, colored by
that hyperparameter's value — so you can see where each setting lives in PR space.

Usage
-----
    python scripts/plot_search_by_hparam.py outputs/random_search/random_search_results.csv \
        [outputs/random_search/random_search_by_hparam.png]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HPARAMS = ["threshold_pct", "freq_min", "vert_aspect",
           "min_freq_sweep_frac", "min_mask_cols", "score_threshold"]


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("outputs/random_search/random_search_results.csv")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        csv_path.with_name("random_search_by_hparam.png")

    rows = list(csv.DictReader(open(csv_path)))
    rec = [float(r["sam3_recall"]) for r in rows]
    prec = [float(r["sam3_precision"]) for r in rows]
    # best = max f1 (matches best_config.json selection)
    best = max(rows, key=lambda r: float(r["sam3_f1"]))
    bx, by = float(best["sam3_recall"]), float(best["sam3_precision"])

    ncol = 3
    nrow = (len(HPARAMS) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow),
                             squeeze=False)

    for ax, hp in zip(axes.flat, HPARAMS):
        vals = [float(r[hp]) for r in rows]
        sc = ax.scatter(rec, prec, c=vals, cmap="viridis", s=22, alpha=0.85)
        ax.plot(bx, by, "*", color="red", markersize=18,
                markeredgecolor="k", markeredgewidth=0.6, zorder=5)
        fig.colorbar(sc, ax=ax, label=hp)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(hp)
        ax.grid(alpha=0.3)

    # hide any unused panels
    for ax in axes.flat[len(HPARAMS):]:
        ax.set_visible(False)

    fig.suptitle(f"Random search: {len(rows)} configs, colored by hyperparameter "
                 f"(★ = best f1)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
