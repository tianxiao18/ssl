"""
Compare per-event spectral-feature distributions across detectors, split by
whether each detection is a true or false positive.

Reads outputs/spectral_features/{sam3_best,ridge_filtered,squeakout_filtered}.csv
(written by scripts/spectral_features.py, which tags every detection tp/fp by
overlap with the recording's GT vox events). Renders a method x feature grid:
one row per detector, one column per feature, each cell overlaying the TP and FP
density curves. Density (area-normalised) makes the two classes comparable
despite very different counts, so the question the plot answers is *where in
feature space do the false positives sit* relative to the true detections.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

METHODS = ["sam3_best", "ridge_filtered", "squeakout_filtered"]

# TP/FP is a state (correct vs incorrect), so it wears the reserved status
# colours — good vs critical — never the categorical series hues. Each ships
# with a text label in the legend, so identity is never colour-alone.
CLASSES = {"tp": ("True positive", "#0ca30c"), "fp": ("False positive", "#d03b3b")}

# (column, panel title, unit label, scale factor applied to the raw column)
PANELS = [
    ("centroid_mean",  "Spectral centroid",  "kHz", 1e-3),
    ("bandwidth_mean", "Spectral bandwidth", "kHz", 1e-3),
    ("f_band_hz",      "Detection band",     "kHz", 1e-3),
    ("flatness_mean",  "Spectral flatness",  "",    1.0),
    ("zcr_mean",       "Zero-crossing rate", "",    1.0),
    ("duration_sec",   "Event duration",     "ms",  1e3),
]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e2"
KDE_CAP = 40000


def kde_curve(values, grid, cap=KDE_CAP):
    """Gaussian-KDE density of `values` on `grid`, subsampled for speed."""
    v = values[np.isfinite(values)]
    if len(v) < 5 or np.ptp(v) == 0:
        return None
    if len(v) > cap:
        v = np.random.default_rng(0).choice(v, cap, replace=False)
    return gaussian_kde(v)(grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=Path("outputs/spectral_features"))
    ap.add_argument("--out", type=Path, default=Path("outputs/spectral_features/feature_distributions_tpfp.png"))
    args = ap.parse_args()

    data = {m: pd.read_csv(args.in_dir / f"{m}.csv") for m in METHODS}

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    nrow, ncol = len(METHODS), len(PANELS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.5 * nrow),
                             constrained_layout=True, squeeze=False)

    # Column x-ranges are shared down each column (pooled 1-99th pct over all
    # methods & classes) so a feature reads on one scale across every row.
    col_range = {}
    for col, _, _, scale in PANELS:
        pooled = np.concatenate([data[m][col].dropna().values for m in METHODS]) * scale
        col_range[col] = np.percentile(pooled, [1, 99])

    for r, m in enumerate(METHODS):
        df = data[m]
        n_tp = int((df["label"] == "tp").sum())
        n_fp = int((df["label"] == "fp").sum())
        for c, (col, title, unit, scale) in enumerate(PANELS):
            ax = axes[r][c]
            lo, hi = col_range[col]
            grid = np.linspace(lo, hi, 400)
            for cls, (_, color) in CLASSES.items():
                vals = df.loc[df["label"] == cls, col].dropna().values * scale
                dens = kde_curve(vals, grid)
                if dens is not None:
                    ax.plot(grid, dens, color=color, lw=1.8, solid_capstyle="round")
                    ax.fill_between(grid, dens, color=color, alpha=0.10)
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if r == 0:
                ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
            if r == nrow - 1 and unit:
                ax.set_xlabel(unit, color=MUTED, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{m}\nTP {n_tp:,} · FP {n_fp:,}",
                              fontsize=10, fontweight="bold", rotation=90,
                              labelpad=10, va="center")

    handles = [plt.Line2D([], [], color=color, lw=2.5, label=name)
               for name, color in CLASSES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.05), fontsize=11)
    fig.suptitle("Spectral features of true vs false positives, by detector "
                 "(band-restricted, density-normalised)",
                 fontsize=13, fontweight="bold", y=1.09)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    for m in METHODS:
        vc = data[m]["label"].value_counts(dropna=False).to_dict()
        print(f"  {m}: {vc}")


if __name__ == "__main__":
    main()
