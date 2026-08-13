"""
Compare bout-level detection accuracy across methods.

Reads <eval_base>/<method>/bouts.csv (written by scripts/evaluation/bout_analysis.py) for
each method, bins bouts by n_calls, and plots Recall / Precision / merge_ratio
by bout size side by side, one bar group per method — so you can see whether a
method's accuracy holds up as calls get packed closer together within a bout,
and whether that holds up because calls are individuated (merge_ratio ~1) or
because several got fused into one predicted blob (merge_ratio << 1).

Usage:
    python scripts/evaluation/compare_bouts.py outputs/eval --methods sam3_best,ridge,squeakout
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("eval_base")
parser.add_argument("--methods", default="sam3_best,ridge,squeakout")
parser.add_argument("--bouts-csv", default="bouts.csv",
                    help="bouts CSV filename inside each method dir (default: bouts.csv)")
parser.add_argument("--bins", default="1,2,4,7",
                    help="bout-size bin edges, e.g. '1,2,4,7' -> [1],[2-3],[4-6],[7+]")
parser.add_argument("--out", default="outputs/eval/compare_bouts.png")
args = parser.parse_args()

methods   = [m.strip() for m in args.methods.split(",")]
eval_base = Path(args.eval_base)
bin_edges = sorted(int(x) for x in args.bins.split(","))


def bin_label(n):
    for lo, hi in zip(bin_edges, bin_edges[1:] + [None]):
        if hi is None or n < hi:
            return f"{lo}" if hi == lo + 1 else (f"{lo}+" if hi is None else f"{lo}-{hi - 1}")
    return f"{bin_edges[-1]}+"


def load_bouts(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({"n_calls": int(row["n_calls"]),
                         "tp": int(row["tp"]), "fn": int(row["fn"]),
                         "pred_tp": int(row["pred_tp"]), "fp": int(row["fp"]),
                         "n_pred_events": int(row.get("n_pred_events", 0))})
    return rows


def micro_by_bin(rows):
    """{bin_label: (recall, precision, n_bouts, merge_ratio)}"""
    bins = {}
    for r in rows:
        bins.setdefault(bin_label(r["n_calls"]), []).append(r)
    out = {}
    for lb, rs in bins.items():
        tp = sum(r["tp"] for r in rs); fn = sum(r["fn"] for r in rs)
        pred_tp = sum(r["pred_tp"] for r in rs); fp = sum(r["fp"] for r in rs)
        n_calls = sum(r["n_calls"] for r in rs); n_pred_events = sum(r["n_pred_events"] for r in rs)
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) > 0 else float("nan")
        merge_ratio = n_pred_events / n_calls if n_calls > 0 else float("nan")
        out[lb] = (rec, prec, len(rs), merge_ratio)
    return out


data = {}
for method in methods:
    path = eval_base / method / args.bouts_csv
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {method}")
        continue
    data[method] = micro_by_bin(load_bouts(path))

if not data:
    raise SystemExit("No bouts CSVs found.")

all_bins = sorted({lb for m in data.values() for lb in m},
                  key=lambda lb: int(lb.rstrip("+").split("-")[0]))
method_names = list(data.keys())
n_methods = len(method_names)

fig, (ax_rec, ax_prec, ax_merge) = plt.subplots(1, 3, figsize=(17, 5))
colors = plt.cm.tab10(np.linspace(0, 0.3, n_methods))
x = np.arange(len(all_bins))
bar_width = 0.8 / n_methods
offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width
NAN4 = (float("nan"), float("nan"), 0, float("nan"))


def panel(ax, metric_idx, title, ylabel, ylim, ref_line=None):
    for k, method in enumerate(method_names):
        vals = np.array([data[method].get(lb, NAN4)[metric_idx] for lb in all_bins])
        bars = ax.bar(x + offsets[k], vals, bar_width, label=method, color=colors[k], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)
    if ref_line is not None:
        ax.axhline(ref_line, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ns = [next((data[m][lb][2] for m in method_names if lb in data[m]), 0) for lb in all_bins]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\n(n={n})" for lb, n in zip(all_bins, ns)])
    ax.set_xlabel("calls / bout")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)


panel(ax_rec,   0, "Recall",     "Recall",    (0, 1.15))
panel(ax_prec,  1, "Precision",  "Precision", (0, 1.15))
panel(ax_merge, 3, "Merge ratio (1.0 = one-to-one)", "Predicted events / GT calls",
     (0, max(1.3, max(v[3] for m in data.values() for v in m.values() if v[3] == v[3]) + 0.1)),
     ref_line=1.0)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved → {out_path}")

print("\nSummary:")
for method in method_names:
    parts = "  ".join(f"{lb}:R={data[method][lb][0]:.2f}/P={data[method][lb][1]:.2f}"
                      for lb in all_bins if lb in data[method])
    print(f"  {method:12s}: {parts}")
