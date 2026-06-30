"""
Visualize and compare evaluation metrics across detection methods.

Usage:
    python scripts/compare_methods.py <eval_base> [--methods ridge,sam3,squeakout] [--out fig.png]
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("eval_base")
parser.add_argument("--methods", default="ridge,sam3,squeakout")
parser.add_argument("--out",     default="outputs/eval/comparison.png")
args = parser.parse_args()

methods   = [m.strip() for m in args.methods.split(",")]
eval_base = Path(args.eval_base)


def load_metrics(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "session":  row["session"],
                "channel":  row["channel"],
                "n_gt":     int(row["n_gt"]),
                "tp":       int(row["tp"]),
                # pred_tp = predictions overlapping >=1 GT (precision numerator);
                # falls back to tp for older metrics.csv without the column.
                "pred_tp":  int(row.get("pred_tp", row["tp"])),
                "fp":       int(row["fp"]),
                "fn":       int(row["fn"]),
            })
    return rows


def micro_stats(rows):
    tp      = sum(r["tp"] for r in rows)        # GT events detected (recall)
    pred_tp = sum(r["pred_tp"] for r in rows)   # correct predictions (precision)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return rec, prec, f1


def session_stats(rows, sessions):
    """Return per-session (rec, prec, f1) arrays."""
    recs, precs, f1s = [], [], []
    for sess in sessions:
        sr = [r for r in rows if r["session"] == sess]
        rec, prec, f1 = micro_stats(sr)
        recs.append(rec); precs.append(prec); f1s.append(f1)
    return np.array(recs), np.array(precs), np.array(f1s)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


data = {}
for method in methods:
    csv_path = eval_base / method / "metrics.csv"
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found — skipping {method}")
        continue
    data[method] = load_metrics(csv_path)

if not data:
    raise SystemExit("No metrics found.")

channels = sorted({r["channel"] for rows in data.values() for r in rows})
sessions = sorted({r["session"] for rows in data.values() for r in rows})
method_names = list(data.keys())
n_methods = len(method_names)

# ── layout ────────────────────────────────────────────────────────────────────
# Single row: grouped bar — overall + per channel (Recall / Precision / F1)

n_groups  = 1 + len(channels)   # overall + each channel

fig = plt.figure(figsize=(14, 5))
gs  = fig.add_gridspec(1, n_groups, wspace=0.35)

colors = plt.cm.tab10(np.linspace(0, 0.3, n_methods))
metric_labels = ["Recall", "Precision", "F1"]
bar_width = 0.22
offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

# ── Row 0: bar charts ─────────────────────────────────────────────────────────

def bar_panel(ax, rows_by_method, title):
    x = np.arange(3)  # Recall, Precision, F1
    for k, (method, rows) in enumerate(rows_by_method.items()):
        rec, prec, f1 = micro_stats(rows)
        vals = np.array([rec, prec, f1])
        recs, precs, f1s = session_stats(rows, sessions)
        errs = np.array([np.nanstd(recs), np.nanstd(precs), np.nanstd(f1s)])
        bars = ax.bar(x + offsets[k], vals, bar_width,
                      label=method, color=colors[k], alpha=0.85,
                      yerr=errs, capsize=3, error_kw={"elinewidth": 1, "ecolor": "black"})
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    despine(ax)


# Overall
ax_overall = fig.add_subplot(gs[0, 0])
bar_panel(ax_overall, data, "Overall")

# Per channel
for ci, ch in enumerate(channels):
    ax = fig.add_subplot(gs[0, ci + 1])
    ch_data = {m: [r for r in rows if r["channel"] == ch]
               for m, rows in data.items()}
    bar_panel(ax, ch_data, f"Channel {ch}")

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved → {out_path}")
