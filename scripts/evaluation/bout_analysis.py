"""
Bout-level detection accuracy.

A bout is a run of GT calls whose inter-call intervals (gap from one call's end
to the next call's start) are all <= --gap-threshold; a bout of size 1 is an
isolated call. Bouts are derived from GT only (vox_tracer.scoring.group_into_bouts).
For each bout, predictions overlapping its time span are matched against its
calls with the same event-level logic as scripts/evaluate.py
(vox_tracer.scoring.score_combined), giving a per-bout recall/precision/F1.

Also reports merge_ratio = n_pred_events / n_calls per bout (n_pred_events is
the count of merged predicted events overlapping the bout). Boolean event-overlap
recall can look perfect even when several calls were fused into one predicted
blob that happens to touch all of them; merge_ratio << 1 is exactly that failure
mode, so it's the check on whether recall gains in dense bouts are real
call-by-call detection or an artifact of under-segmentation.

Outputs
-------
  bouts.csv    — one row per bout: session, bout_id, n_calls, start, stop,
                 duration, mean_ici, tp, fn, pred_tp, fp, recall, precision, f1,
                 n_pred_events, merge_ratio
  bouts.png    — recall/precision/merge_ratio by bout-size bin (only with --plot)

Usage
-----
    python scripts/evaluation/bout_analysis.py data outputs/sam3_best \
        --gap-threshold 0.25 \
        --out outputs/eval/sam3_best/bouts.csv \
        --plot outputs/eval/sam3_best/bouts.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_tracer.scoring import load_gt_from_csv, read_pred_boxes, score_bouts

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("recording_base", help="dir mirroring experiment_*/idx_*, each with *annotations_gt.csv")
parser.add_argument("pred_dir", help="dir holding experiment_*/idx_* prediction trees (coco_ch_<ch>.json)")
parser.add_argument("--channels", default="118,35")
parser.add_argument("--iou-threshold", type=float, default=0.0)
parser.add_argument("--gap-threshold", type=float, default=0.25,
                    help="max inter-call gap (s) within a bout (default: 0.25)")
parser.add_argument("--bins", default="1,2,4,7",
                    help="bout-size bin edges, e.g. '1,2,4,7' -> [1],[2-3],[4-6],[7+]")
parser.add_argument("--out", default="outputs/eval/bouts.csv")
parser.add_argument("--plot", default=None, help="if set, write a bout-size-vs-accuracy figure here")
args = parser.parse_args()

channels      = [int(c) for c in args.channels.split(",")]
bin_edges     = sorted(int(x) for x in args.bins.split(","))

recording_base = Path(args.recording_base)
pred_base      = Path(args.pred_dir)
rows = []
for rec_dir in sorted(recording_base.glob("experiment_*/idx_*")):
    rel = Path(rec_dir.parent.name) / rec_dir.name
    session = str(rel)
    try:
        gt, _combined = load_gt_from_csv(rec_dir)
    except FileNotFoundError as e:
        print(f"  {session}: {e}, skipping")
        continue
    boxes = read_pred_boxes(pred_base / rel, channels)
    bouts = score_bouts(gt, boxes, args.gap_threshold, args.iou_threshold)
    for i, b in enumerate(bouts):
        rows.append({"session": session, "bout_id": i, **b})

if not rows:
    raise SystemExit("No recordings with GT found.")

print(f"{len(rows)} bouts across {len({r['session'] for r in rows})} recordings "
      f"(gap-threshold={args.gap_threshold}s)")

# ── write per-bout CSV ──────────────────────────────────────────────────────────
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fields = ["session", "bout_id", "n_calls", "start", "stop", "duration", "mean_ici",
          "tp", "fn", "pred_tp", "fp", "recall", "precision", "f1",
          "n_pred_events", "merge_ratio"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in rows)
print(f"bouts → {out_path}")


# ── aggregate by bout-size bin (micro-averaged recall/precision) ───────────────
def bin_label(n):
    for lo, hi in zip(bin_edges, bin_edges[1:] + [None]):
        if hi is None or n < hi:
            return f"{lo}" if hi == lo + 1 else (f"{lo}+" if hi is None else f"{lo}-{hi - 1}")
    return f"{bin_edges[-1]}+"


def micro(rows_):
    tp = sum(r["tp"] for r in rows_); fn = sum(r["fn"] for r in rows_)
    pred_tp = sum(r["pred_tp"] for r in rows_); fp = sum(r["fp"] for r in rows_)
    n_calls = sum(r["n_calls"] for r in rows_); n_pred_events = sum(r["n_pred_events"] for r in rows_)
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else float("nan")
    merge_ratio = n_pred_events / n_calls if n_calls > 0 else float("nan")
    return rec, prec, f1, merge_ratio


bins = {}
for r in rows:
    bins.setdefault(bin_label(r["n_calls"]), []).append(r)
bin_order = sorted(bins, key=lambda lb: int(lb.rstrip("+").split("-")[0]))

print("\nBy bout size (calls/bout):")
summary = []
for lb in bin_order:
    rs = bins[lb]
    rec, prec, f1, merge_ratio = micro(rs)
    summary.append((lb, rec, prec, f1, len(rs), merge_ratio))
    print(f"  {lb:>6s} calls  n_bouts={len(rs):4d}  Recall={rec:.3f}  Precision={prec:.3f}  "
          f"F1={f1:.3f}  merge_ratio={merge_ratio:.3f}")

# ── optional plot ────────────────────────────────────────────────────────────────
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax_merge) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(summary))
    width = 0.35
    colors = plt.cm.tab10(np.linspace(0, 0.3, 2))
    recs  = [s[1] for s in summary]
    precs = [s[2] for s in summary]
    ax.bar(x - width / 2, recs,  width, label="Recall",    color=colors[0], alpha=0.85)
    ax.bar(x + width / 2, precs, width, label="Precision", color=colors[1], alpha=0.85)
    for xi, r, p in zip(x, recs, precs):
        if r == r:
            ax.text(xi - width / 2, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=7)
        if p == p:
            ax.text(xi + width / 2, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[0]}\n(n={s[4]})" for s in summary])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Detection accuracy by bout size (gap>{args.gap_threshold}s)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    merges = [s[5] for s in summary]
    bars = ax_merge.bar(x, merges, width=0.5, color=colors[0], alpha=0.85)
    ax_merge.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    for bar, m in zip(bars, merges):
        if m == m:
            ax_merge.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.2f}",
                          ha="center", va="bottom", fontsize=7)
    ax_merge.set_xticks(x)
    ax_merge.set_xticklabels([f"{s[0]}\n(n={s[4]})" for s in summary])
    ax_merge.set_ylabel("Predicted events / GT calls")
    ax_merge.set_title("Merge ratio by bout size (1.0 = one-to-one)", fontsize=10)
    ax_merge.spines["top"].set_visible(False)
    ax_merge.spines["right"].set_visible(False)
    ax_merge.grid(axis="y", alpha=0.3)

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"plot  → {plot_path}")
