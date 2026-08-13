"""
Precision-recall curves for detection methods on contiguous ground truth.

Each method is run ONCE (at a permissive setting) and exports a per-detection
``score`` in its COCO annotations. This script traces a PR curve by sweeping a
score threshold post-hoc: at each threshold it drops sub-threshold detections,
then scores with the SAME combined-channel, event-level logic as
scripts/evaluate.py (vox_tracer.scoring.score_combined) — so the curves and the
scripts/compare_methods.py bar chart are measured identically.

Aggregation is micro (pool TP/FP/FN across all recordings). A method whose COCO
carries no ``score`` can only contribute a single operating point (all its
detections); the plot marks it and warns — export a score to get a full curve.

Usage
-----
    python scripts/evaluation/pr_curves.py outputs \
        --methods ridge_pr,sam3_pr,squeakout_pr \
        --gt-csv data/sampled_contiguous_detections.csv \
        --out outputs/eval/pr_curves.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_tracer.scoring import load_gt_map, read_pred_boxes, score_combined

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pred_base", help="dir holding <method>/experiment_*/idx_* prediction trees")
parser.add_argument("--methods", default="ridge,sam3_best,squeakout")
parser.add_argument("--gt-csv", required=True,
                    help="contiguous detections CSV (defines GT events + trustworthy coverage)")
parser.add_argument("--channels", default="118,35")
parser.add_argument("--iou-threshold", type=float, default=0.0)
parser.add_argument("--n-thresholds", type=int, default=40,
                    help="number of score cut-points sampled from the score distribution")
parser.add_argument("--recall-targets", default="0.7,0.8,0.9",
                    help="print precision at these recall levels (interpolated)")
parser.add_argument("--out", default="outputs/eval/pr_curves.png")
args = parser.parse_args()

channels      = [int(c) for c in args.channels.split(",")]
methods       = [m.strip() for m in args.methods.split(",")]
iou_threshold = args.iou_threshold
rec_targets   = [float(x) for x in args.recall_targets.split(",")]

gt_by_rec, coverage_by_rec = load_gt_map(args.gt_csv)
print(f"GT: {sum(len(v) for v in gt_by_rec.values())} events across {len(gt_by_rec)} recordings; "
      f"{len(coverage_by_rec)} with labeled coverage")


def load_method(method):
    """Return [(gt_intervals, pred_boxes)] over covered recordings + all seen scores."""
    base = Path(args.pred_base) / method
    recs, scores = [], []
    for pdir in sorted(base.glob("experiment_*/idx_*")):
        session = f"{pdir.parent.name}/{pdir.name}"
        coverage = coverage_by_rec.get(session)
        if coverage is None:                      # no trustworthy coverage → skip (as evaluate.py)
            continue
        boxes = read_pred_boxes(pdir, channels, coverage)
        recs.append((gt_by_rec.get(session, []), boxes))
        scores.extend(s for _, _, s in boxes if s is not None)
    return recs, scores


def micro_point(recs, score_threshold):
    """Aggregate score_combined over recordings → one (recall, precision, f1, counts)."""
    tp = fn = pred_tp = fp = 0
    for gt, boxes in recs:
        r = score_combined(gt, boxes, iou_threshold, score_threshold)
        tp += r["tp"]; fn += r["fn"]; pred_tp += r["pred_tp"]; fp += r["fp"]
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) > 0 else 1.0  # no preds → precision 1 (limit)
    f1   = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else float("nan")
    return {"threshold": score_threshold, "recall": rec, "precision": prec, "f1": f1,
            "tp": tp, "fn": fn, "pred_tp": pred_tp, "fp": fp}


def auc_pr(pts):
    """Area under the PR curve via trapezoid over recall (points sorted by recall)."""
    p = sorted(pts, key=lambda d: d["recall"])
    r = np.array([d["recall"] for d in p]); pr = np.array([d["precision"] for d in p])
    m = ~(np.isnan(r) | np.isnan(pr))
    trap = getattr(np, "trapezoid", np.trapz)   # np.trapz deprecated in newer numpy
    return float(trap(pr[m], r[m])) if m.sum() >= 2 else float("nan")


def prec_at_recall(pts, target):
    """Highest precision achievable at recall >= target (nan if unreachable)."""
    ok = [d["precision"] for d in pts if d["recall"] >= target and not np.isnan(d["precision"])]
    return max(ok) if ok else float("nan")


curves = {}
for method in methods:
    recs, scores = load_method(method)
    if not recs:
        print(f"  {method}: no covered recordings found — skipping")
        continue
    if scores:
        # sweep score cut-points from the score distribution (+ an all-kept anchor)
        qs = np.quantile(scores, np.linspace(0.0, 1.0, args.n_thresholds))
        thresholds = sorted(set(np.round(qs, 6)))
        pts = [micro_point(recs, None)] + [micro_point(recs, t) for t in thresholds]
        curves[method] = {"pts": pts, "scored": True}
        print(f"  {method}: {len(recs)} recs, scored — {len(pts)} operating points, "
              f"score range [{min(scores):.3f}, {max(scores):.3f}]")
    else:
        pts = [micro_point(recs, None)]           # single point: no score to threshold
        curves[method] = {"pts": pts, "scored": False}
        print(f"  {method}: {len(recs)} recs, NO score field — single operating point "
              f"(R={pts[0]['recall']:.3f} P={pts[0]['precision']:.3f}); export a score for a full curve")

if not curves:
    raise SystemExit("No methods with data.")

# ── plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
colors = plt.cm.tab10(np.linspace(0, 0.3, len(curves)))

print("\nSummary:")
for (method, c), color in zip(curves.items(), colors):
    pts = c["pts"]
    rec = [d["recall"] for d in pts]; prec = [d["precision"] for d in pts]
    if c["scored"]:
        order = np.argsort(rec)
        ax.plot(np.array(rec)[order], np.array(prec)[order], "-", color=color, alpha=0.8)
        ax.plot(rec, prec, "o", color=color, ms=3, label=method)
        ap = auc_pr(pts)
        par = "  ".join(f"P@R{t:g}={prec_at_recall(pts, t):.2f}" for t in rec_targets)
        print(f"  {method:10s}: AUC-PR={ap:.3f}   {par}")
    else:
        ax.plot(rec, prec, "X", color=color, ms=13, markeredgecolor="k",
                markeredgewidth=0.6, label=f"{method} (single pt — no score)")
        print(f"  {method:10s}: single point R={rec[0]:.3f} P={prec[0]:.3f} (no score field)")

ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_title("Precision-Recall (channels combined, event-level)")
ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nsaved → {out_path}")

# also dump the raw operating points for downstream use
csv_path = out_path.with_suffix(".csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "threshold", "recall", "precision", "f1",
                                      "tp", "fn", "pred_tp", "fp"])
    w.writeheader()
    for method, c in curves.items():
        for d in c["pts"]:
            w.writerow({"method": method, **{k: d[k] for k in
                        ["threshold", "recall", "precision", "f1", "tp", "fn", "pred_tp", "fp"]}})
print(f"points  → {csv_path}")
