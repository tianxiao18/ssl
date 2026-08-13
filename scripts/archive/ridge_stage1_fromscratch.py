"""
From-scratch stage-1 search (NOT anchored to the current filter).

Enumerate ~18 stage-1 component filters built independently from the gate pool
  area>=30/50 | aspect in [0.2,5] | f_high>=20k/28k | ridge>=.05 | solidity<=.7
(no "current + X" — current is just one entry). Run each through the full pipeline
(component gate -> morph-close -> gate again -> fixed best stage-2 n_cols>=9 &
f_high>=28kHz) and rank by event-level F1 with grouped k-fold CV. Spectral gates
are excluded: the component-level AUC showed them weak on fragments (flatness 0.55).

    python scripts/ridge_stage1_fromscratch.py --n-sessions 6 --folds 4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))
import ridge_pipeline_viz as rpv  # noqa: E402
from ridge_stage1_tuning import run_custom, best_s2, BASE  # noqa: E402
from ridge_filter_experiment import discover_sessions, event_metrics  # noqa: E402
from vox_tracer.spec import parse_spec_fname  # noqa: E402

# gate atoms (feature, op, thr); thresholds in feature units (f_high in Hz)
A30 = ("area", ">=", 30);   A50 = ("area", ">=", 50)
AV = ("aspect", "<=", 5.0); AH = ("aspect", ">=", 0.2)   # the two current aspect cuts
F20 = ("f_high", ">=", 20000.0); F28 = ("f_high", ">=", 28000.0)
R = ("ridge", ">=", 0.05);  S = ("solidity", "<=", 0.7)

CONFIGS = {
    "empty":                 [],
    "area30":                [A30],
    "f_high>=20k":           [F20],
    "f_high>=28k":           [F28],
    "ridge>=.05":            [R],
    "solidity<=.7":          [S],
    "aspect only":           [AV, AH],
    "area30+f20":            [A30, F20],
    "area30+f28":            [A30, F28],
    "area30+ridge":          [A30, R],
    "area30+solidity":       [A30, S],
    "area30+aspect":         [A30, AV, AH],
    "area30+f20+ridge":      [A30, F20, R],
    "area30+f20+aspect*":    [A30, F20, AV, AH],   # * == the current filter
    "area30+f28+ridge":      [A30, F28, R],
    "area30+f28+aspect":     [A30, F28, AV, AH],
    "area50+f20+aspect":     [A50, F20, AV, AH],
    "area30+f20+aspect+ridge": [A30, F20, AV, AH, R],
}
CURRENT = "area30+f20+aspect*"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=6)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval/ridge_stage1_fromscratch"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(args.n_sessions)
    names = [s[0] for s in sessions]
    print(f"{len(sessions)} sessions")

    fn = rpv._build_filter_fn(BASE["filter_name"], list(BASE["sigmas"]))
    rows = {c: [] for c in CONFIGS}
    for session, spec_dir, _, _, vox in sessions:
        for ch in (118, 35):
            for png in sorted(spec_dir.glob(f"headmic_{ch}_*.png")):
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                H, W = gray.shape
                resp = fn(gray.astype(np.float64) / 255.0)
                _, wt0, wt1 = parse_spec_fname(png.name)
                for c, gates in CONFIGS.items():
                    for d in run_custom(resp, H, W, gates, wt0, wt1, vox):
                        d["session"] = session
                        rows[c].append(d)

    sess_by = {s[0]: s for s in sessions}
    fold_sets = [names[i::args.folds] for i in range(args.folds)]

    summary = []
    for c, r in rows.items():
        df = pd.DataFrame(r)
        f1s, rcs, prs = [], [], []
        for te in fold_sets:
            sub = [sess_by[n] for n in te]
            keep = best_s2(df).fillna(False) if len(df) else pd.Series([], dtype=bool)
            m = event_metrics(df, keep, sub)
            f1s.append(m["f1"]); rcs.append(m["recall"]); prs.append(m["precision"])
        summary.append((c, np.mean(f1s), np.std(f1s), np.mean(rcs), np.std(rcs),
                        np.mean(prs), np.std(prs)))

    summary.sort(key=lambda t: -t[1])
    pd.DataFrame(summary, columns=["filter", "f1", "f1_std", "recall", "recall_std",
                                   "prec", "prec_std"]).to_csv(
        args.out_dir / "fromscratch_metrics.csv", index=False)
    print(f"\n=== from-scratch stage-1 filters, ranked by CV F1 (fixed best stage-2) ===")
    print(f"  {'rank':>4} {'stage-1 filter':26s} {'F1':>14} {'recall':>8} {'prec':>8}")
    for i, (c, f1, sd, rc, rcsd, pr, prsd) in enumerate(summary, 1):
        star = "  <- CURRENT" if c == CURRENT else ""
        print(f"  {i:>4} {c:26s} {f1:.3f}±{sd:.3f}  {rc:.3f}  {pr:.3f}{star}")
    best = summary[0]
    cur = next(t for t in summary if t[0] == CURRENT)
    print(f"\nbest={best[0]} (F1 {best[1]:.3f})   current={CURRENT} (F1 {cur[1]:.3f}, "
          f"rank {[t[0] for t in summary].index(CURRENT)+1})")

    # visualize (no suptitle)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    labels = [t[0] for t in summary][::-1]
    f1 = [t[1] for t in summary][::-1]; sd = [t[2] for t in summary][::-1]
    colors = ["#e08a1e" if l == CURRENT else ("#0ca30c" if t[0] == best[0] else "#8a8984")
              for l, t in zip(labels, summary[::-1])]
    ax.barh(range(len(labels)), f1, xerr=sd, color=colors, capsize=2)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("event-level F1 (grouped CV, fixed best stage-2)")
    ax.set_title("From-scratch stage-1 filters ranked "
                 "(green=best, orange=current)", fontweight="bold")
    ax.axvline(cur[1], color="#e08a1e", ls=":", lw=1)
    for i, (v, s) in enumerate(zip(f1, sd)):
        ax.text(v + s + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.out_dir / "fromscratch_rank.png", dpi=140, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out_dir}/fromscratch_rank.png")


if __name__ == "__main__":
    main()
