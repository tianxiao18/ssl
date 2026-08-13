"""
One-at-a-time hyperparameter sweep for the SAM3 ridge-exemplar pipeline.

For each tunable variable, hold every other variable at its baseline and sweep
that one across a grid. Each grid value is scored against ground truth
(*annotations_gt.csv, vox intervals) by 1-D time IoU, giving one
precision/recall operating point. Connecting the points in grid order traces a
precision-recall curve per variable.

Stage-1 variables (threshold_pct, min_area, vert_aspect, horiz_aspect,
freq_min, close_kernel_w) change the ridge candidate box, so they re-run SAM3
and we plot BOTH the ridge-stage PR curve and the SAM3 PR curve.

Stage-2 variables (max_mask_area_frac, min_freq_sweep_frac, min_mask_cols) only
post-filter SAM3's output masks, so a single cached SAM3 pass is re-scored for
every value (cheap, no extra GPU work) and only the SAM3 curve is drawn.

Caveat: this is a sensitivity screen around one baseline. It does not capture
interactions between variables (use scripts/hparam_search/random_search.py for that).

Requires a GPU (SAM3 inference runs under torch.autocast(cuda)).

Usage
-----
    python scripts/hparam_search/sweep_hparams.py <spec_base> <recording_base> <out_dir> \
        --sessions experiment_445/idx_011,experiment_445/idx_012 \
        --sam3-checkpoint sam3/sam3.pt

    python scripts/hparam_search/sweep_hparams.py <spec_base> <recording_base> <out_dir> \
        --n-sessions 4 --sam3-checkpoint sam3/sam3.pt --only threshold_pct,min_mask_cols

Outputs
-------
    <out_dir>/pr_<variable>.png    PR curve(s) for each swept variable
    <out_dir>/sweep_results.csv    every (variable, value, detector) operating point
"""
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vox_tracer.sam3_runner import build_processor
from vox_tracer.sweep_core import (
    gpu_pass, load_gt_for_sessions, resolve_sessions, ridge_preds, sam3_preds,
    score_all, stage1_of, stage2_of,
)

# ── baseline config + sweep grids (edit these to retune) ───────────────────────

BASELINE = dict(
    sigmas=(2, 3, 4),
    threshold_pct=99.0,
    sample_rate=125000,
    freq_min=20000.0,
    min_area=30,
    vert_aspect=5.0,
    horiz_aspect=0.2,
    close_kernel=(7, 3),
    max_mask_area_frac=0.15,
    min_freq_sweep_frac=0.04,
    min_mask_cols=5,
    score_threshold=0.5,   # SAM3 confidence; applied at processor build, not swept here
)


@dataclass
class Var:
    label: str
    stage: int                       # 1 = re-runs SAM3 (+ridge curve), 2 = post-hoc only
    values: list
    apply: Callable                  # (baseline_dict, value) -> overridden cfg dict
    base: object                     # baseline value (highlighted on the plot)


def _set(key):
    return lambda b, v: {**b, key: v}


SWEEP_VARS = [
    Var("threshold_pct",       1, [97.0, 98.0, 99.0, 99.5, 99.8], _set("threshold_pct"),       99.0),
    Var("min_area",            1, [10, 30, 60, 100, 150],         _set("min_area"),             30),
    Var("vert_aspect",         1, [3.0, 5.0, 8.0, 12.0],          _set("vert_aspect"),          5.0),
    Var("horiz_aspect",        1, [0.1, 0.2, 0.3, 0.5],           _set("horiz_aspect"),         0.2),
    Var("freq_min",            1, [10000.0, 20000.0, 30000.0, 40000.0], _set("freq_min"),       20000.0),
    Var("close_kernel_w",      1, [3, 5, 7, 9, 11],   lambda b, v: {**b, "close_kernel": (v, 3)}, 7),
    Var("max_mask_area_frac",  2, [0.05, 0.1, 0.15, 0.25, 0.4],   _set("max_mask_area_frac"),   0.15),
    Var("min_freq_sweep_frac", 2, [0.0, 0.02, 0.04, 0.08, 0.15],  _set("min_freq_sweep_frac"),  0.04),
    Var("min_mask_cols",       2, [1, 3, 5, 10, 20],              _set("min_mask_cols"),        5),
]


# ── plotting ───────────────────────────────────────────────────────────────────

def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def plot_pr(title, series, base, out_path):
    """series: list of (name, [(value, metrics)], color). base: value to star."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pts, color in series:
        rec = [m["recall"] for _, m in pts]
        prec = [m["precision"] for _, m in pts]
        ax.plot(rec, prec, "-o", color=color, label=name)
        for v, m in pts:
            ax.annotate(_fmt(v), (m["recall"], m["precision"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points", color=color)
            if v == base:
                ax.plot(m["recall"], m["precision"], "*", color=color,
                        markersize=16, markeredgecolor="k", markeredgewidth=0.5, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title + "   (★ = baseline)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec_base")
    p.add_argument("recording_base", help="base dir holding <rel>/*annotations_gt.csv")
    p.add_argument("out_dir")
    p.add_argument("--sessions", default=None,
                   help="comma-list of experiment_X/idx_Y rel-paths (overrides --n-sessions)")
    p.add_argument("--n-sessions", type=int, default=4,
                   help="if --sessions unset, auto-pick the first N discovered sessions")
    p.add_argument("--channels", default="118,35")
    p.add_argument("--sam3-checkpoint", default=None)
    p.add_argument("--iou-threshold", type=float, default=0.0,
                   help="min 1-D time IoU to count a match (default 0.0 = any overlap)")
    p.add_argument("--only", default=None,
                   help="comma-list of variable labels to sweep (default: all)")
    args = p.parse_args()

    channels = [int(c) for c in args.channels.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = resolve_sessions(args.spec_base, args.recording_base, args.sessions, args.n_sessions)
    if not sessions:
        print("No sessions resolved, aborting.")
        sys.exit(1)
    print(f"Sweeping over {len(sessions)} session(s), channels {channels}:")
    for rel, _, _ in sessions:
        print(f"  {rel}")

    gt_by_session = load_gt_for_sessions(sessions)
    print(f"GT vox events: {sum(len(v) for v in gt_by_session.values())} total")

    vars_to_run = SWEEP_VARS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        vars_to_run = [v for v in SWEEP_VARS if v.label in wanted]
        if not vars_to_run:
            print(f"--only matched no variables; choose from {[v.label for v in SWEEP_VARS]}")
            sys.exit(1)

    processor = build_processor(args.sam3_checkpoint, BASELINE["score_threshold"])

    rows = []
    baseline_windows = None  # cached SAM3 pass for all stage-2 sweeps

    for var in vars_to_run:
        print(f"\n=== sweeping {var.label} (stage {var.stage}) ===")

        if var.stage == 2:
            if baseline_windows is None:
                print("  running baseline SAM3 pass (cached for stage-2 sweeps)…")
                baseline_windows = gpu_pass(processor, sessions, channels, stage1_of(BASELINE))
            sam3_series = []
            for v in var.values:
                cfg = var.apply(BASELINE, v)
                m = score_all(sam3_preds(baseline_windows, stage2_of(cfg)),
                              gt_by_session, sessions, channels, args.iou_threshold)
                sam3_series.append((v, m))
                rows.append({"variable": var.label, "value": _fmt(v), "detector": "sam3", **m})
                print(f"    {var.label}={_fmt(v)}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
            plot_pr(f"PR sweep: {var.label} (stage 2)",
                    [("SAM3", sam3_series, "C0")], var.base, out_dir / f"pr_{var.label}.png")

        else:
            ridge_series, sam3_series = [], []
            for v in var.values:
                cfg = var.apply(BASELINE, v)
                windows = gpu_pass(processor, sessions, channels, stage1_of(cfg))
                rm = score_all(ridge_preds(windows), gt_by_session, sessions, channels, args.iou_threshold)
                sm = score_all(sam3_preds(windows, stage2_of(BASELINE)),
                               gt_by_session, sessions, channels, args.iou_threshold)
                ridge_series.append((v, rm))
                sam3_series.append((v, sm))
                rows.append({"variable": var.label, "value": _fmt(v), "detector": "ridge", **rm})
                rows.append({"variable": var.label, "value": _fmt(v), "detector": "sam3", **sm})
                print(f"    {var.label}={_fmt(v)}: ridge P={rm['precision']:.3f} R={rm['recall']:.3f} | "
                      f"sam3 P={sm['precision']:.3f} R={sm['recall']:.3f}")
            plot_pr(f"PR sweep: {var.label} (stage 1)",
                    [("ridge", ridge_series, "C1"), ("SAM3", sam3_series, "C0")],
                    var.base, out_dir / f"pr_{var.label}.png")

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variable", "value", "detector",
                                               "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
