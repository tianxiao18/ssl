"""
Random search over the SAM3 ridge-exemplar hyperparameter space.

Unlike scripts/sweep_hparams.py (one-at-a-time around a baseline), this samples
FULL configurations from the joint space, so it captures interactions between
variables. Each config is scored against ground truth (*annotations_gt.csv, vox
intervals) by 1-D time IoU.

Cost structure: stage-1 params change the ridge candidate box and require a SAM3
GPU pass; stage-2 params only post-filter the resulting masks. So each "trial"
samples ONE stage-1 config (one GPU pass over the session subset) and then
evaluates several random stage-2 samples on the cached masks for free. Total
configs explored = n_trials * stage2_per_trial for only n_trials GPU passes.

Requires a GPU (SAM3 inference runs under torch.autocast(cuda)).

Usage
-----
    python scripts/random_search.py <spec_base> <recording_base> <out_dir> \
        --n-sessions 3 --n-trials 60 --stage2-per-trial 10 \
        --sam3-checkpoint sam3/sam3.pt --objective f1 --seed 0

Outputs
-------
    <out_dir>/random_search_results.csv   one row per evaluated config (sorted by objective)
    <out_dir>/best_config.json            best config + its metrics
    <out_dir>/random_search_scatter.png   all configs in PR space, colored by objective
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vox_tracer.sam3_runner import build_processor, _passes_mask_filters
from vox_tracer.sweep_core import (
    _bbox_to_interval, discover_sessions, gpu_pass, load_gt_for_sessions,
    resolve_sessions, ridge_preds, score_all, stage1_of, stage2_of,
)

# ── search space (edit ranges to retune) ───────────────────────────────────────
# Each sampler takes a random.Random and returns one value. The space was pruned
# using scripts/sweep_hparams.py: variables whose SAM3-precision span was < ~0.02
# across their whole sweep (min_area, horiz_aspect, close_kernel, max_mask_area_frac
# — and sigmas, which had no measurable effect) are dropped to FIXED. Ranges are
# clipped to the useful region of each sweep, away from the cliffs where recall or
# precision collapses (e.g. freq_min ≥ 35k, min_freq_sweep_frac ≥ 0.12,
# score_threshold ≤ 0.3 or ≥ 0.9). See FIXED for the pinned (non-searched) params.

# Pinned params that the pipeline needs but the sweeps showed don't matter.
# score_threshold is NOT here: the processor is built at 0.0 so every mask is
# returned with its score, and score_threshold is applied post-hoc per sample
# (a free stage-2-style knob — see sam3_preds_scored).
FIXED = dict(
    sample_rate=125000,
    sigmas=(2, 3, 4),
    min_area=30,
    horiz_aspect=0.2,
    close_kernel=(7, 3),
    max_mask_area_frac=0.15,
)

# Stage-1 params change the ridge candidate box → cost one SAM3 GPU pass each.
STAGE1_SPACE = {
    "threshold_pct": lambda r: round(r.uniform(99.7, 97), 2),
    "freq_min":      lambda r: float(r.choice([15000, 20000, 25000, 30000])),
    "vert_aspect":   lambda r: round(r.uniform(2.5, 6.0), 2),
}

# Stage-2 params post-filter cached masks → evaluated for free, many per pass.
# score_threshold is the strongest knob in the whole study (precision span ~0.8).
# SCORE_THR_MIN is the floor of the score_threshold sweep: the processor is built
# at this confidence (not 0.0) so SAM3's ~200 query proposals are pruned at the
# source to the masks that *could* survive some sampled score_threshold. Masks
# scoring <= SCORE_THR_MIN are dropped by every sample anyway (see sam3_preds_scored),
# so this is exact — it just avoids caching ~200 full-res masks per window.
# KEEP THIS EQUAL TO the lower bound of score_threshold below.
SCORE_THR_MIN = 0.4
STAGE2_SPACE = {
    "min_freq_sweep_frac": lambda r: round(r.uniform(0.03, 0.10), 3),
    "min_mask_cols":       lambda r: r.randint(1, 12),
    "score_threshold":     lambda r: round(r.uniform(SCORE_THR_MIN, 0.8), 2),
}


def sam3_preds_scored(windows, stage2, score_thr):
    """sweep_core.sam3_preds, but also drop masks scoring <= score_thr.

    The processor is built at confidence 0.0, so raw_masks holds every mask with
    its score; thresholding here reproduces building at that confidence exactly
    (the model applies score as a pure cutoff), for free.
    """
    out = defaultdict(list)
    for w in windows:
        if w["best_box"] is None:
            continue
        for mask_u8, score in w["raw_masks"]:
            if score <= score_thr:
                continue
            if not _passes_mask_filters(mask_u8, w["H"], w["W"], stage2["max_mask_area_frac"],
                                        stage2["min_freq_sweep_frac"], stage2["min_mask_cols"]):
                continue
            x, _, w_box, _ = cv2.boundingRect((mask_u8 > 0).astype(np.uint8))
            out[(w["session"], w["ch"])].append(
                _bbox_to_interval(x, w_box, w["window_start"], w["window_end"], w["W"]))
    return out


def sample(space, rng):
    return {k: fn(rng) for k, fn in space.items()}


def _obj(metrics, objective):
    v = metrics[objective]
    return -1.0 if (v is None or math.isnan(v)) else v


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_scatter(rows, objective, best, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    rec = [r["sam3_recall"] for r in rows]
    prec = [r["sam3_precision"] for r in rows]
    col = [_obj({"f1": r["sam3_f1"], "precision": r["sam3_precision"],
                 "recall": r["sam3_recall"]}, objective) for r in rows]
    sc = ax.scatter(rec, prec, c=col, cmap="viridis", s=25, alpha=0.8)
    ax.plot(best["sam3_recall"], best["sam3_precision"], "*",
            color="red", markersize=22, markeredgecolor="k", label="best", zorder=5)
    fig.colorbar(sc, ax=ax, label=objective)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Random search: {len(rows)} configs (color = {objective})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

CSV_FIELDS = ["trial", "threshold_pct", "freq_min", "vert_aspect",
              "min_freq_sweep_frac", "min_mask_cols", "score_threshold",
              "sam3_tp", "sam3_fp", "sam3_fn", "sam3_precision", "sam3_recall", "sam3_f1",
              "ridge_precision", "ridge_recall", "ridge_f1"]


def _row(trial, cfg, sm, rm):
    return {
        "trial": trial,
        "threshold_pct": cfg["threshold_pct"], "freq_min": cfg["freq_min"],
        "vert_aspect": cfg["vert_aspect"],
        "min_freq_sweep_frac": cfg["min_freq_sweep_frac"],
        "min_mask_cols": cfg["min_mask_cols"],
        "score_threshold": cfg["score_threshold"],
        "sam3_tp": sm["tp"], "sam3_fp": sm["fp"], "sam3_fn": sm["fn"],
        "sam3_precision": sm["precision"], "sam3_recall": sm["recall"], "sam3_f1": sm["f1"],
        "ridge_precision": rm["precision"], "ridge_recall": rm["recall"], "ridge_f1": rm["f1"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec_base")
    p.add_argument("recording_base", help="base dir holding <rel>/*annotations_gt.csv")
    p.add_argument("out_dir")
    p.add_argument("--sessions", default=None,
                   help="comma-list of experiment_X/idx_Y rel-paths (overrides --n-sessions)")
    p.add_argument("--n-sessions", type=int, default=20,
                   help="if --sessions unset, randomly sample N sessions across ALL "
                        "experiments (seeded by --seed for reproducibility)")
    p.add_argument("--channels", default="118,35")
    p.add_argument("--sam3-checkpoint", default=None)
    p.add_argument("--n-trials", type=int, default=40,
                   help="number of stage-1 configs sampled (each = one SAM3 GPU pass)")
    p.add_argument("--stage2-per-trial", type=int, default=8,
                   help="random stage-2 samples scored per stage-1 pass (cheap)")
    p.add_argument("--objective", default="f1", choices=["f1", "precision", "recall"])
    p.add_argument("--iou-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    channels = [int(c) for c in args.channels.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Session selection. With --sessions, use exactly those. Otherwise sample
    # n_sessions uniformly at random across ALL discovered sessions (spanning
    # experiments) — NOT the first N in sorted order, which would tune to a
    # single experiment. A dedicated RNG (same seed) keeps the draw reproducible
    # and independent of the config-sampling stream.
    if args.sessions:
        sessions = resolve_sessions(args.spec_base, args.recording_base, args.sessions, None)
    else:
        all_rels = list(discover_sessions(args.spec_base))
        n = min(args.n_sessions, len(all_rels))
        chosen = sorted(random.Random(args.seed).sample(all_rels, n))
        sessions = resolve_sessions(args.spec_base, args.recording_base, ",".join(chosen), None)
    if not sessions:
        print("No sessions resolved, aborting.")
        sys.exit(1)
    print(f"Random search over {len(sessions)} session(s) "
          f"(of {len(list(discover_sessions(args.spec_base)))} total), channels {channels}:")
    for rel, _, _ in sessions:
        print(f"  {rel}")
    # persist the chosen sessions so the run is auditable / reproducible
    with open(out_dir / "sessions.txt", "w") as f:
        f.write("\n".join(rel for rel, _, _ in sessions) + "\n")

    gt_by_session = load_gt_for_sessions(sessions)
    print(f"GT vox events: {sum(len(v) for v in gt_by_session.values())} total")
    n_configs = args.n_trials * args.stage2_per_trial
    print(f"Plan: {args.n_trials} GPU passes x {args.stage2_per_trial} stage-2 samples "
          f"= {n_configs} configs, objective={args.objective}")

    # Build at the score_threshold floor (not 0.0): SAM3 emits ~200 query
    # proposals per prompt, and confidence 0.0 would cache every one at full
    # resolution across all sessions at once (tens-to-hundreds of GB → OOM).
    # Masks scoring <= SCORE_THR_MIN are dropped by every sampled score_threshold
    # anyway, so pruning them here is exact. The searched score_threshold is
    # still applied post-hoc in sam3_preds_scored for the remaining masks.
    processor = build_processor(args.sam3_checkpoint, SCORE_THR_MIN)

    rows = []
    best, best_score = None, -1.0

    for trial in range(args.n_trials):
        s1 = {**FIXED, **sample(STAGE1_SPACE, rng)}
        windows = gpu_pass(processor, sessions, channels, stage1_of(s1),
                           progress_desc=f"trial {trial + 1}/{args.n_trials}")
        rm = score_all(ridge_preds(windows), gt_by_session, sessions, channels, args.iou_threshold)

        best_in_trial = -1.0
        for _ in range(args.stage2_per_trial):
            s2 = sample(STAGE2_SPACE, rng)
            cfg = {**s1, **s2}
            sm = score_all(sam3_preds_scored(windows, stage2_of(cfg), cfg["score_threshold"]),
                           gt_by_session, sessions, channels, args.iou_threshold)
            row = _row(trial, cfg, sm, rm)
            rows.append(row)
            sc = _obj(sm, args.objective)
            best_in_trial = max(best_in_trial, sc)
            if sc > best_score:
                best_score, best = sc, row
        del windows  # free this trial's cached masks before the next pass
        print(f"  trial {trial + 1}/{args.n_trials}: ridge_F1={rm['f1']:.3f} "
              f"best_{args.objective}_in_trial={best_in_trial:.3f} "
              f"| global_best={best_score:.3f}")

    rows.sort(key=lambda r: _obj({"f1": r["sam3_f1"], "precision": r["sam3_precision"],
                                  "recall": r["sam3_recall"]}, args.objective), reverse=True)

    csv_path = out_dir / "random_search_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults ({len(rows)} configs) → {csv_path}")

    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    print(f"Best config → {out_dir / 'best_config.json'}")

    plot_scatter(rows, args.objective, best, out_dir / "random_search_scatter.png")

    print(f"\nTop 5 by {args.objective}:")
    for r in rows[:5]:
        print(f"  P={r['sam3_precision']:.3f} R={r['sam3_recall']:.3f} F1={r['sam3_f1']:.3f} | "
              f"conf={r['score_threshold']} thr={r['threshold_pct']} freq_min={r['freq_min']:.0f} "
              f"vert={r['vert_aspect']} sweep={r['min_freq_sweep_frac']} cols={r['min_mask_cols']}")


if __name__ == "__main__":
    main()
