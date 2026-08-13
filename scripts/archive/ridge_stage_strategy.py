"""
Two ways to split the filtering work between stage-1 and stage-2 — run both,
cross-validate, and report which wins.

  Strategy A (fix stage-2, tune stage-1): hold stage-2 fixed at a given filter
      (current, and the best geometric one) and pick the stage-1 config that
      maximises event F1 on the training folds.
  Strategy B (recall then precision): pick the stage-1 config that preserves
      (within eps) the best achievable raw recall while handing the FEWEST
      candidates downstream, freeze it, then tune stage-2 for precision.

Everything is scored at the event level (vox_tracer.scoring), with grouped
k-fold over sessions (tune on train folds, evaluate on held out), so the numbers
are out-of-sample and come with a spread. All features are geometric (no audio),
so the stage-1 sweep is cheap; the sato response is computed once per chunk and
reused across configs.

    python scripts/ridge_stage_strategy.py --n-sessions 8 --folds 4
"""
import argparse
import sys
from itertools import product
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
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))
import ridge_pipeline_viz as rpv  # noqa: E402
from ridge_filter_experiment import discover_sessions, event_metrics  # noqa: E402
from spectral_features import bbox_to_time  # noqa: E402
from vox_tracer.spec import parse_spec_fname  # noqa: E402
from vox_tracer.scoring import score_combined  # noqa: E402

# stage-1 configs: (min_area, use_aspect_cuts, use_freq_cut)
S1_CONFIGS = {
    "area30":                 (30,  False, False),
    "area30+freq":            (30,  False, True),
    "area30+aspect+freq*":    (30,  True,  True),   # * = current stage-1
    "area60+freq":            (60,  False, True),
    "area100+freq":           (100, False, True),
}
CURRENT_S1 = "area30+aspect+freq*"


def cfg_params(base, min_area, aspect, freq, H):
    p = dict(base)
    p["min_area"] = min_area
    p["vert_aspect"] = 5.0 if aspect else 1e9
    p["horiz_aspect"] = 0.2 if aspect else 0.0
    nyq = base["sample_rate"] / 2.0
    p["freq_cutoff_row"] = int((H - 1) * (1.0 - base["freq_min"] / nyq)) if freq else H + 1
    return p, nyq


def build(sessions, channels=(118, 35)):
    """Return {config_name: df} of geometric detections, sharing one response/chunk."""
    base = dict(rpv.DEFAULTS)
    out = {name: [] for name in S1_CONFIGS}
    passall = []  # (session, t0, t1, label) for the recall/candidate frontier
    fn = rpv._build_filter_fn(base["filter_name"], list(base["sigmas"]))
    for session, spec_dir, _, _, vox in sessions:
        for ch in channels:
            for png in sorted(spec_dir.glob(f"headmic_{ch}_*.png")):
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                H, W = gray.shape
                resp = fn(gray.astype(np.float64) / 255.0)
                _, wt0, wt1 = parse_spec_fname(png.name)
                for name, (ma, asp, frq) in S1_CONFIGS.items():
                    p, nyq = cfg_params(base, ma, asp, frq, H)
                    res = rpv.run_pipeline(gray, p, stage1=True, response=resp)
                    for comp, _, st in res["s2"]:
                        t0, t1 = bbox_to_time(st["bx"], st["bw"], wt0, wt1, W)
                        bmask = comp > 0
                        out[name].append(dict(
                            session=session, t0=t0, t1=t1,
                            label=rpv.label_tpfp(st["x0"], st["x1"], wt0, wt1, W, vox),
                            n_cols=st["n_cols"], sweep_frac=st["sweep_frac"],
                            area_frac=st["area_frac"],
                            f_high_khz=(H - 1 - st["by"]) / (H - 1) * nyq / 1e3,
                            ridge_score=float(resp[bmask].mean()) if bmask.any() else 0.0))
                # pass-all frontier point (stage-1 OFF), lightweight stats only
                p, nyq = cfg_params(base, 1, False, False, H)
                res = rpv.run_pipeline(gray, p, stage1=False, response=resp)
                for _, _, st in res["s2"]:
                    t0, t1 = bbox_to_time(st["bx"], st["bw"], wt0, wt1, W)
                    passall.append(dict(session=session, t0=t0, t1=t1,
                                        label=rpv.label_tpfp(st["x0"], st["x1"], wt0, wt1, W, vox)))
    return {k: pd.DataFrame(v) for k, v in out.items()}, pd.DataFrame(passall)


# stage-2 filters
def current_s2(d):
    return (d.area_frac <= 0.15) & (d.n_cols >= 5) & (d.sweep_frac >= 0.04)


def best_s2(d):
    return (d.n_cols >= 9) & (d.f_high_khz >= 28)


def s2_from_thr(d, ncol, fhigh):
    return (d.area_frac <= 0.15) & (d.n_cols >= ncol) & (d.f_high_khz >= fhigh)


def raw_recall_candidates(df, sess_subset):
    """Event recall with NO stage-2, and total candidate count, over a subset.

    Score-independent (no ridge_score needed), so it also works on the pass-all
    frontier table which carries only geometry.
    """
    sub = df[df.session.isin([s[0] for s in sess_subset])]
    tp = fn = 0
    for name, _, _, _, vox in sess_subset:
        s = sub[sub.session == name]
        boxes = list(zip(s.t0, s.t1, [None] * len(s)))
        r = score_combined(vox, boxes, 0.0)
        tp += r["tp"]; fn += r["fn"]
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return recall, len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=8)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.01, help="recall tolerance for strategy B")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval/ridge_stage_strategy"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(args.n_sessions)
    names = [s[0] for s in sessions]
    print(f"{len(sessions)} sessions: {names}")
    data, passall = build(sessions)
    for k, df in data.items():
        print(f"  {k:24s} {len(df):5d} dets  ({(df.label=='tp').sum()} TP)")
    print(f"  {'pass-all (stage-1 OFF)':24s} {len(passall):5d} dets  ({(passall.label=='tp').sum()} TP)")

    sess_by_name = {s[0]: s for s in sessions}
    folds = [names[i::args.folds] for i in range(args.folds)]

    def subset(nm):
        return [sess_by_name[n] for n in nm]

    def f1_config_s2(cfg_df, s2fn, sess_subset):
        return event_metrics(cfg_df, s2fn(cfg_df).fillna(False), sess_subset)

    # ── run the strategies across folds ─────────────────────────────────────
    results = {s: [] for s in ["baseline", "A: fix current-s2, tune s1",
                               "A: fix best-s2, tune s1", "B: recall->precision"]}
    picks = {k: [] for k in results}
    for te_names in folds:
        tr_names = [n for n in names if n not in te_names]
        tr, te = subset(tr_names), subset(te_names)

        # baseline: current stage-1 + current stage-2 (no tuning)
        results["baseline"].append(f1_config_s2(data[CURRENT_S1], current_s2, te))

        # Strategy A, stage-2 fixed, tune stage-1 by train F1
        for label, s2fn in (("A: fix current-s2, tune s1", current_s2),
                            ("A: fix best-s2, tune s1", best_s2)):
            best = max(S1_CONFIGS, key=lambda c: f1_config_s2(data[c], s2fn, tr)["f1"])
            picks[label].append(best)
            results[label].append(f1_config_s2(data[best], s2fn, te))

        # Strategy B: stage-1 = max recall (within eps) w/ fewest candidates, then tune stage-2
        rc = {c: raw_recall_candidates(data[c], tr) for c in S1_CONFIGS}
        best_rec = max(r for r, _ in rc.values())
        cand = [(c, n) for c, (r, n) in rc.items() if r >= best_rec - args.eps]
        s1_pick = min(cand, key=lambda x: x[1])[0]
        picks["B: recall->precision"].append(s1_pick)
        df1 = data[s1_pick]
        grid = list(product([5, 7, 9, 11], [0, 24, 26, 28, 30]))
        best_thr = max(grid, key=lambda g: event_metrics(
            df1, s2_from_thr(df1, *g).fillna(False), tr)["f1"])
        results["B: recall->precision"].append(
            event_metrics(df1, s2_from_thr(df1, *best_thr).fillna(False), te))

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\n=== grouped {args.folds}-fold CV (mean ± std over held-out folds) ===")
    print(f"  {'strategy':30s} {'F1':>14} {'recall':>13} {'precision':>13}")
    summary = {}
    for s, ms in results.items():
        f1 = np.array([m["f1"] for m in ms]); rc = np.array([m["recall"] for m in ms])
        pr = np.array([m["precision"] for m in ms])
        summary[s] = (f1.mean(), f1.std(), rc.mean(), rc.std(), pr.mean(), pr.std())
        extra = f"  picks={picks[s]}" if picks[s] else ""
        print(f"  {s:30s} {f1.mean():.3f}±{f1.std():.3f}  {rc.mean():.3f}±{rc.std():.3f}"
              f"  {pr.mean():.3f}±{pr.std():.3f}{extra}")

    winner = max(summary, key=lambda s: summary[s][0])
    print(f"\nWINNER by mean F1: {winner}  (F1={summary[winner][0]:.3f})")

    # ── frontier data (recall vs candidates per stage-1 config, all sessions) ──
    front = []
    for c in S1_CONFIGS:
        r, n = raw_recall_candidates(data[c], sessions)
        front.append((c, n, r))
    r, n = raw_recall_candidates(passall, sessions)
    front.append(("pass-all (stage-1 OFF)", n, r))

    # ── visualize ─────────────────────────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    order = ["baseline", "A: fix current-s2, tune s1", "A: fix best-s2, tune s1",
             "B: recall->precision"]
    x = np.arange(len(order)); w = 0.26
    for i, (metric, color, off) in enumerate([("recall", "#0ca30c", -w),
                                              ("precision", "#2a78d6", 0),
                                              ("f1", "#d03b3b", w)]):
        means = [summary[s][{"f1":0,"recall":2,"precision":4}[metric]] for s in order]
        stds = [summary[s][{"f1":1,"recall":3,"precision":5}[metric]] for s in order]
        axL.bar(x + off, means, w, yerr=stds, capsize=3, color=color, label=metric)
    for i, s in enumerate(order):
        axL.text(i + w, summary[s][0] + summary[s][1] + 0.01, f"{summary[s][0]:.2f}",
                 ha="center", fontsize=9, fontweight="bold")
    axL.set_xticks(x)
    axL.set_xticklabels([s.replace(": ", ":\n") for s in order], fontsize=8.5)
    axL.set_ylim(0, 1.02); axL.set_ylabel("event-level metric (held-out CV)")
    axL.set_title(f"Strategy comparison — grouped {args.folds}-fold CV", fontweight="bold")
    axL.legend(loc="lower left", ncol=3, fontsize=9)
    axL.grid(axis="y", alpha=0.3); axL.spines[["top", "right"]].set_visible(False)

    for c, n, r in front:
        is_pa = "pass-all" in c
        is_cur = c == CURRENT_S1
        axR.scatter(n, r, s=90, zorder=3,
                    color="#d03b3b" if is_pa else ("#e08a1e" if is_cur else "#2a78d6"))
        axR.annotate(c, (n, r), fontsize=8, xytext=(6, -4), textcoords="offset points")
    axR.set_xscale("log")
    axR.set_xlabel("candidates handed to stage-2 (log)"); axR.set_ylabel("raw recall (no stage-2)")
    axR.set_title("Stage-1: recall vs. candidate count\n(pass-all = degenerate recall-max corner)",
                  fontweight="bold")
    axR.grid(alpha=0.3); axR.spines[["top", "right"]].set_visible(False)

    fig.savefig(args.out_dir / "strategy_comparison.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    print(f"\nwrote {args.out_dir}/strategy_comparison.png")


if __name__ == "__main__":
    main()
