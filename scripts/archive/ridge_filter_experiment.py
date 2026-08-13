"""
Experiment: what is the best way to filter ridge detections?

Builds a labeled dataset of RAW ridge detections (every contour of the final
segmentation mask, before passes_mask_filters) across several recordings, with
both spectral and geometric features, then:

  1. ranks every feature by how well it separates TP from FP (ROC-AUC),
     for stage-2 detections AND stage-1 raw components;
  2. fits a small model (depth-3 tree + logistic reg) with a session-level
     train/test split to find the best feature *combination*;
  3. evaluates candidate filter STRATEGIES at the event level (the real metric:
     recording-level recall/precision via vox_tracer.scoring), on held-out
     sessions, so the numbers are not in-sample.

TP = detection's time span overlaps a GT vox call; FP otherwise.

Usage:
    python scripts/ridge_filter_experiment.py --n-sessions 8 \
        --out-dir outputs/eval/ridge_filter_experiment
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))

import ridge_pipeline_viz as rpv  # run_pipeline, DEFAULTS, label helpers  # noqa: E402
from spectral_features import bbox_to_band, bbox_to_time, event_features  # noqa: E402
from vox_tracer.spec import load_channel_audio, parse_spec_fname  # noqa: E402
from vox_tracer.scoring import score_combined  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

SPEC_FEATS = ["centroid", "bandwidth", "rolloff", "flatness", "rms", "zcr", "peak_freq"]
GEO_FEATS = ["area_px", "aspect", "n_cols", "sweep_frac", "area_frac",
             "solidity", "f_high_khz", "f_band_khz", "duration_ms", "ridge_score"]
ALL_FEATS = SPEC_FEATS + GEO_FEATS


# ── dataset construction ──────────────────────────────────────────────────────

def discover_sessions(n, min_vox=15):
    """Pick up to n sessions (one per distinct experiment) with GT vox + audio."""
    picks, seen_exp = [], set()
    for exp in sorted((ROOT / "outputs/spectrograms").glob("experiment_*")):
        if exp.name in seen_exp:
            continue
        for idx in sorted(exp.glob("idx_*")):
            data_dir = ROOT / "data" / exp.name / idx.name
            gt = list(data_dir.glob("*annotations_gt.csv"))
            wav = list(data_dir.glob("headmic_*_*.wav"))
            if not (gt and wav):
                continue
            vox = rpv.load_gt_vox(gt[0])
            if len(vox) < min_vox:
                continue
            picks.append((f"{exp.name}/{idx.name}", idx, data_dir, gt[0], vox))
            seen_exp.add(exp.name)
            break
        if len(picks) >= n:
            break
    return picks


def build_dataset(sessions, channels=(118, 35), min_area=None):
    """Run raw ridge on every session; return (detections_df, components_df).

    ``min_area`` overrides the stage-1 component-area cut (rest of stage-1 = current):
    pass the best stage-1's value so the stage-2 study is built on the best stage-1.
    """
    det_rows, comp_rows = [], []
    p = dict(rpv.DEFAULTS)
    if min_area is not None:
        p["min_area"] = min_area
    for session, spec_dir, data_dir, gt_csv, vox in sessions:
        print(f"[{session}] {len(vox)} GT vox")
        for ch in channels:
            loaded = load_channel_audio(data_dir, ch)
            if loaded is None:
                continue
            sr, audio, _ = loaded
            nyq = sr / 2.0
            for png in sorted(spec_dir.glob(f"headmic_{ch}_*.png")):
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                H, W = gray.shape
                pc = dict(p)
                pc["freq_cutoff_row"] = int((H - 1) * (1.0 - p["freq_min"] / nyq))
                res = rpv.run_pipeline(gray, pc)
                _, wt0, wt1 = parse_spec_fname(png.name)
                response = res["response"]

                # stage-1 components (geometric only)
                for r in res["comp_rows"]:
                    y_top = r["y_top"]
                    comp_rows.append(dict(
                        session=session, area_px=r["area"], aspect=r["aspect"],
                        f_high_khz=(H - 1 - y_top) / (H - 1) * nyq / 1e3,
                        label=rpv.label_tpfp(r["x0"], r["x1"], wt0, wt1, W, vox)))

                # stage-2 raw detections (geometric + spectral)
                for comp, fate, st in res["s2"]:
                    t0, t1 = bbox_to_time(st["bx"], st["bw"], wt0, wt1, W)
                    f_low, f_high = bbox_to_band(st["by"], st["bh"], H, nyq)
                    feats = event_features(audio, sr, t0, t1, f_low, f_high)
                    if feats is None:
                        continue
                    bmask = comp > 0
                    row = dict(
                        session=session, channel=ch, t0=t0, t1=t1,
                        label=rpv.label_tpfp(st["x0"], st["x1"], wt0, wt1, W, vox),
                        area_px=int(bmask.sum()), aspect=st["bh"] / max(st["bw"], 1),
                        n_cols=st["n_cols"], sweep_frac=st["sweep_frac"],
                        area_frac=st["area_frac"],
                        solidity=bmask.sum() / max(st["bw"] * st["bh"], 1),
                        f_high_khz=f_high / 1e3, f_band_khz=(f_high - f_low) / 1e3,
                        duration_ms=(t1 - t0) * 1e3,
                        ridge_score=float(response[bmask].mean()) if bmask.any() else 0.0,
                    )
                    for name, (m, _) in feats.items():
                        row[name] = m
                    det_rows.append(row)
    return pd.DataFrame(det_rows), pd.DataFrame(comp_rows)


# ── analysis ──────────────────────────────────────────────────────────────────

def auc_table(df, feats):
    """Per-feature ROC-AUC for TP vs FP (oriented so AUC>=0.5), with TP/FP means."""
    y = (df["label"] == "tp").astype(int).values
    rows = []
    for f in feats:
        v = df[f].values.astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], v[ok])
        rows.append(dict(feature=f, auc=max(a, 1 - a),
                         higher_is_tp=a >= 0.5,
                         tp_med=np.median(v[ok & (y == 1)]),
                         fp_med=np.median(v[ok & (y == 0)])))
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def event_metrics(det_df, keep_mask, sessions_subset):
    """Aggregate recording-level TP/FN/FP over sessions for the kept detections."""
    kept = det_df[keep_mask]
    tp = fn = pred_tp = fp = 0
    vox_by = {s: v for s, _, _, _, v in sessions_subset}
    for session in [s for s, *_ in sessions_subset]:
        sub = kept[kept["session"] == session]
        boxes = list(zip(sub["t0"], sub["t1"], sub["ridge_score"]))
        r = score_combined(vox_by[session], boxes, 0.0)
        tp += r["tp"]; fn += r["fn"]; pred_tp += r["pred_tp"]; fp += r["fp"]
    prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return dict(recall=rec, precision=prec, f1=f1, tp=tp, fn=fn, fp=fp,
                n_kept=int(keep_mask.sum()))


SPEC_SET = {"centroid", "bandwidth", "rolloff", "flatness", "rms", "zcr", "peak_freq"}


def summary_figure(auc_df, strat_labels, strat_rows, out_path):
    """Two-panel: per-feature AUC (left) + strategy grouped-CV metrics w/ std (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    a = auc_df.sort_values("auc")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

    colors = ["#2a78d6" if f in SPEC_SET else "#e08a1e" for f in a.feature]
    axL.barh(range(len(a)), a.auc, color=colors)
    axL.set_yticks(range(len(a))); axL.set_yticklabels(a.feature)
    axL.axvline(0.5, color="#888", ls="--", lw=1); axL.set_xlim(0.5, 0.95)
    for i, (v, f) in enumerate(zip(a.auc, a.feature)):
        axL.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=8)
    axL.set_xlabel("ROC-AUC (TP vs FP), all sessions")
    axL.set_title("Per-feature separability of raw ridge detections", fontweight="bold")
    axL.legend(handles=[Patch(color="#2a78d6", label="spectral"),
                        Patch(color="#e08a1e", label="geometric / response")], loc="lower right")
    axL.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(strat_rows)); w = 0.26
    R = [r["recall"] for r in strat_rows]; Rs = [r["recall_std"] for r in strat_rows]
    P = [r["precision"] for r in strat_rows]; Ps = [r["precision_std"] for r in strat_rows]
    F = [r["f1"] for r in strat_rows]; Fs = [r["f1_std"] for r in strat_rows]
    ekw = dict(capsize=3, ecolor="#333", error_kw=dict(lw=1))
    axR.bar(x - w, R, w, yerr=Rs, label="recall", color="#0ca30c", **ekw)
    axR.bar(x, P, w, yerr=Ps, label="precision", color="#2a78d6", **ekw)
    axR.bar(x + w, F, w, yerr=Fs, label="F1", color="#d03b3b", **ekw)
    for i, (f, s) in enumerate(zip(F, Fs)):
        axR.text(i + w, f + s + 0.012, f"{f:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels(strat_labels, fontsize=8)
    axR.set_ylim(0, 1.05); axR.set_ylabel("event-level metric (grouped CV, mean ± std)")
    axR.set_title("Filter strategies under grouped CV (incl. from-scratch)", fontweight="bold")
    axR.legend(loc="lower left", ncol=3, fontsize=9)
    axR.grid(axis="y", alpha=0.3); axR.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sessions", type=int, default=8)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=None,
                    help="stage-1 min_area (set to the best stage-1's value; default=current 30)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval/ridge_filter_experiment"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(args.n_sessions)
    print(f"\n=== {len(sessions)} sessions  (stage-1 min_area="
          f"{args.min_area if args.min_area else 30}) ===")
    det, comp = build_dataset(sessions, min_area=args.min_area)
    det.to_csv(args.out_dir / "detections.csv", index=False)
    comp.to_csv(args.out_dir / "components.csv", index=False)
    print(f"\ndetections: {len(det)}  ({(det.label=='tp').sum()} TP / "
          f"{(det.label=='fp').sum()} FP)")
    print(f"components : {len(comp)}  ({(comp.label=='tp').sum()} TP / "
          f"{(comp.label=='fp').sum()} FP)")

    # session-level train/test split
    all_sess = [s for s, *_ in sessions]
    n_tr = max(1, int(round(len(all_sess) * 0.6)))
    train_sess, test_sess = set(all_sess[:n_tr]), set(all_sess[n_tr:])
    tr = det[det.session.isin(train_sess)]
    te = det[det.session.isin(test_sess)]
    print(f"\ntrain sessions: {sorted(train_sess)}")
    print(f"test  sessions: {sorted(test_sess)}")

    # 1. univariate ranking
    print("\n=== STAGE-2 detection features, ranked by TP/FP AUC (all sessions) ===")
    a2 = auc_table(det, ALL_FEATS)
    for r in a2.itertuples():
        arrow = "TP high" if r.higher_is_tp else "TP low "
        print(f"  {r.feature:13s} AUC={r.auc:.3f}  ({arrow}; TP med {r.tp_med:.3g} "
              f"vs FP med {r.fp_med:.3g})")

    print("\n=== STAGE-1 component features, ranked by TP/FP AUC ===")
    a1 = auc_table(comp, ["area_px", "aspect", "f_high_khz"])
    for r in a1.itertuples():
        arrow = "TP high" if r.higher_is_tp else "TP low "
        print(f"  {r.feature:13s} AUC={r.auc:.3f}  ({arrow}; TP med {r.tp_med:.3g} "
              f"vs FP med {r.fp_med:.3g})")

    # 2. multivariate model (train -> test AUC)
    Xtr = tr[ALL_FEATS].fillna(tr[ALL_FEATS].median()).values
    ytr = (tr.label == "tp").astype(int).values
    Xte = te[ALL_FEATS].fillna(tr[ALL_FEATS].median()).values
    yte = (te.label == "tp").astype(int).values

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logit.fit(Xtr, ytr)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0)
    tree.fit(Xtr, ytr)
    print("\n=== multivariate models (held-out test AUC) ===")
    print(f"  logistic regression : {roc_auc_score(yte, logit.predict_proba(Xte)[:,1]):.3f}")
    print(f"  decision tree (d=3) : {roc_auc_score(yte, tree.predict_proba(Xte)[:,1]):.3f}")
    imp = sorted(zip(ALL_FEATS, tree.feature_importances_), key=lambda x: -x[1])
    print("  tree feature importance:", ", ".join(f"{f}={w:.2f}" for f, w in imp if w > 0))
    print("\n  learned tree rule:\n" + export_text(tree, feature_names=ALL_FEATS))

    # 3. candidate filter strategies, evaluated at EVENT level with grouped k-fold CV
    def base(d):   # current stage-2 passes_mask_filters
        return (d.area_frac <= 0.15) & (d.n_cols >= 5) & (d.sweep_frac >= 0.04)

    names = list(all_sess)
    sess_by = {s[0]: s for s in sessions}
    fold_sets = [names[i::args.folds] for i in range(args.folds)]

    def tree_keep(trd, ted):
        X = trd[ALL_FEATS].fillna(trd[ALL_FEATS].median())
        y = (trd.label == "tp").astype(int)
        t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(X, y)
        Xt = ted[ALL_FEATS].fillna(trd[ALL_FEATS].median())
        return pd.Series(t.predict(Xt) == 1, index=ted.index)

    def fixed(cond):
        return lambda trd, ted: cond(ted)

    # threshold-based gates use fixed rules; the tree is refit each fold.
    strategies = {
        "raw\n(no filter)":              fixed(lambda d: pd.Series(True, index=d.index)),
        "current\ngeo filter":           fixed(base),
        "flatness\n<= 0.21":             fixed(lambda d: d.flatness <= 0.21),
        "learned tree\n(d=3)":           tree_keep,
        "n_cols\n>= 9":                  fixed(lambda d: d.n_cols >= 9),
        "n_cols>=9 &\ncentroid>=25kHz":  fixed(lambda d: (d.n_cols >= 9) & (d.centroid >= 25000)),
    }

    def cv_metrics(make_keep):
        R, P, F = [], [], []
        for te_names in fold_sets:
            trd = det[det.session.isin([n for n in names if n not in te_names])]
            ted = det[det.session.isin(te_names)]
            keep = make_keep(trd, ted).reindex(det.index, fill_value=False)
            m = event_metrics(det, keep, [sess_by[n] for n in te_names])
            R.append(m["recall"]); P.append(m["precision"]); F.append(m["f1"])
        return dict(recall=np.mean(R), recall_std=np.std(R),
                    precision=np.mean(P), precision_std=np.std(P),
                    f1=np.mean(F), f1_std=np.std(F))

    print(f"\n=== filter strategies, grouped {args.folds}-fold CV (event level) ===")
    print(f"  {'strategy':30s} {'recall':>13} {'prec':>13} {'F1':>13}")
    rows = []
    for name, mk in strategies.items():
        m = cv_metrics(mk)
        rows.append({"strategy": name.replace("\n", " "), **m})
        print(f"  {name.replace(chr(10),' '):30s} {m['recall']:.3f}±{m['recall_std']:.3f}"
              f"  {m['precision']:.3f}±{m['precision_std']:.3f}  {m['f1']:.3f}±{m['f1_std']:.3f}")

    pd.DataFrame(rows).to_csv(args.out_dir / "strategy_metrics.csv", index=False)
    a2.to_csv(args.out_dir / "feature_auc.csv", index=False)
    summary_figure(a2, list(strategies.keys()), rows, args.out_dir / "summary.png")
    print(f"\nwrote {args.out_dir}/detections.csv, components.csv, feature_auc.csv, "
          f"strategy_metrics.csv, summary.png")


if __name__ == "__main__":
    main()
