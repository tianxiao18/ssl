"""
Thorough stage-1 tuning under a FIXED best stage-2 (n_cols>=9 & f_high>=28 kHz).

Answers the two things the shallow 5-config sweep did not:
  1. Do ANY features — geometric, the ridge response itself, or spectral — separate
     TP from FP at the *component* level (i.e. is there signal to gate on at stage-1)?
     Reported as ROC-AUC per feature (spectral on a bounded sample, since spectral
     on a 3-pixel fragment is meaningless and costly).
  2. Do stage-1 feature *combinations* beat the current stage-1? Curated gate-sets
     (current, +ridge-response, +solidity, both, aspect-free) are run through the
     full pipeline (component gate -> morph-close -> gate again -> stage-2) and scored
     at the event level with grouped k-fold CV.

Stage-1 gates must be evaluated in-pipeline (not post-hoc) because they act before
the morphological close and thus change the final masks.

    python scripts/ridge_stage1_tuning.py --n-sessions 6 --folds 4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))
import ridge_pipeline_viz as rpv  # noqa: E402
from ridge_filter_experiment import discover_sessions, event_metrics  # noqa: E402
from spectral_features import bbox_to_band, bbox_to_time, event_features  # noqa: E402
from vox_tracer.spec import load_channel_audio, parse_spec_fname  # noqa: E402

BASE = dict(rpv.DEFAULTS)
NYQ = BASE["sample_rate"] / 2.0
SPEC_TOPK = 10   # per chunk, compute spectral only for the K largest components


def components(response, H, W, thresh):
    """Connected components of the thresholded response, with geometric + response feats."""
    binary = (response >= thresh).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return labels, []
    means = ndimage.mean(response, labels, index=np.arange(1, n))
    comps = []
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                       cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]]
        comps.append(dict(lbl=lbl, x0=int(x), x1=int(x + w), y=int(y), w=int(w), h=int(h),
                          area=int(area), aspect=h / max(w, 1),
                          f_high=(H - 1 - y) / (H - 1) * NYQ, solidity=area / max(w * h, 1),
                          ridge=float(means[lbl - 1])))
    return labels, comps


def passes(c, gates):
    for feat, op, thr in gates:
        v = c[feat]
        if (op == ">=" and not v >= thr) or (op == "<=" and not v <= thr):
            return False
    return True


def ncols_per_label(labels, nlab, W):
    ys, xs = np.nonzero(labels)
    key = labels[ys, xs].astype(np.int64) * W + xs
    lbl = (np.unique(key) // W).astype(int)
    return np.bincount(lbl, minlength=nlab)


def run_custom(response, H, W, gates, wt0, wt1, vox):
    """Full pipeline with arbitrary stage-1 gates; return list of detection dicts."""
    thresh = np.percentile(response, BASE["threshold_pct"])
    labels, comps = components(response, H, W, thresh)
    kept = np.zeros((H, W), np.uint8)
    for c in comps:
        if passes(c, gates):
            kept[labels == c["lbl"]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(BASE["close_kernel"]))
    closed = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
    labels2, comps2 = components_from_mask(closed, response, H, W)
    final_lbls = [c["lbl"] for c in comps2 if passes(c, gates)]
    if not final_lbls:
        return []
    keepmask = np.isin(labels2, final_lbls)
    fl = np.where(keepmask, labels2, 0)
    nlab = labels2.max() + 1
    ncol = ncols_per_label(fl, nlab, W)
    dets = []
    for c in comps2:
        if c["lbl"] not in final_lbls:
            continue
        t0, t1 = bbox_to_time(c["x0"], c["w"], wt0, wt1, W)
        dets.append(dict(t0=t0, t1=t1, n_cols=int(ncol[c["lbl"]]),
                         f_high_khz=c["f_high"] / 1e3,
                         label="tp" if any(min(t1, e) > max(t0, s) for s, e in vox) else "fp",
                         ridge_score=c["ridge"]))
    return dets


def components_from_mask(mask, response, H, W):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return labels, []
    means = ndimage.mean(response, labels, index=np.arange(1, n))
    comps = []
    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                       cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]]
        comps.append(dict(lbl=lbl, x0=int(x), x1=int(x + w), y=int(y), w=int(w), h=int(h),
                          area=int(area), aspect=h / max(w, 1),
                          f_high=(H - 1 - y) / (H - 1) * NYQ, solidity=area / max(w * h, 1),
                          ridge=float(means[lbl - 1])))
    return labels, comps


# fixed best stage-2
def best_s2(d):
    return (d.n_cols >= 9) & (d.f_high_khz >= 28)


GATES = {
    "current":                [("area", ">=", 30), ("aspect", "<=", 5.0), ("aspect", ">=", 0.2), ("f_high", ">=", 20000.0)],
    "current+ridge":          [("area", ">=", 30), ("aspect", "<=", 5.0), ("aspect", ">=", 0.2), ("f_high", ">=", 20000.0), ("ridge", ">=", 0.05)],
    "current+solidity":       [("area", ">=", 30), ("aspect", "<=", 5.0), ("aspect", ">=", 0.2), ("f_high", ">=", 20000.0), ("solidity", "<=", 0.7)],
    "current+ridge+solidity": [("area", ">=", 30), ("aspect", "<=", 5.0), ("aspect", ">=", 0.2), ("f_high", ">=", 20000.0), ("ridge", ">=", 0.05), ("solidity", "<=", 0.7)],
    "area+freq (no aspect)":  [("area", ">=", 30), ("f_high", ">=", 20000.0)],
    "area+freq+ridge":        [("area", ">=", 30), ("f_high", ">=", 20000.0), ("ridge", ">=", 0.05)],
}

SPEC_NAMES = ["centroid", "bandwidth", "rolloff", "flatness", "rms", "zcr", "peak_freq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=6)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    sessions = discover_sessions(args.n_sessions)
    names = [s[0] for s in sessions]
    print(f"{len(sessions)} sessions")

    fn = rpv._build_filter_fn(BASE["filter_name"], list(BASE["sigmas"]))
    det_rows = {g: [] for g in GATES}
    comp_rows = []   # component-level features for AUC (geom for all, spectral for top-K)

    for session, spec_dir, data_dir, _, vox in sessions:
        for ch in (118, 35):
            loaded = load_channel_audio(data_dir, ch)
            sr, audio = (loaded[0], loaded[1]) if loaded else (None, None)
            for png in sorted(spec_dir.glob(f"headmic_{ch}_*.png")):
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                H, W = gray.shape
                resp = fn(gray.astype(np.float64) / 255.0)
                _, wt0, wt1 = parse_spec_fname(png.name)

                # in-pipeline gate-sets
                for g, gates in GATES.items():
                    for d in run_custom(resp, H, W, gates, wt0, wt1, vox):
                        d["session"] = session
                        det_rows[g].append(d)

                # component-level features for AUC diagnostic
                thresh = np.percentile(resp, BASE["threshold_pct"])
                _, comps = components(resp, H, W, thresh)
                comps = [c for c in comps if c["area"] >= 15]
                lab = lambda c: "tp" if any(  # noqa: E731
                    min(bbox_to_time(c["x0"], c["w"], wt0, wt1, W)[1], e)
                    > max(bbox_to_time(c["x0"], c["w"], wt0, wt1, W)[0], s) for s, e in vox) else "fp"
                topk = sorted(comps, key=lambda c: -c["area"])[:SPEC_TOPK]
                topk_ids = {id(c) for c in topk}
                for c in comps:
                    row = dict(label=lab(c), area=c["area"], aspect=c["aspect"],
                               f_high=c["f_high"], solidity=c["solidity"], ridge=c["ridge"])
                    if audio is not None and id(c) in topk_ids:
                        t0, t1 = bbox_to_time(c["x0"], c["w"], wt0, wt1, W)
                        fl, fh = bbox_to_band(c["y"], c["h"], H, sr / 2.0)
                        feats = event_features(audio, sr, t0, t1, fl, fh)
                        if feats:
                            for k, (m, _) in feats.items():
                                row[k] = m
                    comp_rows.append(row)

    comp = pd.DataFrame(comp_rows)
    y = (comp.label == "tp").astype(int).values
    print(f"\n=== component-level AUC (all features; spectral on {comp[SPEC_NAMES[0]].notna().sum()} "
          f"sampled comps of {len(comp)}) ===")
    feats = ["ridge", "solidity", "area", "aspect", "f_high"] + SPEC_NAMES
    auc_rows = []
    for f in feats:
        if f not in comp:
            continue
        v = comp[f].values.astype(float); ok = np.isfinite(v)
        if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
            continue
        a = roc_auc_score(y[ok], v[ok])
        auc_rows.append({"feature": f, "auc": max(a, 1 - a)})
        print(f"  {f:11s} AUC={max(a,1-a):.3f}  ({'TP high' if a>=0.5 else 'TP low '})")
    out = Path("outputs/eval/ridge_stage1_fromscratch"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(auc_rows).to_csv(out / "component_auc.csv", index=False)
    print(f"wrote {out}/component_auc.csv")

    # in-pipeline combination CV
    sess_by = {s[0]: s for s in sessions}
    fold_sets = [names[i::args.folds] for i in range(args.folds)]
    print(f"\n=== stage-1 gate-sets under fixed best stage-2, grouped {args.folds}-fold CV ===")
    print(f"  {'stage-1 gate-set':26s} {'F1':>14} {'recall':>13} {'precision':>13} {'kept':>7}")
    dfs = {g: pd.DataFrame(r) for g, r in det_rows.items()}
    for g, df in dfs.items():
        f1s, rcs, prs, kepts = [], [], [], []
        for te in fold_sets:
            sub = [sess_by[n] for n in te]
            keep = best_s2(df).fillna(False)
            m = event_metrics(df, keep, sub)
            f1s.append(m["f1"]); rcs.append(m["recall"]); prs.append(m["precision"])
            kepts.append(int((keep & df.session.isin(te)).sum()))
        print(f"  {g:26s} {np.mean(f1s):.3f}±{np.std(f1s):.3f}  {np.mean(rcs):.3f}±{np.std(rcs):.3f}"
              f"  {np.mean(prs):.3f}±{np.std(prs):.3f}  {int(np.mean(kepts)):7d}")


if __name__ == "__main__":
    main()
