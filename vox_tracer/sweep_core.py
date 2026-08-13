"""Shared machinery for hyperparameter tuning over the SAM3 ridge pipeline.

Used by scripts/hparam_search/sweep_hparams.py (one-at-a-time PR sweep) and
scripts/hparam_search/random_search.py (random search). Holds ground-truth loading, 1-D
time-IoU scoring (mirrors scripts/evaluate.py), the GPU inference pass, and
prediction extraction for both the ridge and SAM3 stages.
"""
import math
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from vox_tracer.ridge import passes_mask_filters
from vox_tracer.sam3_runner import iter_sam3_windows
from vox_tracer.spec import group_specs_by_channel

# Config keys grouped by which pipeline stage consumes them.
STAGE1_KEYS = ("sigmas", "threshold_pct", "sample_rate", "freq_min",
               "min_area", "vert_aspect", "horiz_aspect", "close_kernel")
STAGE2_KEYS = ("max_mask_area_frac", "min_freq_sweep_frac", "min_mask_cols")


def stage1_of(cfg):
    return {k: cfg[k] for k in STAGE1_KEYS}


def stage2_of(cfg):
    return {k: cfg[k] for k in STAGE2_KEYS}


# ── ground truth + scoring (mirrors scripts/evaluate.py) ───────────────────────

def load_gt(recording_dir):
    """Return list of (t0, t1) vox ground-truth intervals from *annotations_gt.csv."""
    matches = sorted(glob(str(Path(recording_dir) / "*annotations_gt.csv")))
    if not matches:
        raise FileNotFoundError(f"No *annotations_gt.csv in {recording_dir}")
    df = pd.read_csv(matches[0])
    out = []
    for _, row in df.iterrows():
        if row["name"] != "vox":
            continue
        try:
            s, e = float(row["start_seconds"]), float(row["stop_seconds"])
        except (TypeError, ValueError):
            continue
        if not (math.isnan(s) or math.isnan(e)):
            out.append((s, e))
    return out


def iou_1d(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def match_1d(gt, pred, iou_thresh):
    """Greedy best-first 1-D IoU matching; returns (#matched_gt, #matched_pred)."""
    pairs = []
    for i, (gs, ge) in enumerate(gt):
        for j, (ps, pe) in enumerate(pred):
            iou = iou_1d(gs, ge, ps, pe)
            if iou > iou_thresh:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    for _, i, j in pairs:
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
    return len(used_gt), len(used_pred)


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    if math.isnan(p) or math.isnan(r) or (p + r) == 0:
        f = float("nan")
    else:
        f = 2 * p * r / (p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def score_all(preds_by_key, gt_by_session, sessions, channels, iou_thresh):
    """Aggregate TP/FP/FN across every (session, channel) into one metric dict."""
    tp = fp = fn = 0
    for rel, _, _ in sessions:
        gt = gt_by_session.get(rel, [])
        for ch in channels:
            preds = preds_by_key.get((rel, ch), [])
            m_gt, m_pred = match_1d(gt, preds, iou_thresh)
            tp += m_gt
            fn += len(gt) - m_gt
            fp += len(preds) - m_pred
    return metrics(tp, fp, fn)


# ── inference + prediction extraction ──────────────────────────────────────────

def gpu_pass(processor, sessions, channels, stage1, progress_desc=None):
    """Run the (expensive) ridge + SAM3 pass over the whole subset for one stage-1 config.

    Returns a list of window dicts (see iter_sam3_windows) tagged with session/ch.
    Pass progress_desc (e.g. "trial 3/40") to show a tqdm bar over windows; this
    is the long-running part of each trial, so the bar gives within-trial progress.
    """
    # List entries per (session, channel) up front so the bar has a total. This is
    # cheap (group_specs_by_channel just globs filenames).
    work = []  # (rel, ch, entries)
    for rel, spec_dir, _ in sessions:
        for ch, entries in group_specs_by_channel(spec_dir, channels).items():
            work.append((rel, ch, entries))
    total = sum(len(entries) for _, _, entries in work)

    windows = []
    with tqdm(total=total, desc=progress_desc, unit="win", leave=False,
              disable=progress_desc is None) as bar:
        for rel, ch, entries in work:
            for w in iter_sam3_windows(processor, entries, **stage1):
                w["session"] = rel
                w["ch"] = ch
                windows.append(w)
                bar.update(1)
    return windows


def _bbox_to_interval(x, w_box, window_start, window_end, img_width):
    dur = window_end - window_start
    return (window_start + (x / img_width) * dur,
            window_start + ((x + w_box) / img_width) * dur)


def sam3_preds(windows, stage2):
    """Apply post-hoc filters to cached SAM3 masks; return {(session,ch): [(t0,t1)]}."""
    out = defaultdict(list)
    for w in windows:
        if w["best_box"] is None:
            continue
        for mask_u8, _score in w["raw_masks"]:
            if not passes_mask_filters(mask_u8, w["H"], w["W"], stage2["max_mask_area_frac"],
                                       stage2["min_freq_sweep_frac"], stage2["min_mask_cols"]):
                continue
            x, _, w_box, _ = cv2.boundingRect((mask_u8 > 0).astype(np.uint8))
            out[(w["session"], w["ch"])].append(
                _bbox_to_interval(x, w_box, w["window_start"], w["window_end"], w["W"]))
    return out


def ridge_preds(windows):
    """Ridge-stage detections (all filtered components); return {(session,ch): [(t0,t1)]}."""
    out = defaultdict(list)
    for w in windows:
        n, _, stats, _ = cv2.connectedComponentsWithStats(w["seg_mask"])
        for lbl in range(1, n):
            x = stats[lbl, cv2.CC_STAT_LEFT]
            w_box = stats[lbl, cv2.CC_STAT_WIDTH]
            out[(w["session"], w["ch"])].append(
                _bbox_to_interval(x, w_box, w["window_start"], w["window_end"], w["W"]))
    return out


# ── session discovery ──────────────────────────────────────────────────────────

def discover_sessions(base):
    for exp_dir in sorted(Path(base).glob("experiment_*")):
        for idx_dir in sorted(exp_dir.glob("idx_*")):
            yield str(Path(exp_dir.name) / idx_dir.name)


def resolve_sessions(spec_base, recording_base, sessions_arg, n_sessions):
    """Return list of (rel, spec_dir, recording_dir) for the requested subset."""
    if sessions_arg:
        rels = [s.strip() for s in sessions_arg.split(",") if s.strip()]
    else:
        rels = list(discover_sessions(spec_base))
        if n_sessions:
            rels = rels[:n_sessions]
    out = []
    for rel in rels:
        spec_dir = Path(spec_base) / rel
        rec_dir = Path(recording_base) / rel
        if not spec_dir.exists():
            print(f"  skip {rel}: spec dir not found ({spec_dir})")
            continue
        out.append((rel, str(spec_dir), str(rec_dir)))
    return out


def load_gt_for_sessions(sessions):
    """Return {rel: [(t0,t1)]}; sessions with no CSV become empty (warned)."""
    gt_by_session = {}
    for rel, _, rec_dir in sessions:
        try:
            gt_by_session[rel] = load_gt(rec_dir)
        except FileNotFoundError as e:
            print(f"  WARNING {rel}: {e} (treated as 0 GT)")
            gt_by_session[rel] = []
    return gt_by_session
