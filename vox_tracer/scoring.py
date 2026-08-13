"""
Combined-channel, event-level scoring — the single source of truth shared by
scripts/evaluate.py (bar-chart metrics) and scripts/evaluation/pr_curves.py (PR sweeps).

Model of the task
-----------------
A vocalization is one physical event on the recording timeline, picked up by
several mics (channels). So predictions from all channels are pooled onto the
shared time axis and unioned into disjoint predicted *events* — a call heard on
two mics, or over-segmented into several boxes, becomes one event. Ground-truth
rows are already curated events (discrete, disjoint), so GT is left as-is.

Matching is many-to-many by 1-D time overlap, which yields two distinct TP
counts (see score_combined):
  * recall side  — a GT event is TP if it overlaps >=1 predicted event
  * precision side — a predicted event is TP if it overlaps >=1 GT event

A per-detection ``score`` (confidence) may be swept: passing ``score_threshold``
drops predictions below it *before* pooling/merging, which is exactly how a
precision-recall curve is traced from a single permissive run.
"""
import json
import math
from glob import glob
from pathlib import Path

import pandas as pd


# ── geometry helpers ────────────────────────────────────────────────────────────

def iou_1d(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def bbox_to_time(bbox, window_start, window_end, img_width):
    x, _, w, _ = bbox
    dur = window_end - window_start
    return window_start + (x / img_width) * dur, window_start + ((x + w) / img_width) * dur


def merge_intervals(intervals):
    """Union temporally overlapping/touching (start, stop) intervals into disjoint events."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# ── the scorer ──────────────────────────────────────────────────────────────────

def score_combined(gt_intervals, pred_boxes, iou_threshold=0.0, score_threshold=None):
    """Score one recording's pooled predictions against its GT events.

    Parameters
    ----------
    gt_intervals : [(start, stop)]         GT events (already discrete; not merged).
    pred_boxes   : [(start, stop, score)]  predictions pooled across ALL channels;
                                           ``score`` may be None if the model exports none.
    iou_threshold : float                  min 1-D time IoU to count an overlap
                                           (0.0 = any overlap, strict ``>``).
    score_threshold : float or None        if set, drop pred_boxes with score < threshold
                                           (a box with score None is dropped, since it
                                           cannot be thresholded).

    Returns a dict with tp/fn/pred_tp/fp, n_pred (merged events), n_boxes (raw),
    recall/precision/f1, plus matched_gt / pred_events / tp_events for callers that
    need to render or introspect the matching.
    """
    if score_threshold is not None:
        pred_boxes = [b for b in pred_boxes if b[2] is not None and b[2] >= score_threshold]

    pooled = [(t0, t1) for (t0, t1, _) in pred_boxes]

    # recall side — GT event detected if any raw prediction overlaps it
    matched_gt = set()
    for i, (gs, ge) in enumerate(gt_intervals):
        if any(iou_1d(gs, ge, ps, pe) > iou_threshold for ps, pe in pooled):
            matched_gt.add(i)
    tp = len(matched_gt)
    fn = len(gt_intervals) - tp

    # precision side — predictions merged into events, event correct if it overlaps any GT
    pred_events = merge_intervals(pooled)
    tp_events = [(es, ee) for es, ee in pred_events
                 if any(iou_1d(es, ee, gs, ge) > iou_threshold for gs, ge in gt_intervals)]
    n_boxes = len(pooled)
    n_pred = len(pred_events)
    pred_tp = len(tp_events)
    fp = n_pred - pred_tp

    prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

    return {"tp": tp, "fn": fn, "pred_tp": pred_tp, "fp": fp,
            "n_pred": n_pred, "n_boxes": n_boxes,
            "recall": rec, "precision": prec, "f1": f1,
            "matched_gt": matched_gt, "pred_events": pred_events, "tp_events": tp_events}


# ── bout analysis ───────────────────────────────────────────────────────────────
# A bout is a run of GT calls whose inter-call gaps (end of one call to start of
# the next) are all <= gap_threshold. Bouts are derived from GT only — they
# describe animal behavior, not model output.

def group_into_bouts(gt_intervals, gap_threshold):
    """Group GT (start, stop) intervals into bouts by inter-call-interval.

    Returns a list of bouts, each a time-sorted list of (start, stop) tuples.
    A bout of size 1 is an isolated call (no neighbor within gap_threshold).
    """
    if not gt_intervals:
        return []
    ordered = sorted(gt_intervals)
    bouts = [[ordered[0]]]
    for cur in ordered[1:]:
        prev_end = bouts[-1][-1][1]
        if cur[0] - prev_end <= gap_threshold:
            bouts[-1].append(cur)
        else:
            bouts.append([cur])
    return bouts


def score_bouts(gt_intervals, pred_boxes, gap_threshold, iou_threshold=0.0, score_threshold=None):
    """Per-bout recall/precision: group GT into bouts, score each against the
    predictions that overlap its time span.

    A prediction is attributed to a bout if it overlaps [bout_start, bout_end]
    (the span of the bout's own calls) — so precision reflects only predictions
    plausibly meant for that bout. A prediction straddling two bouts separated by
    slightly more than gap_threshold can be double-counted; this is a rare edge
    case at the scale bouts are normally analyzed.

    Returns a list of dicts, one per bout: n_calls, start, stop, duration,
    mean_ici (nan for size-1 bouts), plus score_combined's tp/fn/pred_tp/fp/
    recall/precision/f1 restricted to that bout, and n_pred_events/merge_ratio
    (see below).

    merge_ratio = n_pred_events / n_calls, where n_pred_events is the count of
    *merged* predicted events overlapping the bout (score_combined's n_pred —
    raw boxes unioned where touching/overlapping). Boolean event-overlap recall
    can't tell a well-resolved bout from one where several calls were fused into
    a single predicted blob that happens to touch all of them; merge_ratio << 1
    is exactly that failure mode (fewer predicted blobs than GT calls), while
    ~1 means predictions are individuating calls one-for-one and >1 means
    over-segmentation (or extra FPs) rather than merging.
    """
    if score_threshold is not None:
        pred_boxes = [b for b in pred_boxes if b[2] is not None and b[2] >= score_threshold]

    results = []
    for bout in group_into_bouts(gt_intervals, gap_threshold):
        b_start = bout[0][0]
        b_end = max(e for _, e in bout)
        overlapping = [p for p in pred_boxes if p[0] < b_end and p[1] > b_start]
        res = score_combined(bout, overlapping, iou_threshold)
        icis = [bout[i + 1][0] - bout[i][1] for i in range(len(bout) - 1)]
        n_calls = len(bout)
        results.append({
            "n_calls": n_calls, "start": b_start, "stop": b_end, "duration": b_end - b_start,
            "mean_ici": sum(icis) / len(icis) if icis else float("nan"),
            **{k: res[k] for k in ("tp", "fn", "pred_tp", "fp", "recall", "precision", "f1")},
            "n_pred_events": res["n_pred"],
            "merge_ratio": res["n_pred"] / n_calls if n_calls > 0 else float("nan"),
        })
    return results


# ── prediction / GT loaders ───────────────────────────────────────────────────────

def read_pred_boxes(pred_dir, channels, coverage=None):
    """Pool one recording's predictions across channels onto the time axis.

    Returns [(start, stop, score)] with score = annotation['score'] or None.
    In coverage mode, predictions whose midpoint falls outside the labeled span
    are dropped (FP is only trustworthy where GT coverage is complete).
    """
    boxes = []
    for ch in channels:
        p = Path(pred_dir) / f"coco_ch_{ch}.json"
        if not p.exists():
            continue
        with open(p) as f:
            coco = json.load(f)
        img_by_id = {im["id"]: im for im in coco["images"]}
        for ann in coco["annotations"]:
            im = img_by_id[ann["image_id"]]
            t0, t1 = bbox_to_time(ann["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
            if coverage is not None and not (coverage[0] <= (t0 + t1) / 2 <= coverage[1]):
                continue
            boxes.append((t0, t1, ann.get("score")))
    return boxes


def load_gt_map(csv_path):
    """(gt_by_rec, coverage_by_rec) from a global contiguous-detections CSV.

    gt_by_rec       : {session -> [(start, stop)]} of is_vocalization=='yes' events.
    coverage_by_rec : {session -> (cov_start, cov_stop)} over labeled (yes/no) rows —
                      the region where GT is complete, so FP there is trustworthy.
    Sessions are keyed 'experiment_<e>/idx_<i:03d>' to match the pred-dir layout.
    """
    df = pd.read_csv(csv_path)
    gt_by_rec, coverage_by_rec = {}, {}
    for _, row in df.iterrows():
        try:
            s, e = float(row["start_seconds"]), float(row["stop_seconds"])
            if math.isnan(s) or math.isnan(e):
                continue
        except (TypeError, ValueError):
            continue
        label = str(row["is_vocalization"]).strip().lower()
        if label not in ("yes", "no"):
            continue
        session = f"experiment_{int(row['experiment'])}/idx_{int(row['idx']):03d}"
        cs, ce = coverage_by_rec.get(session, (s, e))
        coverage_by_rec[session] = (min(cs, s), max(ce, e))
        if label == "yes":
            gt_by_rec.setdefault(session, []).append((s, e))
    return gt_by_rec, coverage_by_rec


def load_gt_from_csv(recording_dir):
    """(vox_intervals, combined_intervals) from a per-recording *annotations_gt.csv."""
    matches = sorted(glob(str(Path(recording_dir) / "*annotations_gt.csv")))
    if not matches:
        raise FileNotFoundError(f"No *annotations_gt.csv found in {recording_dir}")
    df = pd.read_csv(matches[0])
    vox_intervals, combined_intervals = [], []
    for _, row in df.iterrows():
        try:
            s, e = float(row["start_seconds"]), float(row["stop_seconds"])
            if math.isnan(s) or math.isnan(e):
                continue
        except (TypeError, ValueError):
            continue
        if row["name"] == "vox":
            vox_intervals.append((s, e))
        elif row["name"] == "combined":
            combined_intervals.append((s, e))
    return vox_intervals, combined_intervals
