"""
Compare model predictions against ground-truth events from the annotations CSV.

Ground truth is loaded from *annotations_gt.csv in recording_dir — both 'vox'
and 'combined' rows are used.  das_dir still provides the DAS-YOLO COCO JSONs
for neutral YOLO-box visualization in montages.  Matching is done at recording
level by 1-D time-axis IoU, so chunk boundaries don't matter.

Outputs
-------
  metrics.csv          — one row per (session, channel) with TP/FP/FN/Recall/Precision/F1.
                         Re-run on any single session to drill into it visually.
  samples_tp/fp/fn/    — sampled montage pages (only with --montage-samples N)

Single recording:
    python scripts/evaluate.py <spec_dir> <das_dir> <recording_dir> <pred_dir> <out_dir> [options]

All recordings (mirrors experiment_*/idx_* structure):
    python scripts/evaluate.py <spec_base> <das_base> <recording_base> <pred_base> <out_base> --all [--workers 4]

Examples
--------
# Single recording — with montage samples to visually inspect TP/FP/FN
python scripts/evaluate.py \
    outputs/spectrograms/gerbil_ssl/experiment_384/idx_000 \
    outputs/das_yolo/experiment_384/idx_000 \
    data/gerbil_ssl/experiment_384/idx_000 \
    outputs/sam3_best/experiment_384/idx_000 \
    outputs/eval/sam3_best/experiment_384/idx_000 \
    --montage-samples 20 \
    --viz-session

# All recordings — sam3 predictions, 4 workers
python scripts/evaluate.py \
    outputs/spectrograms/gerbil_ssl \
    outputs/das_yolo \
    data \
    outputs/squeakout \
    outputs/eval/squeakout \
    --all --workers 4 \
    --gt-csv data/sampled_contiguous_detections.csv
"""
import argparse
import csv
import json
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import pandas as pd

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_tracer.montage import (
    make_label_strip, overlay_boxes, overlay_polygons, resize_cell, save_pages,
    viz_session_strip,
)
from vox_tracer.scoring import (
    iou_1d as _iou_1d,
    bbox_to_time as _bbox_to_time,
    merge_intervals as _merge_intervals,
    load_gt_from_csv as _load_gt_from_csv,
    load_gt_map as _load_gt_map,
    score_combined,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("spec_dir")
parser.add_argument("das_dir",       help="dir with das_yolo coco_ch_<ch>.json (YOLO boxes for visualization)")
parser.add_argument("recording_dir", help="dir with *annotations_gt.csv (source of GT intervals)")
parser.add_argument("pred_dir")
parser.add_argument("--gt-csv", default=None,
                    help="global detections CSV (e.g. data/sampled_contiguous_detections.csv). "
                         "When set, GT = is_vocalization=='yes' rows for the matching "
                         "experiment/idx instead of the per-recording *annotations_gt.csv. "
                         "For a *contiguous* CSV, the labeled (yes/no) rows define a covered "
                         "time span per recording; predictions outside it are ignored so that "
                         "FP/precision is only counted where GT coverage is complete.")
parser.add_argument("out_dir")
parser.add_argument("--channels",         default="118,35")
parser.add_argument("--cols",             type=int,   default=20)
parser.add_argument("--iou-threshold",    type=float, default=0.0,
                    help="minimum 1-D time IoU to count as a match (default: 0.0 = any overlap)")
parser.add_argument("--montage-samples",  type=int,   default=0,
                    help="if >0, sample this many TP/FP/FN chunks and write montages")
parser.add_argument("--seed",             type=int,   default=0)
parser.add_argument("--all",              action="store_true",
                    help="evaluate all experiment_*/idx_* dirs; mirror structure into out_dir")
parser.add_argument("--workers",          type=int,   default=4)
parser.add_argument("--viz-session",      action="store_true",
                    help="write full-session strip (raw | sato+exemplar | pred) for each recording")
args = parser.parse_args()

channels      = [int(c) for c in args.channels.split(",")]
iou_threshold = args.iou_threshold
random.seed(args.seed)

CSV_FIELDS = ["session", "channel", "n_gt", "n_pred", "tp", "pred_tp", "fp", "fn",
              "recall", "precision", "f1",
              "n_combined", "combined_tp", "combined_fn", "combined_recall"]

COLOR_TP_BAND   = (0,  200,   0)   # green band  — DAS event matched
COLOR_FN_BAND   = (0,    0, 200)   # red band    — DAS event missed
COLOR_YOLO      = (160, 160, 160)  # gray outline — YOLO box (spatial ref only)
COLOR_PRED_TP   = (255, 255,   0)  # cyan   — prediction matched a DAS event
COLOR_PRED_FP   = (0,  165, 255)   # orange — prediction matched nothing

ROW_LABELS = ["GT (DAS+YOLO)", "pred"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_coco(path):
    with open(path) as f:
        return json.load(f)


# GT loaders, geometry helpers, and the event-level scorer live in vox_tracer.scoring
# (imported above) so scripts/evaluation/pr_curves.py scores with the exact same logic.


# ── per-recording evaluation ──────────────────────────────────────────────────

def _evaluate_recording(spec_dir, das_dir, recording_dir, pred_dir, channels, iou_threshold, session,
                        gt_by_rec=None, coverage_by_rec=None):
    """
    Returns
    -------
    rows     : list of metric dicts (one per channel), ready for CSV
    examples : dict {'tp': [...], 'fp': [...], 'fn': [...]}
               each entry is a dict with img_path, gt_boxes, pred_polys,
               pred_boxes, label — enough to render a montage column
    """
    spec_dir      = Path(spec_dir)
    das_dir       = Path(das_dir)
    recording_dir = Path(recording_dir)
    pred_dir      = Path(pred_dir)

    rows     = []
    examples = {"tp": [], "fp": [], "fn": []}

    if gt_by_rec is not None:
        vox_intervals_all = gt_by_rec.get(session, [])
        combined_intervals_all = []
        # Coverage span defines where FP is trustworthy. No coverage → all rows for
        # this recording were blank/unlabeled, so we can't judge anything: skip.
        coverage = coverage_by_rec.get(session) if coverage_by_rec is not None else None
        if coverage is None:
            print(f"  {session}: no labeled coverage in CSV, skipping")
            return rows, examples
    else:
        coverage = None  # per-recording GT covers the whole recording — no filtering
        try:
            vox_intervals_all, combined_intervals_all = _load_gt_from_csv(recording_dir)
        except FileNotFoundError as e:
            print(f"  {session}: {e}, skipping")
            return rows, examples

    # ── Pass 1: gather per-channel GT-chunk mapping, YOLO boxes, and predictions ──
    # The GT (vox_intervals_all) is recording-level: the SAME events for every
    # channel — a vocalization is one physical event picked up by several mics.
    # So detection must be judged across channels combined, not per channel: an
    # event detected in ch118 but not ch35 is a true positive, not a ch35 FN.
    per_channel = []
    n_pred_dropped = 0
    for ch in channels:
        das_path  = das_dir  / f"coco_ch_{ch}.json"
        pred_path = pred_dir / f"coco_ch_{ch}.json"

        if not pred_path.exists():
            print(f"  {session} ch {ch}: pred not found, skipping")
            continue

        pred_coco = _load_coco(pred_path)
        # das_yolo COCO is only used for the YOLO-box/DAS-band montage overlay, not
        # for scoring (score_combined below matches GT directly against pred boxes).
        # Datasets without a DAS baseline (e.g. dryad_gerbil) fall back to pred's own
        # chunk windows so GT intervals can still be assigned to a chunk fname.
        if das_path.exists():
            das_coco = _load_coco(das_path)
        else:
            das_coco = {"images": pred_coco["images"], "annotations": []}

        das_img_by_id    = {im["id"]:        im for im in das_coco["images"]}
        das_img_by_fname = {im["file_name"]: im for im in das_coco["images"]}
        pred_img_by_id   = {im["id"]:        im for im in pred_coco["images"]}

        # Assign each vox GT interval to a chunk fname using das_coco image windows.
        # Index i aligns with vox_intervals_all, so it is comparable across channels.
        das_chunks = sorted(das_coco["images"], key=lambda im: im.get("window_start_sec", 0))
        das_intervals = []   # (t0, t1, fname)
        for t0, t1 in vox_intervals_all:
            mid = (t0 + t1) / 2
            fname = None
            for im in das_chunks:
                if im["window_start_sec"] <= mid < im["window_end_sec"]:
                    fname = im["file_name"]
                    break
            if fname is None and das_chunks:
                fname = (das_chunks[-1]["file_name"] if mid >= das_chunks[-1]["window_start_sec"]
                         else das_chunks[0]["file_name"])
            das_intervals.append((t0, t1, fname))

        pred_intervals = []  # (t0, t1, fname, ann)
        for ann in pred_coco["annotations"]:
            im    = pred_img_by_id[ann["image_id"]]
            fname = im["file_name"]
            t0, t1 = _bbox_to_time(ann["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
            # In coverage mode, ignore predictions outside the GT-complete span so
            # they can't be counted as (untrustworthy) FP.
            if coverage is not None and not (coverage[0] <= (t0 + t1) / 2 <= coverage[1]):
                n_pred_dropped += 1
                continue
            pred_intervals.append((t0, t1, fname, ann))

        das_ann_by_fname = {}
        for ann in das_coco["annotations"]:
            fname = das_img_by_id[ann["image_id"]]["file_name"]
            das_ann_by_fname.setdefault(fname, []).append(ann)

        per_channel.append({
            "ch":               ch,
            "das_img_by_fname": das_img_by_fname,
            "das_intervals":    das_intervals,
            "pred_intervals":   pred_intervals,
            "yolo_by_fname":    {fn: [a["bbox"] for a in anns]
                                 for fn, anns in das_ann_by_fname.items()},
        })

    if not per_channel:
        return rows, examples

    # ── Combine predictions across channels, then match against GT ──
    # Delegate to the shared event-level scorer (also used by scripts/evaluation/pr_curves.py).
    # score_threshold=None → keep all predictions (evaluate.py reports the operating
    # point as-run; the PR sweep is what varies the threshold).
    pooled_pred_boxes = [(t0, t1, ann.get("score"))
                         for pc in per_channel for (t0, t1, _, ann) in pc["pred_intervals"]]
    res = score_combined(vox_intervals_all, pooled_pred_boxes, iou_threshold)
    matched_gt, tp_events = res["matched_gt"], res["tp_events"]
    tp, fn, pred_tp, fp = res["tp"], res["fn"], res["pred_tp"], res["fp"]
    n_pred, n_boxes = res["n_pred"], res["n_boxes"]
    prec, rec, f1 = res["precision"], res["recall"], res["f1"]

    pooled_pred_times = [(t0, t1) for (t0, t1, _) in pooled_pred_boxes]

    # --- combined-window evaluation: TP = window has ≥1 pooled-pred overlap ---
    combined_tp = sum(
        1 for ct0, ct1 in combined_intervals_all
        if any(_iou_1d(ct0, ct1, pt0, pt1) > 0 for pt0, pt1 in pooled_pred_times)
    )
    combined_fn = len(combined_intervals_all) - combined_tp
    combined_rec = combined_tp / len(combined_intervals_all) if combined_intervals_all else float("nan")

    cov_note = f" (+{n_pred_dropped} pred outside coverage)" if coverage is not None else ""
    print(f"  {session} [{len(per_channel)}ch combined]: GT={len(vox_intervals_all)} "
          f"pred={n_pred}ev/{n_boxes}box{cov_note} "
          f"TP={tp} pred_TP={pred_tp} FP={fp} FN={fn} | Recall={rec:.3f} Prec={prec:.3f} F1={f1:.3f} | "
          f"Combined({len(combined_intervals_all)}): TP={combined_tp} FN={combined_fn} Recall={combined_rec:.3f}")

    rows.append({"session": session, "channel": "combined",
                 "n_gt": len(vox_intervals_all), "n_pred": n_pred,
                 "tp": tp, "pred_tp": pred_tp, "fp": fp, "fn": fn,
                 "recall": rec, "precision": prec, "f1": f1,
                 "n_combined": len(combined_intervals_all),
                 "combined_tp": combined_tp, "combined_fn": combined_fn,
                 "combined_recall": combined_rec})

    # ── Per-channel chunk tagging for montage rendering (uses combined match) ──
    # A GT event's TP/FN is the recording-level (combined) verdict, so it renders
    # the same in every channel — a chunk only lands in the FN pool if the event
    # was missed everywhere. A prediction's TP/FP is judged against the GT it lands on.
    for pc in per_channel:
        ch               = pc["ch"]
        das_intervals    = pc["das_intervals"]
        pred_intervals   = pc["pred_intervals"]
        das_img_by_fname = pc["das_img_by_fname"]
        yolo_by_fname    = pc["yolo_by_fname"]

        chunk_has           = {}  # fname -> set of outcome strings
        chunk_das_tp        = {}  # fname -> [(t0,t1)]  matched DAS events
        chunk_das_fn        = {}  # fname -> [(t0,t1)]  unmatched DAS events
        chunk_pred_tp_polys = {}; chunk_pred_tp_boxes = {}
        chunk_pred_fp_polys = {}; chunk_pred_fp_boxes = {}

        for i, (t0, t1, fname) in enumerate(das_intervals):
            if i in matched_gt:
                chunk_has.setdefault(fname, set()).add("tp")
                chunk_das_tp.setdefault(fname, []).append((t0, t1))
            else:
                chunk_has.setdefault(fname, set()).add("fn")
                chunk_das_fn.setdefault(fname, []).append((t0, t1))

        for (pt0, pt1, fname, ann) in pred_intervals:
            polys = ann.get("segmentation", [])
            bbox  = ann["bbox"]
            # Event-level verdict: this box is TP if its merged event overlaps GT
            # (a box overlaps only its own merged event, and those events are disjoint).
            is_tp = any(_iou_1d(pt0, pt1, es, ee) > 0 for es, ee in tp_events)
            if is_tp:
                chunk_has.setdefault(fname, set()).add("tp")
                chunk_pred_tp_polys.setdefault(fname, []).extend(polys)
                if not polys: chunk_pred_tp_boxes.setdefault(fname, []).append(bbox)
            else:
                chunk_has.setdefault(fname, set()).add("fp")
                chunk_pred_fp_polys.setdefault(fname, []).extend(polys)
                if not polys: chunk_pred_fp_boxes.setdefault(fname, []).append(bbox)

        for fname, outcome_set in chunk_has.items():
            im   = das_img_by_fname[fname]
            t0w, t1w = im.get("window_start_sec", ""), im.get("window_end_sec", "")
            label = (f"{session} ch{ch} t={t0w:.1f}-{t1w:.1f}s"
                     if isinstance(t0w, float) else fname)
            ex = {
                "img_path":        str(spec_dir / fname),
                "das_tp":          chunk_das_tp.get(fname, []),
                "das_fn":          chunk_das_fn.get(fname, []),
                "yolo_boxes":      yolo_by_fname.get(fname, []),
                "window_start":    im["window_start_sec"],
                "window_end":      im["window_end_sec"],
                "img_width":       im["width"],
                "img_height":      im["height"],
                "pred_tp_polys":   chunk_pred_tp_polys.get(fname, []),
                "pred_tp_boxes":   chunk_pred_tp_boxes.get(fname, []),
                "pred_fp_polys":   chunk_pred_fp_polys.get(fname, []),
                "pred_fp_boxes":   chunk_pred_fp_boxes.get(fname, []),
                "label":           label,
            }
            has_tp = "tp" in outcome_set
            has_fn = "fn" in outcome_set
            has_fp = "fp" in outcome_set
            # Strict pools: TP = all DAS events matched (no FN); FN = any missed event;
            # FP = spurious predictions with no missed events (no FN)
            if has_fn:
                examples["fn"].append(ex)
            elif has_tp:
                examples["tp"].append(ex)
            if has_fp and not has_fn:
                examples["fp"].append(ex)

    return rows, examples


def _draw_das_bands(img, das_events, window_start, window_end, color, alpha=0.35):
    """Draw semi-transparent vertical bands for each DAS time interval."""
    dur = window_end - window_start
    h, w = img.shape[:2]
    out = img.copy()
    for t0, t1 in das_events:
        x0 = max(0, int((t0 - window_start) / dur * w))
        x1 = min(w, int((t1 - window_start) / dur * w))
        if x1 > x0:
            overlay = out.copy()
            cv2.rectangle(overlay, (x0, 0), (x1, h), color, -1)
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    return out


def _render_column(ex):
    gray = cv2.imread(ex["img_path"], cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ws, we = ex["window_start"], ex["window_end"]

    # GT row: colored bands for DAS events, gray outlines for YOLO boxes
    gt_img = bgr.copy()
    gt_img = _draw_das_bands(gt_img, ex["das_fn"], ws, we, COLOR_FN_BAND, alpha=0.25)
    gt_img = _draw_das_bands(gt_img, ex["das_tp"], ws, we, COLOR_TP_BAND, alpha=0.18)
    gt_img = overlay_boxes(gt_img, ex["yolo_boxes"], COLOR_YOLO)

    # Pred row: cyan = TP prediction, orange = FP prediction
    pred_img = bgr.copy()
    if ex["pred_tp_polys"]:
        pred_img = overlay_polygons(pred_img, ex["pred_tp_polys"], COLOR_PRED_TP)
    else:
        pred_img = overlay_boxes(pred_img, ex["pred_tp_boxes"], COLOR_PRED_TP)
    if ex["pred_fp_polys"]:
        pred_img = overlay_polygons(pred_img, ex["pred_fp_polys"], COLOR_PRED_FP)
    else:
        pred_img = overlay_boxes(pred_img, ex["pred_fp_boxes"], COLOR_PRED_FP)

    return np.vstack([resize_cell(gt_img, label=ex["label"]),
                      resize_cell(pred_img)])


def _write_montage(examples, n, out_dir, prefix, cols, label_strip):
    sampled = random.sample(examples, min(n, len(examples)))
    columns = [c for c in (_render_column(ex) for ex in sampled) if c is not None]
    if columns:
        save_pages(columns, out_dir, prefix=prefix, cols_per_page=cols, label_strip=label_strip)
        print(f"  wrote {len(columns)} {prefix} examples → {out_dir}")


def _discover(base):
    for exp_dir in sorted(Path(base).glob("experiment_*")):
        for idx_dir in sorted(exp_dir.glob("idx_*")):
            yield Path(exp_dir.name) / idx_dir.name, idx_dir


# ── main ─────────────────────────────────────────────────────────────────────

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# GT source switch: --gt-csv → global detections CSV; otherwise per-recording CSVs.
gt_by_rec, coverage_by_rec = _load_gt_map(args.gt_csv) if args.gt_csv else (None, None)
if gt_by_rec is not None:
    n_vox = sum(len(v) for v in gt_by_rec.values())
    print(f"GT: {n_vox} vocalizations across {len(gt_by_rec)} recordings; "
          f"{len(coverage_by_rec)} recordings with labeled coverage (FP-evaluable) "
          f"from {args.gt_csv}")

all_rows     = []
all_examples = {"tp": [], "fp": [], "fn": []}

if not args.all:
    session = f"{Path(args.spec_dir).parent.name}/{Path(args.spec_dir).name}"
    rows, examples = _evaluate_recording(
        args.spec_dir, args.das_dir, args.recording_dir, args.pred_dir,
        channels, iou_threshold, session, gt_by_rec, coverage_by_rec,
    )
    all_rows.extend(rows)
    for k in all_examples:
        all_examples[k].extend(examples[k])

else:
    spec_base      = Path(args.spec_dir)
    das_base       = Path(args.das_dir)
    recording_base = Path(args.recording_dir)
    pred_base      = Path(args.pred_dir)

    tasks = [
        (str(idx_dir), str(das_base / rel), str(recording_base / rel), str(pred_base / rel), str(rel))
        for rel, idx_dir in _discover(spec_base)
    ]
    print(f"Found {len(tasks)} recordings, {args.workers} worker(s) …")

    def _worker(t):
        spec, das, recording, pred, session = t
        return _evaluate_recording(spec, das, recording, pred, channels, iou_threshold, session,
                                   gt_by_rec, coverage_by_rec)

    if args.workers == 1:
        results = []
        for t in tasks:
            try:
                results.append(_worker(t))
            except Exception as e:
                print(f"ERROR {t[0]}: {e}")
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"ERROR {futures[fut][0]}: {e}")

    for rows, examples in results:
        all_rows.extend(rows)
        for k in all_examples:
            all_examples[k].extend(examples[k])

# --- aggregate summary ---
all_tp      = sum(r["tp"] for r in all_rows)        # GT events detected (recall numerator)
all_pred_tp = sum(r["pred_tp"] for r in all_rows)   # correct predictions (precision numerator)
all_fp      = sum(r["fp"] for r in all_rows)
all_fn      = sum(r["fn"] for r in all_rows)
p  = all_pred_tp / (all_pred_tp + all_fp) if (all_pred_tp + all_fp) > 0 else float("nan")
r  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else float("nan")
f1 = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
print(f"\nOverall ({len(all_rows)} channel-sessions): "
      f"TP={all_tp} pred_TP={all_pred_tp} FP={all_fp} FN={all_fn} | "
      f"Recall={r:.3f} Prec={p:.3f} F1={f1:.3f}")

all_ctp = sum(r["combined_tp"] for r in all_rows)
all_cfn = sum(r["combined_fn"] for r in all_rows)
all_cn  = sum(r["n_combined"]  for r in all_rows)
crec = all_ctp / all_cn if all_cn > 0 else float("nan")
print(f"  Combined({all_cn}): TP={all_ctp} FN={all_cfn} | Recall={crec:.3f}")
print(f"  TP examples pool: {len(all_examples['tp'])}  "
      f"FP: {len(all_examples['fp'])}  FN: {len(all_examples['fn'])}")

# --- write CSV ---  (name reflects GT source so gt-csv runs don't overwrite the default)
csv_name = f"metrics_{Path(args.gt_csv).stem}.csv" if args.gt_csv else "metrics.csv"
csv_path = out_dir / csv_name
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"  metrics → {csv_path}")

# --- write montage samples ---
if args.montage_samples > 0:
    label_strip = make_label_strip(ROW_LABELS)
    for outcome in ("tp", "fp", "fn"):
        pool = all_examples[outcome]
        if not pool:
            print(f"  no {outcome} examples to sample")
            continue
        _write_montage(pool, args.montage_samples, out_dir / f"samples_{outcome}",
                       prefix="page", cols=args.cols, label_strip=label_strip)

# --- session strip ---
if args.viz_session:
    if not args.all:
        viz_session_strip(args.spec_dir, args.pred_dir, out_dir / "session_strip", channels)
    else:
        for rel, idx_dir in _discover(Path(args.spec_dir)):
            viz_session_strip(str(idx_dir), str(Path(args.pred_dir) / rel),
                              out_dir / rel / "session_strip", channels)

