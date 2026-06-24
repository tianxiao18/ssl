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
    outputs/spectrograms/experiment_445/idx_011 \
    outputs/das_yolo/experiment_445/idx_011 \
    data/experiment_445/idx_011 \
    outputs/sam3/experiment_445/idx_011 \
    outputs/eval/sam3/experiment_445/idx_011 \
    --montage-samples 20 \
    --viz-session

# All recordings — sam3 predictions, 4 workers
python scripts/evaluate.py \
    outputs/spectrograms \
    outputs/das_yolo \
    data \
    outputs/sam3 \
    outputs/eval/sam3 \
    --all --workers 4
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("spec_dir")
parser.add_argument("das_dir",       help="dir with das_yolo coco_ch_<ch>.json (YOLO boxes for visualization)")
parser.add_argument("recording_dir", help="dir with *annotations_gt.csv (source of GT intervals)")
parser.add_argument("pred_dir")
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

CSV_FIELDS = ["session", "channel", "n_gt", "n_pred", "tp", "fp", "fn",
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


def _load_gt_from_csv(recording_dir):
    """Return (vox_intervals, combined_intervals) from *annotations_gt.csv."""
    matches = sorted(glob(str(Path(recording_dir) / "*annotations_gt.csv")))
    if not matches:
        raise FileNotFoundError(f"No *annotations_gt.csv found in {recording_dir}")
    df = pd.read_csv(matches[0])
    vox_intervals      = []
    combined_intervals = []
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


def _iou_1d(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def _bbox_to_time(bbox, window_start, window_end, img_width):
    x, _, w, _ = bbox
    dur = window_end - window_start
    return window_start + (x / img_width) * dur, window_start + ((x + w) / img_width) * dur


def _match_1d(das_intervals, pred_intervals, iou_thresh):
    """Sort all overlapping pairs by IoU descending, then greedily assign best-first."""
    pairs = []
    for i, (ds, de) in enumerate(das_intervals):
        for j, (ps, pe) in enumerate(pred_intervals):
            iou = _iou_1d(ds, de, ps, pe)
            if iou > iou_thresh:  # strict > so threshold=0.0 still requires real overlap
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_das, used_pred = set(), set()
    matched_das, matched_pred = [], []
    for _, i, j in pairs:
        if i not in used_das and j not in used_pred:
            matched_das.append(i); matched_pred.append(j)
            used_das.add(i);       used_pred.add(j)
    return matched_das, matched_pred


# ── per-recording evaluation ──────────────────────────────────────────────────

def _evaluate_recording(spec_dir, das_dir, recording_dir, pred_dir, channels, iou_threshold, session):
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

    try:
        vox_intervals_all, combined_intervals_all = _load_gt_from_csv(recording_dir)
    except FileNotFoundError as e:
        print(f"  {session}: {e}, skipping")
        return rows, examples

    for ch in channels:
        das_path  = das_dir  / f"coco_ch_{ch}.json"
        pred_path = pred_dir / f"coco_ch_{ch}.json"

        if not das_path.exists():
            print(f"  {session} ch {ch}: das_yolo COCO not found, skipping")
            continue
        if not pred_path.exists():
            print(f"  {session} ch {ch}: pred not found, skipping")
            continue

        das_coco  = _load_coco(das_path)
        pred_coco = _load_coco(pred_path)

        das_img_by_id    = {im["id"]:        im for im in das_coco["images"]}
        das_img_by_fname = {im["file_name"]: im for im in das_coco["images"]}
        pred_img_by_id   = {im["id"]:        im for im in pred_coco["images"]}

        # Assign each vox GT interval to a chunk fname using das_coco image windows
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
        pred_ann_by_fname = {}
        for ann in pred_coco["annotations"]:
            im    = pred_img_by_id[ann["image_id"]]
            fname = im["file_name"]
            t0, t1 = _bbox_to_time(ann["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
            pred_intervals.append((t0, t1, fname, ann))
            pred_ann_by_fname.setdefault(fname, []).append(ann)

        # --- match at recording level ---
        matched_das, matched_pred = _match_1d(
            [(t0, t1) for t0, t1, _ in das_intervals],
            [(t0, t1) for t0, t1, _, _ in pred_intervals],
            iou_threshold,
        )
        matched_das_set  = set(matched_das)
        matched_pred_set = set(matched_pred)

        tp = len(matched_das)
        fn = len(das_intervals) - tp
        fp = len(pred_intervals) - len(matched_pred)

        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

        # --- combined evaluation: TP = combined window has ≥1 pred overlap ---
        pred_time_intervals = [(t0, t1) for t0, t1, _, _ in pred_intervals]
        combined_tp = sum(
            1 for ct0, ct1 in combined_intervals_all
            if any(_iou_1d(ct0, ct1, pt0, pt1) > 0 for pt0, pt1 in pred_time_intervals)
        )
        combined_fn = len(combined_intervals_all) - combined_tp
        combined_rec = combined_tp / len(combined_intervals_all) if combined_intervals_all else float("nan")

        print(f"  {session} ch {ch}: GT={len(das_intervals)} pred={len(pred_intervals)} "
              f"TP={tp} FP={fp} FN={fn} | Recall={rec:.3f} Prec={prec:.3f} F1={f1:.3f} | "
              f"Combined({len(combined_intervals_all)}): TP={combined_tp} FN={combined_fn} Recall={combined_rec:.3f}")

        rows.append({"session": session, "channel": ch,
                     "n_gt": len(das_intervals), "n_pred": len(pred_intervals),
                     "tp": tp, "fp": fp, "fn": fn,
                     "recall": rec, "precision": prec, "f1": f1,
                     "n_combined": len(combined_intervals_all),
                     "combined_tp": combined_tp, "combined_fn": combined_fn,
                     "combined_recall": combined_rec})

        # --- per-chunk outcome info for coloured rendering ---
        chunk_has           = {}  # fname -> set of outcome strings
        chunk_das_tp        = {}  # fname -> [(t0,t1)]  matched DAS events
        chunk_das_fn        = {}  # fname -> [(t0,t1)]  unmatched DAS events
        chunk_yolo_boxes    = {}  # fname -> [bbox]     all YOLO boxes (neutral ref)
        chunk_pred_tp_polys = {}; chunk_pred_tp_boxes = {}
        chunk_pred_fp_polys = {}; chunk_pred_fp_boxes = {}

        # Tag chunks directly from DAS events — covers ALL events regardless of YOLO
        for i, (t0, t1, fname) in enumerate(das_intervals):
            if i in matched_das_set:
                chunk_has.setdefault(fname, set()).add("tp")
                chunk_das_tp.setdefault(fname, []).append((t0, t1))
            else:
                chunk_has.setdefault(fname, set()).add("fn")
                chunk_das_fn.setdefault(fname, []).append((t0, t1))

        # Collect all YOLO boxes per chunk (shown as neutral spatial reference)
        das_ann_by_fname = {}
        for ann in das_coco["annotations"]:
            fname = das_img_by_id[ann["image_id"]]["file_name"]
            das_ann_by_fname.setdefault(fname, []).append(ann)
        for fname, anns in das_ann_by_fname.items():
            chunk_yolo_boxes[fname] = [a["bbox"] for a in anns]

        # Tag chunks from predictions
        for j, (_, _, fname, ann) in enumerate(pred_intervals):
            polys = ann.get("segmentation", [])
            bbox  = ann["bbox"]
            if j in matched_pred_set:
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
                "yolo_boxes":      chunk_yolo_boxes.get(fname, []),
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

all_rows     = []
all_examples = {"tp": [], "fp": [], "fn": []}

if not args.all:
    session = f"{Path(args.spec_dir).parent.name}/{Path(args.spec_dir).name}"
    rows, examples = _evaluate_recording(
        args.spec_dir, args.das_dir, args.recording_dir, args.pred_dir,
        channels, iou_threshold, session,
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
        return _evaluate_recording(spec, das, recording, pred, channels, iou_threshold, session)

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
all_tp = sum(r["tp"] for r in all_rows)
all_fp = sum(r["fp"] for r in all_rows)
all_fn = sum(r["fn"] for r in all_rows)
p  = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else float("nan")
r  = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else float("nan")
f1 = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
print(f"\nOverall ({len(all_rows)} channel-sessions): "
      f"TP={all_tp} FP={all_fp} FN={all_fn} | "
      f"Recall={r:.3f} Prec={p:.3f} F1={f1:.3f}")

all_ctp = sum(r["combined_tp"] for r in all_rows)
all_cfn = sum(r["combined_fn"] for r in all_rows)
all_cn  = sum(r["n_combined"]  for r in all_rows)
crec = all_ctp / all_cn if all_cn > 0 else float("nan")
print(f"  Combined({all_cn}): TP={all_ctp} FN={all_cfn} | Recall={crec:.3f}")
print(f"  TP examples pool: {len(all_examples['tp'])}  "
      f"FP: {len(all_examples['fp'])}  FN: {len(all_examples['fn'])}")

# --- write CSV ---
csv_path = out_dir / "metrics.csv"
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

