"""
Visualize individual bouts: spectrogram + GT calls + predicted boxes/masks,
focused on the longest bouts found by scripts/evaluation/bout_analysis.py.

For each selected bout, stitches the spectrogram chunks spanning its time window
(+ padding) into one wide image per channel:
  - GT call bands: green = detected by >=1 prediction, red = missed
  - Predicted boxes/segmentation masks: cyan = overlaps a GT call in this bout,
    orange = does not (drawn only from the annotation's own chunk, so masks
    render exactly as the model produced them)

Which bouts get rendered is driven by bouts.csv and --sort-by:
  n_calls        (default) longest/densest bouts first
  worst_recall   lowest recall first (most calls missed), ties broken by n_calls desc
  worst_precision lowest precision first (most false positives), ties by n_calls desc
  worst_f1       lowest F1 first

Usage:
    python scripts/evaluation/viz_bouts.py data/gerbil_ssl outputs/spectrograms/gerbil_ssl outputs/sam3_best \
        outputs/eval/sam3_best/bouts.csv \
        --out-dir outputs/eval/sam3_best/bout_viz --sort-by worst_recall --top-n 8
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_tracer.montage import overlay_boxes, overlay_polygons
from vox_tracer.scoring import bbox_to_time, iou_1d, load_gt_from_csv, group_into_bouts
from vox_tracer.spec import group_specs_by_channel

COLOR_GT_HIT     = (0, 200, 0)     # green  — GT call detected by >=1 prediction
COLOR_GT_MISS    = (0, 0, 200)     # red    — GT call missed entirely
COLOR_PRED_TP    = (255, 255, 0)   # cyan   — prediction overlaps a GT call IN this bout
COLOR_PRED_OTHER = (180, 180, 180) # gray   — overlaps a GT call OUTSIDE this bout (context
                                   #          padding spilling into a neighboring bout — a
                                   #          real detection, just not this bout's)
COLOR_PRED_FP    = (0, 165, 255)   # orange — genuine false positive (no GT call anywhere nearby)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("recording_base", help="dir mirroring experiment_*/idx_*, each with *annotations_gt.csv")
parser.add_argument("spec_base", help="dir mirroring experiment_*/idx_* with spectrogram chunk PNGs")
parser.add_argument("pred_base", help="dir mirroring experiment_*/idx_* with coco_ch_<ch>.json predictions")
parser.add_argument("bouts_csv", help="bouts.csv written by scripts/evaluation/bout_analysis.py — selects which bouts to render")
parser.add_argument("--channels", default="118,35")
parser.add_argument("--iou-threshold", type=float, default=0.0)
parser.add_argument("--gap-threshold", type=float, default=0.25,
                    help="must match the value bouts.csv was generated with, so bout_id lines up")
parser.add_argument("--top-n", type=int, default=8, help="number of bouts to render")
parser.add_argument("--sort-by", default="n_calls",
                    choices=["n_calls", "worst_recall", "worst_precision", "worst_f1"],
                    help="which bouts to pick (default: n_calls, i.e. longest/densest first)")
parser.add_argument("--min-calls", type=int, default=1,
                    help="ignore bouts with fewer than this many calls (filters trivial size-1 misses "
                         "out of worst_recall/worst_precision selections)")
parser.add_argument("--pad", type=float, default=1.0, help="seconds of context padded on each side")
parser.add_argument("--max-width", type=int, default=3000, help="downscale output image if wider than this")
parser.add_argument("--out-dir", default="outputs/eval/bout_viz")
args = parser.parse_args()

channels = [int(c) for c in args.channels.split(",")]

def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


with open(args.bouts_csv) as f:
    bout_rows = list(csv.DictReader(f))
for r in bout_rows:
    r["n_calls"] = int(r["n_calls"]); r["duration"] = float(r["duration"])
    r["recall"] = _float(r["recall"]); r["precision"] = _float(r["precision"]); r["f1"] = _float(r["f1"])
bout_rows = [r for r in bout_rows if r["n_calls"] >= args.min_calls]

if args.sort_by == "n_calls":
    bout_rows.sort(key=lambda r: (r["n_calls"], r["duration"]), reverse=True)
else:
    metric = {"worst_recall": "recall", "worst_precision": "precision", "worst_f1": "f1"}[args.sort_by]
    # nan (undefined — e.g. no predictions at all for precision) sorts last: it isn't a
    # "worst" case in the same sense as a real low score, ties broken by more calls first
    bout_rows.sort(key=lambda r: (r[metric] if r[metric] == r[metric] else float("inf"), -r["n_calls"]))
selected = bout_rows[:args.top_n]

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

n_written = 0
for row in selected:
    session, bout_id = row["session"], int(row["bout_id"])
    rec_dir = Path(args.recording_base) / session
    try:
        gt, _combined = load_gt_from_csv(rec_dir)
    except FileNotFoundError:
        print(f"  {session}: GT not found, skipping"); continue

    bouts = group_into_bouts(gt, args.gap_threshold)
    if bout_id >= len(bouts):
        print(f"  {session} bout {bout_id}: out of range (gap-threshold mismatch?), skipping")
        continue
    bout = bouts[bout_id]
    b_start, b_end = bout[0][0], max(e for _, e in bout)
    w0, w1 = b_start - args.pad, b_end + args.pad

    spec_dir = Path(args.spec_base) / session
    pred_dir = Path(args.pred_base) / session

    # pool predictions across channels to decide which GT calls were detected at all
    pooled_preds = []
    for ch in channels:
        p = pred_dir / f"coco_ch_{ch}.json"
        if not p.exists():
            continue
        coco = json.load(open(p))
        img_by_id = {im["id"]: im for im in coco["images"]}
        for ann in coco["annotations"]:
            im = img_by_id[ann["image_id"]]
            t0, t1 = bbox_to_time(ann["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
            if t1 >= w0 and t0 <= w1:
                pooled_preds.append((t0, t1))

    gt_hit = [any(iou_1d(cs, ce, ps, pe) > args.iou_threshold for ps, pe in pooled_preds) for cs, ce in bout]

    channel_rows = []
    for ch in channels:
        chunks = sorted(
            (c for c in group_specs_by_channel(spec_dir, [ch]).get(ch, []) if c[2] > w0 and c[1] < w1),
            key=lambda c: c[1],
        )
        if not chunks:
            continue

        imgs, fname_offset = [], {}
        x = 0
        for p, t0, t1 in chunks:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            imgs.append((gray, t0, t1))
            fname_offset[p.name] = (x, t0, t1, gray.shape[1])
            x += gray.shape[1]
        if not imgs:
            continue

        h = max(g.shape[0] for g, _, _ in imgs)
        stitched = np.zeros((h, x, 3), dtype=np.uint8)
        cx = 0
        for gray, t0, t1 in imgs:
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            stitched[:bgr.shape[0], cx:cx + bgr.shape[1]] = bgr
            cx += bgr.shape[1]
        total_w = x

        def time_to_x(t):
            for fname, (ox, ct0, ct1, w) in fname_offset.items():
                if ct0 <= t < ct1:
                    return ox + (t - ct0) / (ct1 - ct0) * w
            return 0 if imgs and t < imgs[0][1] else total_w

        overlay = stitched.copy()
        for (cs, ce), hit in zip(bout, gt_hit):
            x0, x1 = int(time_to_x(cs)), int(time_to_x(ce))
            cv2.rectangle(overlay, (x0, 0), (max(x1, x0 + 1), h), COLOR_GT_HIT if hit else COLOR_GT_MISS, -1)
        stitched = cv2.addWeighted(overlay, 0.25, stitched, 0.75, 0)

        p = pred_dir / f"coco_ch_{ch}.json"
        if p.exists():
            coco = json.load(open(p))
            img_by_id = {im["id"]: im for im in coco["images"]}
            for ann in coco["annotations"]:
                im = img_by_id[ann["image_id"]]
                fname = im["file_name"]
                if fname not in fname_offset:
                    continue
                ox, ct0, ct1, _ = fname_offset[fname]
                bx, by, bw, bh = ann["bbox"]
                pt0, pt1 = bbox_to_time(ann["bbox"], ct0, ct1, im["width"])
                is_tp_bout = any(iou_1d(pt0, pt1, cs, ce) > args.iou_threshold for cs, ce in bout)
                if is_tp_bout:
                    color = COLOR_PRED_TP
                else:
                    is_tp_other = any(iou_1d(pt0, pt1, cs, ce) > args.iou_threshold for cs, ce in gt)
                    color = COLOR_PRED_OTHER if is_tp_other else COLOR_PRED_FP
                polys = ann.get("segmentation", [])
                if polys:
                    shifted = [[c + ox if i % 2 == 0 else c for i, c in enumerate(poly)] for poly in polys]
                    stitched = overlay_polygons(stitched, shifted, color)
                else:
                    stitched = overlay_boxes(stitched, [[bx + ox, by, bw, bh]], color)

        label = f"{session} ch{ch}  bout#{bout_id} n={row['n_calls']} [{b_start:.2f}-{b_end:.2f}s]"
        cv2.putText(stitched, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        channel_rows.append(stitched)

    if not channel_rows:
        continue
    maxw = max(r.shape[1] for r in channel_rows)
    padded = [r if r.shape[1] == maxw else
             np.hstack([r, np.zeros((r.shape[0], maxw - r.shape[1], 3), dtype=np.uint8)])
             for r in channel_rows]
    full = np.vstack(padded)
    if full.shape[1] > args.max_width:
        scale = args.max_width / full.shape[1]
        full = cv2.resize(full, (args.max_width, int(full.shape[0] * scale)))

    safe_session = session.replace("/", "_")
    out_path = out_dir / f"{safe_session}_bout{bout_id}_n{row['n_calls']}.png"
    cv2.imwrite(str(out_path), full)
    n_written += 1
    print(f"  wrote {out_path}  (n_calls={row['n_calls']}, recall={row['recall']}, precision={row['precision']})")

print(f"\n{n_written}/{len(selected)} bouts rendered → {out_dir}")
