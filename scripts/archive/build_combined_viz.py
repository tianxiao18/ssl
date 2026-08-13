"""Visualize how `combined` GT events are evaluated for sam3_best.

Replicates the combined-window scoring from scripts/evaluate.py exactly:
  combined_tp  = window overlapped by >=1 pooled prediction box (raw, IoU>0)
  combined_fn  = window with no overlapping prediction (MISSED)
Predictions that overlap NO combined window are "spurious" relative to the
combined windows (the combined path itself never counts these, but we surface
them for the visualization the user asked for).

Pools predictions across channels 118 & 35 like the evaluator does.
"""
import json, sys
from pathlib import Path

ROOT = Path("/mnt/home/the10/ssl")
sys.path.insert(0, str(ROOT))
from vox_tracer.scoring import iou_1d, bbox_to_time, load_gt_from_csv

PRED_BASE = ROOT / "outputs/sam3_best"
DATA_BASE = ROOT / "data"
CHANNELS = [118, 35]
PAD = 1.0  # seconds of context to keep around each combined window


def pooled_boxes(pred_dir):
    boxes = []
    for ch in CHANNELS:
        p = pred_dir / f"coco_ch_{ch}.json"
        if not p.exists():
            continue
        coco = json.load(open(p))
        img = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            im = img[a["image_id"]]
            t0, t1 = bbox_to_time(a["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
            boxes.append((round(t0, 4), round(t1, 4), ch))
    return boxes


windows = []          # one entry per combined GT window
n_tp = n_fn = 0
recs_with_combined = 0
for exp in sorted(PRED_BASE.glob("experiment_*")):
    for idx in sorted(exp.glob("idx_*")):
        session = f"{exp.name}/{idx.name}"
        try:
            vox, combined = load_gt_from_csv(DATA_BASE / exp.name / idx.name)
        except FileNotFoundError:
            continue
        if not combined:
            continue
        recs_with_combined += 1
        boxes = pooled_boxes(idx)
        for (c0, c1) in combined:
            lo, hi = c0 - PAD, c1 + PAD
            near = [[b0, b1, ch, int(iou_1d(c0, c1, b0, b1) > 0)]
                    for (b0, b1, ch) in boxes if b1 >= lo and b0 <= hi]
            matched = any(hit for *_, hit in near)
            n_tp += matched
            n_fn += not matched
            windows.append({
                "session": session,
                "c0": round(c0, 4), "c1": round(c1, 4),
                "lo": round(lo, 4), "hi": round(hi, 4),
                "matched": int(matched),
                "near": near,
                "n_overlap": sum(hit for *_, hit in near),
            })

out = ROOT / "outputs/eval/sam3_best/combined_windows.json"
payload = {"n_tp": n_tp, "n_fn": n_fn, "n_windows": len(windows),
           "recs": recs_with_combined, "windows": windows}
json.dump(payload, open(out, "w"))
recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else float("nan")
print(f"recordings with combined events: {recs_with_combined}")
print(f"combined windows: {len(windows)}  TP={n_tp}  FN(missed)={n_fn}  recall={recall:.3f}")
print(f"multi-window recordings: {sum(1 for _ in [] )}")
print(f"wrote {out} ({out.stat().st_size/1e3:.0f} KB)")
