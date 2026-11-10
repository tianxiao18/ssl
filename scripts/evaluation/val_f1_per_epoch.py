"""Val F1 per epoch of a SAM3 fine-tune run, scored like evaluate.py (score_combined: both mics pooled, any time overlap).

GT = the val split's own boxes (das_yolo or gold, the same labels val bbox AP uses). Boxes are clipped to their window
and turned into time intervals; GT from the two mics is merged into events (evaluate.py expects disjoint GT events).
Predictions with score >= --score (default 0.5, as at test time). Recording-bootstrap 95% CI as compare_sam3_ft.py.
Reads <run>/dumps/val/epoch_<N>.json (vox_tracer.sam3_ft trainer); older runs only kept the last epoch's
coco_predictions_bbox.json, scored as epoch --last-epoch.

    python scripts/evaluation/val_f1_per_epoch.py outputs/sam3_ft/runs/traces_dasyolo_all_v1 \
        --split-json outputs/sam3_ft/gerbil_ssl_dasyolo_all/val/_annotations.coco.json --last-epoch 5
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_sam3_ft import bootstrap, prf, write_csv  # noqa: E402
from vox_tracer.scoring import bbox_to_time, merge_intervals, score_combined  # noqa: E402


def intervals(anns, images, score=None):
    """{recording: [(t0, t1, score)]}, boxes clipped to their window, pooled over mics."""
    out = defaultdict(list)
    for a in anns:
        if score is not None and a.get("score", 0) < score:
            continue
        im = images[a["image_id"]]
        ws, we = im["window_start_sec"], im["window_end_sec"]
        t0, t1 = bbox_to_time(a["bbox"], ws, we, im["width"])
        t0, t1 = max(t0, ws), min(t1, we)
        if t1 > t0:
            out[re.sub(r"_ch\d+$", "", im["clip"])].append((t0, t1, a.get("score")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir with dumps/val/")
    ap.add_argument("--split-json", required=True, help="the run's val _annotations.coco.json")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--last-epoch", type=int, help="epoch of dumps/val/coco_predictions_bbox.json if no epoch_*.json")
    ap.add_argument("--out", help="csv (default <run>/logs/val_f1.csv)")
    a = ap.parse_args()

    split = json.load(open(a.split_json))
    images = {im["id"]: im for im in split["images"]}
    gt = {r: merge_intervals([(t0, t1) for t0, t1, _ in v]) for r, v in intervals(split["annotations"], images).items()}
    recs = sorted({re.sub(r"_ch\d+$", "", im["clip"]) for im in images.values()})

    dump_dir = Path(a.run) / "dumps" / "val"
    dumps = {int(p.stem.split("_")[1]): p for p in dump_dir.glob("epoch_*.json")}
    if not dumps and a.last_epoch:
        dumps = {a.last_epoch: dump_dir / "coco_predictions_bbox.json"}
    rows = []
    for ep in sorted(dumps):
        pred = intervals(json.load(open(dumps[ep])), images, a.score)
        per_rec = {}
        for r in recs:
            res = score_combined(gt.get(r, []), pred.get(r, []))
            per_rec[r] = np.array([res["tp"], res["pred_tp"], res["fp"], res["fn"]])
        tot = np.sum(list(per_rec.values()), axis=0)
        (p, rc, f1), (lo, hi) = prf(tot), bootstrap(per_rec)
        rows.append({"epoch": ep, "score": a.score, "recordings": len(recs), "gt_events": sum(map(len, gt.values())),
                     "tp": tot[0], "pred_tp": tot[1], "fp": tot[2], "fn": tot[3], "precision": p, "recall": rc,
                     "f1": f1, "f1_lo": lo[2], "f1_hi": hi[2]})
        print(f"epoch {ep:2d}: P {p:.3f} R {rc:.3f} F1 {f1:.3f} [{lo[2]:.3f}, {hi[2]:.3f}]")
    out = Path(a.out or Path(a.run) / "logs" / "val_f1.csv")
    write_csv(out, rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
