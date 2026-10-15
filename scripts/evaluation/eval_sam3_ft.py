"""Score SAM3 "trace" detections (run_sam3_ft.py --images-json) against a fine-tuning dataset split.

Per results file: COCO bbox AP / AP50 (all scores), then at --score: precision / recall / F1 with
greedy box matching at IoU >= 0.5 and >= 0.3 (calls are small boxes), the share of empty windows
with a false detection, and mean mask IoU over matched GT boxes that have a mask label.

    python scripts/evaluation/eval_sam3_ft.py --gt outputs/sam3_ft/smoke/data/val/_annotations.coco.json \
        --pred zeroshot=preds_zs.json box=preds_box.json seg=preds_seg.json
"""
import argparse
import contextlib
import io
import json
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def rle(seg):
    return {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}


def coco_ap(gt, pred_path):
    """(AP, AP50) for bbox."""
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(pred_path)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0], ev.stats[1]


def match_stats(gt_json, preds, score, iou_thr):
    """Greedy per-image matching -> (tp, fp, fn, mask IoUs of matched GT with masks, empty-window FP share)."""
    gts, dts = defaultdict(list), defaultdict(list)
    for a in gt_json["annotations"]:
        gts[a["image_id"]].append(a)
    for d in preds:
        if d["score"] >= score:
            dts[d["image_id"]].append(d)
    tp = fp = fn = 0
    miou, empty, empty_fp = [], 0, 0
    for im in gt_json["images"]:
        g, d = gts[im["id"]], sorted(dts[im["id"]], key=lambda x: -x["score"])
        if not g:
            empty += 1
            empty_fp += bool(d)
        if not g or not d:
            fp, fn = fp + len(d), fn + len(g)
            continue
        ious = mask_util.iou([x["bbox"] for x in d], [x["bbox"] for x in g], [0] * len(g))
        used = set()
        for i in range(len(d)):
            j = next((j for j in np.argsort(-ious[i]) if j not in used and ious[i, j] >= iou_thr), None)
            if j is None:
                fp += 1
                continue
            used.add(j)
            tp += 1
            if "segmentation" in g[j] and "segmentation" in d[i]:
                miou.append(float(mask_util.iou([rle(d[i]["segmentation"])], [rle(g[j]["segmentation"])], [0])[0, 0]))
        fn += len(g) - len(used)
    return tp, fp, fn, miou, empty_fp / max(empty, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="name=results.json ...")
    ap.add_argument("--score", type=float, default=0.5)
    a = ap.parse_args()
    gt_json = json.load(open(a.gt))
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
    n_box = len(gt_json["annotations"])
    n_mask = sum("segmentation" in x for x in gt_json["annotations"])
    print(f"{a.gt}: {len(gt_json['images'])} windows, {n_box} GT boxes ({n_mask} with masks), score >= {a.score}")
    print(f"{'model':10s} {'AP':>5s} {'AP50':>5s} | {'P@.5':>5s} {'R@.5':>5s} {'F1@.5':>5s} | "
          f"{'P@.3':>5s} {'R@.3':>5s} {'F1@.3':>5s} | {'emptyFP':>7s} {'maskIoU':>7s} {'dets':>5s}")
    for spec in a.pred:
        name, path = spec.split("=", 1)
        preds = json.load(open(path))
        apv, ap50 = coco_ap(gt, path) if preds else (0.0, 0.0)
        row = [f"{name:10s} {apv:5.3f} {ap50:5.3f} |"]
        for thr in (0.5, 0.3):
            tp, fp, fn, miou, efp = match_stats(gt_json, preds, a.score, thr)
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            row.append(f"{p:5.3f} {r:5.3f} {2 * p * r / max(p + r, 1e-9):5.3f} |")
            if thr == 0.3:
                row.append(f"{efp:7.3f} {np.mean(miou) if miou else float('nan'):7.3f} "
                           f"{sum(d['score'] >= a.score for d in preds):5d}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
