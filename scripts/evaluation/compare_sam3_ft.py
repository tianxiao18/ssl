"""Compare SAM3 "trace" models on the fine-tuning val recordings (used as the test set).

  detection  full recordings, both mics pooled, scored per recording with the evaluate.py scorer
             (score_combined, any time overlap) against each annotator's corrected GT CSV
             ('vox' + 'combined' rows); predictions centred outside the labeled span are ignored.
             Recording-bootstrap 95% intervals.  Inputs: --det name=<pred base>[@score], laid out
             <base>/<experiment_X/idx_Y>/coco_ch_<ch>.json (run_sam3_ft.py --split-json, or any detector).
  box, mask  on the dataset's val windows (run_sam3_ft.py --images-json): greedy one-to-one GT/pred
             matching by box IoU (> 0) per window; box IoU, onset/offset (ms) and f_lo/f_hi (kHz) error,
             and mask IoU vs the GT mask (a model mask, see build_sam3_ft_dataset.py) where one exists.
             Inputs: --win name=<results.json>[@score]. sam3_best is left out here: most GT boxes and
             masks are its own outputs.
Writes detection.csv, box_mask.csv, quantitative.png and qualitative.png to --out.

    python scripts/evaluation/compare_sam3_ft.py --split-json outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json \
        --det zero-shot=outputs/sam3_zeroshot_trace/gerbil_ssl@0.2 box2mask=outputs/sam3_ft_box2mask/gerbil_ssl \
              seg=outputs/sam3_ft_head/gerbil_ssl sam3_best=outputs/sam3_best/gerbil_ssl@0 \
        --win zero-shot=preds/zeroshot.json@0.2 box2mask=preds/box2mask.json seg=preds/seg.json \
        --out outputs/eval/sam3_ft_val
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_tracer.scoring import read_pred_boxes, score_combined  # noqa: E402

NYQ = 62500.0
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GT_COLOR = "#ffffff"
GT_ROOT = "/mnt/ceph/users/neurostatslab/saneslab_data/vox_detection/vox_detection_gt"


def spec(s, default):
    """'name=path[@score]' -> (name, path, score)."""
    name, rest = s.split("=", 1)
    path, _, thr = rest.partition("@")
    return name, path, float(thr) if thr else default


def rel_of(clip):
    m = re.search(r"exp(\d+)_idx(\d+)", clip)
    return f"experiment_{int(m[1])}/idx_{int(m[2]):03d}"


# ── detection (full recordings) ───────────────────────────────────────────────

def load_gt(gt_root, annotator, recs, coverage):
    """{rec: ([(start, stop)], [(cov_start, cov_stop)])} for the recordings this annotator labeled."""
    cov = coverage[coverage.annotator == annotator]
    out = {}
    for f in sorted(Path(gt_root, annotator, "corrected").glob("sampled_exp*_idx*_annotations.csv")):
        rel = rel_of(f.name)
        if rel not in recs:
            continue
        df = pd.read_csv(f).dropna(subset=["start_seconds", "stop_seconds"])
        df = df[df["name"].isin(["vox", "combined"])]
        spans = [tuple(x) for x in cov[cov["clip"].map(rel_of) == rel][["start", "stop"]].to_numpy()]
        if spans:
            out[rel] = (list(zip(df.start_seconds, df.stop_seconds)), spans)
    return out


def detect_rows(base, thr, gt, channels):
    """{rec: [tp, pred_tp, fp, fn]}, predictions with score >= thr centred in the labeled span."""
    rows = {}
    for rel, (events, spans) in gt.items():
        if not (Path(base) / rel).exists():
            print(f"  missing predictions: {Path(base) / rel}")
            continue
        boxes = [b for b in read_pred_boxes(Path(base) / rel, channels)
                 if (b[2] is None or b[2] >= thr) and any(s <= (b[0] + b[1]) / 2 <= e for s, e in spans)]
        r = score_combined(events, boxes)
        rows[rel] = np.array([r["tp"], r["pred_tp"], r["fp"], r["fn"]])
    return rows


def prf(c):
    tp, pred_tp, fp, fn = c
    p = pred_tp / (pred_tp + fp) if pred_tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return np.array([p, r, 2 * p * r / (p + r) if p + r > 0 else float("nan")])


def bootstrap(rows, n=2000, seed=0):
    """95% recording-bootstrap interval of (P, R, F1): array (2, 3)."""
    m = np.array(list(rows.values()))
    rng = np.random.default_rng(seed)
    draws = np.array([prf(m[rng.integers(len(m), size=len(m))].sum(0)) for _ in range(n)])
    return np.nanpercentile(draws, [2.5, 97.5], axis=0)


# ── box / mask (val windows) ──────────────────────────────────────────────────

def rle(seg):
    return {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}


def group(anns):
    g = defaultdict(list)
    for a in anns:
        g[a["image_id"]].append(a)
    return g


def matches(gt, preds):
    """[(gt ann, pred, box IoU)] greedy one-to-one per window, highest IoU first."""
    gts, dts = group(gt["annotations"]), group(preds)
    out = []
    for iid, g in gts.items():
        d = dts[iid]
        if not d:
            continue
        iou = mask_util.iou([x["bbox"] for x in d], [x["bbox"] for x in g], [0] * len(g))
        used_d, used_g = set(), set()
        for i, j in zip(*np.unravel_index(np.argsort(-iou, axis=None), iou.shape)):
            if iou[i, j] <= 0:
                break
            if i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            out.append((g[j], d[i], float(iou[i, j])))
    return out


def box_mask_rows(gt, preds):
    ims = {im["id"]: im for im in gt["images"]}
    rows = []
    for a, d, biou in matches(gt, preds):
        im = ims[a["image_id"]]
        ms = (im["window_end_sec"] - im["window_start_sec"]) / im["width"] * 1e3
        khz = NYQ / (im["height"] - 1) / 1e3
        (ax, ay, aw, ah), (dx, dy, dw, dh) = a["bbox"], d["bbox"]
        miou = (float(mask_util.iou([rle(d["segmentation"])], [rle(a["segmentation"])], [0])[0, 0])
                if "segmentation" in a and "segmentation" in d else float("nan"))
        rows.append({"box_iou": biou, "mask_iou": miou,
                     "onset_ms": abs(dx - ax) * ms, "offset_ms": abs(dx + dw - ax - aw) * ms,
                     "f_hi_khz": abs(dy - ay) * khz, "f_lo_khz": abs(dy + dh - ay - ah) * khz})
    return rows


# ── figures ───────────────────────────────────────────────────────────────────

def quantitative(det, win, annotators, color, out):
    n_ax = len(annotators) + 2 * bool(win)
    fig, ax = plt.subplots(1, n_ax, figsize=(4.6 * n_ax, 3.9), facecolor="#ffffff", squeeze=False)
    ax = ax[0]
    x = np.arange(3)
    for a_i, ann in enumerate(annotators):
        names = [n for n in det if ann in det[n]]
        w = 0.8 / len(names)
        for k, n in enumerate(names):
            v, ci = det[n][ann]["prf"], det[n][ann]["ci"]
            xs = x + (k - (len(names) - 1) / 2) * w
            ax[a_i].bar(xs, v, w * 0.92, color=color[n], label=n)
            ax[a_i].errorbar(xs, v, yerr=[v - ci[0], ci[1] - v], fmt="none", ecolor="#1a1a19", lw=0.8, capsize=2)
            for xi, vi in zip(xs, v):
                ax[a_i].text(xi, 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=7, color="#ffffff")
        ax[a_i].set_xticks(x, ["precision", "recall", "F1"])
        ax[a_i].set(ylim=(0, 1.12), title=f"detection vs {ann} GT")
    if win:
        for a, key, title in ((ax[-2], "box_iou", "box IoU"), (ax[-1], "mask_iou", "mask IoU")):
            names = list(win)
            data = [[r[key] for r in win[n]["rows"] if not np.isnan(r[key])] for n in names]
            bp = a.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": "#1a1a19"})
            for k, patch in enumerate(bp["boxes"]):
                patch.set(facecolor=color[names[k]], alpha=0.35, edgecolor=color[names[k]])
            for k, vals in enumerate(data):
                a.scatter(np.random.default_rng(k).normal(k + 1, 0.06, len(vals)), vals, s=5,
                          color=color[names[k]], alpha=0.5)
            a.set_xticks(range(1, len(names) + 1), names)
            a.set(ylim=(0, 1.02), title=title)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
        a.set_axisbelow(True)
    handles = {lab: h for a in ax[:len(annotators)] for h, lab in zip(*a.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=max(len(handles), 1), frameon=False,
               fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor="#ffffff")
    plt.close(fig)


def draw(ax, img, anns, color, crop):
    """Spectrogram crop (x0, y0, x1, y1) with mask fill + outline and dashed box of each ann."""
    x0, y0, x1, y1 = crop
    rgb = np.repeat(img[..., None], 3, axis=2).astype(float) / 255
    c = np.array(matplotlib.colors.to_rgb(color))
    masks = [mask_util.decode(rle(a["segmentation"])) for a in anns if "segmentation" in a]
    for m in masks:
        rgb[m.astype(bool)] = 0.5 * rgb[m.astype(bool)] + 0.5 * c
    ax.imshow(rgb[y0:y1, x0:x1], interpolation="nearest", extent=(x0, x1, y1, y0))
    for m in masks:
        ax.contour(np.arange(img.shape[1]), np.arange(img.shape[0]), m, levels=[0.5], colors=[color], linewidths=0.8)
    for a in anns:
        x, y, bw, bh = a["bbox"]
        ax.add_patch(plt.Rectangle((x, y), bw, bh, fill=False, ec=color, lw=0.8, ls="--"))
    ax.set(xlim=(x0, x1), ylim=(y1, y0), xticks=[], yticks=[])


def qualitative(gt, img_dir, win, color, n, seed, out):
    """One column per randomly drawn GT call (with mask), zoomed; rows GT then each model."""
    ims = {im["id"]: im for im in gt["images"]}
    rng = np.random.default_rng(seed)
    cand = [a for a in gt["annotations"] if "segmentation" in a]
    calls = [cand[i] for i in sorted(rng.choice(len(cand), min(n, len(cand)), replace=False))]
    rows = [("GT", GT_COLOR, group(gt["annotations"]))] + [(nm, color[nm], group(w["preds"])) for nm, w in win.items()]
    fig, axs = plt.subplots(len(rows), len(calls), figsize=(1.6 * len(calls), 1.6 * len(rows)),
                            facecolor="#ffffff", squeeze=False)
    for j, a in enumerate(calls):
        im = ims[a["image_id"]]
        img = cv2.imread(str(img_dir / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        x, y, bw, bh = a["bbox"]
        half = max(bw, bh) / 2 + 12
        cx = min(max(x + bw / 2, half), im["width"] - half)
        cy = min(max(y + bh / 2, half), im["height"] - half)
        crop = tuple(int(v) for v in (cx - half, cy - half, cx + half, cy + half))
        for i, (nm, c, anns) in enumerate(rows):
            draw(axs[i, j], img, anns[im["id"]], c, crop)
            if j == 0:
                axs[i, j].set_ylabel(nm, fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.995, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.savefig(out, dpi=150, facecolor="#ffffff")
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-json", required=True, help="dataset split: its recordings and windows are scored")
    ap.add_argument("--det", nargs="*", default=[], help="name=<recording pred base>[@score] ...")
    ap.add_argument("--win", nargs="*", default=[], help="name=<window results.json>[@score] ...")
    ap.add_argument("--score", type=float, default=0.5, help="default score threshold")
    ap.add_argument("--gt-root", default=GT_ROOT)
    ap.add_argument("--annotators", default="niegil,tianxiao")
    ap.add_argument("--coverage", default="outputs/pseudo_boxes/gerbil_ssl_corrected_gt_coverage.csv")
    ap.add_argument("--channels", default="118,35")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    split = json.load(open(a.split_json))
    recs = {rel_of(im["clip"]) for im in split["images"]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dets = [spec(s, a.score) for s in a.det]
    wins = [spec(s, a.score) for s in a.win]
    names = list(dict.fromkeys([d[0] for d in dets] + [w[0] for w in wins]))
    color = {n: COLORS[k % len(COLORS)] for k, n in enumerate(names)}
    annotators = a.annotators.split(",")
    channels = [int(c) for c in a.channels.split(",")]

    det, det_table = {}, []
    if dets:
        coverage = pd.read_csv(a.coverage)
        gts = {ann: load_gt(a.gt_root, ann, recs, coverage) for ann in annotators}
        for ann, g in gts.items():
            print(f"{ann}: {len(g)} recordings, {sum(len(v[0]) for v in g.values())} GT events")
        for nm, base, thr in dets:
            det[nm] = {}
            for ann, g in gts.items():
                rows = detect_rows(base, thr, g, channels)
                if not rows:
                    continue
                c = np.sum(list(rows.values()), axis=0)
                v, (lo, hi) = prf(c), bootstrap(rows, seed=a.seed)
                det[nm][ann] = {"prf": v, "ci": np.array([lo, hi])}
                row = {"model": nm, "score": thr, "gt": ann, "recordings": len(rows),
                       **dict(zip(("tp", "pred_tp", "fp", "fn"), map(int, c)))}
                for k, m in enumerate(("precision", "recall", "f1")):
                    row.update({m: v[k], f"{m}_lo": lo[k], f"{m}_hi": hi[k]})
                det_table.append(row)
        write_csv(out / "detection.csv", det_table)
        print(f"{'model':10s} {'score':>5s} {'gt':9s} {'P':>5s} {'R':>5s} {'F1':>5s}  F1 95%")
        for r in det_table:
            print(f"{r['model']:10s} {r['score']:5.2f} {r['gt']:9s} {r['precision']:5.3f} {r['recall']:5.3f} "
                  f"{r['f1']:5.3f}  [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]")

    win, bm_table = {}, []
    for nm, path, thr in wins:
        preds = [d for d in json.load(open(path)) if d["score"] >= thr]
        rows = box_mask_rows(split, preds)
        win[nm] = {"preds": preds, "rows": rows}
        mi = [x["mask_iou"] for x in rows if not np.isnan(x["mask_iou"])]
        bm_table.append({"model": nm, "score": thr, "gt_boxes": len(split["annotations"]), "boxes_matched": len(rows),
                         **{k: float(np.median([x[k] for x in rows])) if rows else float("nan")
                            for k in ("box_iou", "onset_ms", "offset_ms", "f_lo_khz", "f_hi_khz")},
                         "masks_matched": len(mi), "mask_iou": float(np.mean(mi)) if mi else float("nan")})
    if wins:
        write_csv(out / "box_mask.csv", bm_table)
        print(f"\n{'model':10s} {'score':>5s} {'matched':>9s} {'boxIoU':>6s} {'on ms':>5s} {'off ms':>6s} "
              f"{'flo kHz':>7s} {'fhi kHz':>7s} | {'nmask':>5s} {'maskIoU':>7s}")
        for t in bm_table:
            print(f"{t['model']:10s} {t['score']:5.2f} {t['boxes_matched']:4d}/{t['gt_boxes']:<4d} {t['box_iou']:6.3f} "
                  f"{t['onset_ms']:5.1f} {t['offset_ms']:6.1f} {t['f_lo_khz']:7.2f} {t['f_hi_khz']:7.2f} | "
                  f"{t['masks_matched']:5d} {t['mask_iou']:7.3f}")
        qualitative(split, Path(a.split_json).parent, win, color, a.n_examples, a.seed, out / "qualitative.png")
    quantitative(det, win, [ann for ann in annotators if any(ann in v for v in det.values())], color,
                 out / "quantitative.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
