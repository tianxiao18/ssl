"""SAM3 fine-tuning dataset from the corrected-GT pseudo boxes: 1 s spectrogram windows + COCO boxes.

Images are the existing 1 s PNGs of each clip's annotated mic that overlap the clip's labeled span.
  positive  clean GT row with a box, clipped to the window; fragments < MIN_FRAG_SEC become ignore.
  ignore    rows with status 'ignore' (unmatched, combined, overlapping) and short fragments:
            their time columns are blacked out in the written image (no positive, no negative).
  unlabeled window parts outside the labeled span are blacked out the same way.
Split is by recording (experiment/idx), so both mics of a recording land in the same split.
Windows whose visible (labeled, non-ignore) time is < --min-labeled of the window are dropped.
Train windows without boxes are subsampled to --neg-ratio x the number of windows with boxes; val keeps all.
Each box gets a mask (COCO RLE) from the model that produced it: its polygons on the same PNG window,
cut to the box. das_yolo boxes (no polygons) or empty cuts get no mask; SAM3 then skips them in the
mask loss but still trains box/score on them.

    python scripts/dataset_prep/build_sam3_ft_dataset.py --dry-run
    python scripts/dataset_prep/build_sam3_ft_dataset.py --out outputs/sam3_ft/gerbil_ssl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

MIN_FRAG_SEC = 0.01   # window-clipped box fragments shorter than this become ignore regions
NYQ = 62500.0
MASK_DIRS = {"sam3": "outputs/sam3_best", "sam3_best": "outputs/sam3_best",
             "ridge": "outputs/ridge", "squeakout": "outputs/squeakout"}


def rel_of(clip):
    m = re.search(r"exp(\d+)_idx(\d+)", clip)
    return f"experiment_{m[1]}/idx_{int(m[2]):03d}"


def windows(spec_dir, ch):
    """[(png, t0, t1)] for one mic, sorted by time."""
    out = []
    for p in Path(spec_dir).glob(f"headmic_{ch}_*_t*.png"):
        m = re.search(r"_t([\d.]+)-([\d.]+)\.png$", p.name)
        out.append((p, float(m[1]), float(m[2])))
    return sorted(out, key=lambda w: w[1])


def build(d, cov, spec_root, stats):
    """Yield (clip, rel, png, t0, t1, boxes[(s, e, f_lo, f_hi, source)], ignores[(s, e)])."""
    for clip, g in d.groupby("clip"):
        rel, ch = rel_of(clip), int(g.channel.iloc[0])
        spans = cov[cov["clip"] == clip][["start", "stop"]].to_numpy()
        for r in g.itertuples():
            if not ((spans[:, 0] <= r.start + 1e-6) & (spans[:, 1] >= r.stop - 1e-6)).any():
                stats["rows: outside labeled span"] += 1
        wins = windows(Path(spec_root) / rel, ch)
        covered = set()
        for png, t0, t1 in wins:
            if not ((spans[:, 0] < t1) & (spans[:, 1] > t0)).any():
                continue
            boxes, ignores = [], []
            lab = [(max(s, t0), min(e, t1)) for s, e in spans if s < t1 and e > t0]
            edges = [t0] + [x for iv in sorted(lab) for x in iv] + [t1]
            ignores += [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a > 1e-6]
            for r in g[(g.start < t1) & (g.stop > t0)].itertuples():
                s, e = max(r.start, t0), min(r.stop, t1)
                if r.status != "clean":
                    ignores.append((s, e))
                elif e - s < MIN_FRAG_SEC:
                    ignores.append((s, e))
                    stats["fragments: < 10 ms piece at window edge -> ignore"] += 1
                else:
                    boxes.append((s, e, r.f_lo, r.f_hi, r.source))
                    covered.add(r.Index)
                    if (s, e) != (r.start, r.stop):
                        stats["fragments: clipped at window edge, kept"] += 1
            yield clip, rel, png, t0, t1, boxes, ignores
        for r in g.itertuples():
            if r.status == "clean" and r.Index not in covered:
                stats["rows: clean but no window (no PNG yet / outside span)"] += 1


def masked_frac(ignores, t0, t1):
    """Fraction of [t0, t1] covered by the union of the ignore intervals."""
    tot, end = 0.0, t0
    for s, e in sorted(ignores):
        s, e = max(s, end), min(e, t1)
        if e > s:
            tot, end = tot + e - s, e
    return tot / (t1 - t0)


def to_px(s, e, lo, hi, t0, t1, w, h):
    x0, x1 = (s - t0) / (t1 - t0) * w, (e - t0) / (t1 - t0) * w
    y0, y1 = (1 - hi / NYQ) * (h - 1), (1 - lo / NYQ) * (h - 1)
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]


def model_polys(src, rel, ch, fname, cache):
    """[segmentation polygons] of `src`'s detections on PNG window `fname` (empty if none / no masks)."""
    if src not in MASK_DIRS:
        return []
    path = Path(MASK_DIRS[src]) / "gerbil_ssl" / rel / f"coco_ch_{ch}.json"
    if path not in cache:
        cache.clear()
        by_img = {}
        if path.exists():
            c = json.load(open(path))
            names = {im["id"]: im["file_name"] for im in c["images"]}
            for an in c["annotations"]:
                if an.get("segmentation"):
                    by_img.setdefault(names[an["image_id"]], []).extend(an["segmentation"])
        cache[path] = by_img
    return [q for q in cache[path].get(fname, []) if len(q) >= 6]


def box_mask(polys, bb, h, w):
    """RLE of the polygons' union cut to box bb (xywh px), or None if empty."""
    if not polys:
        return None
    m = mask_util.decode(mask_util.merge(mask_util.frPyObjects(polys, h, w)))
    x0, y0 = int(np.floor(bb[0])), int(np.floor(bb[1]))
    x1, y1 = int(np.ceil(bb[0] + bb[2])), int(np.ceil(bb[1] + bb[3]))
    cut = np.zeros_like(m)
    cut[y0:y1 + 1, x0:x1 + 1] = m[y0:y1 + 1, x0:x1 + 1]
    if not cut.any():
        return None
    rle = mask_util.encode(np.asfortranarray(cut))
    return {"size": rle["size"], "counts": rle["counts"].decode()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", default="outputs/pseudo_boxes/gerbil_ssl_corrected_gt_models.csv")
    ap.add_argument("--coverage", default="outputs/pseudo_boxes/gerbil_ssl_corrected_gt_coverage.csv")
    ap.add_argument("--spec-root", default="outputs/spectrograms/gerbil_ssl")
    ap.add_argument("--out", default="outputs/sam3_ft/gerbil_ssl")
    ap.add_argument("--prompt", default="trace", help="category name = SAM3 text prompt")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-labeled", type=float, default=0.0, help="drop windows less visible than this")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="train empty windows per box window")
    ap.add_argument("--max-windows", type=int, nargs=2, metavar=("TRAIN", "VAL"),
                    help="random subset of the final windows per split (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print statistics only")
    a = ap.parse_args()

    d, cov = pd.read_csv(a.boxes), pd.read_csv(a.coverage)
    stats = Counter()
    stats["rows: total GT intervals"] = len(d)
    stats["rows: clean with box (positives)"] = int((d.status == "clean").sum())
    ign = d[d.status != "clean"]
    stats["rows: ignore, no model box (unmatched)"] = int(((ign.source == "none") & (ign.label == "vox")).sum())
    stats["rows: ignore, 'combined' label"] = int((ign.label == "combined").sum())
    stats["rows: ignore, overlaps another GT interval"] = int(((ign.source != "none") & (ign.label == "vox")).sum())

    recs = sorted({rel_of(c) for c in d["clip"]})
    rng = np.random.default_rng(a.seed)
    val = set(rng.choice(recs, max(1, round(a.val_frac * len(recs))), replace=False))
    coco = {k: {"images": [], "annotations": [], "categories": [{"id": 1, "name": a.prompt}]} for k in ("train", "val")}
    w_stats = Counter()
    items = list(build(d, cov, a.spec_root, stats))
    keep = [1 - masked_frac(it[6], it[3], it[4]) >= a.min_labeled for it in items]
    for it, k in zip(items, keep):
        if not k:
            split = "val" if it[1] in val else "train"
            w_stats[f"{split}: windows dropped, < {a.min_labeled:.0%} visible"] += 1
            w_stats[f"{split}: boxes lost with them"] += len(it[5])
    items = [it for it, k in zip(items, keep) if k]
    n_pos = sum(bool(it[5]) for it in items if it[1] not in val)
    neg = [i for i, it in enumerate(items) if it[1] not in val and not it[5]]
    drop = set(rng.choice(neg, max(0, len(neg) - round(a.neg_ratio * n_pos)), replace=False).tolist())
    w_stats["train: empty windows dropped by --neg-ratio"] = len(drop)
    if a.max_windows:
        for split, cap in zip(("train", "val"), a.max_windows):
            idx = [i for i, it in enumerate(items) if i not in drop and (it[1] in val) == (split == "val")]
            drop |= set(rng.choice(idx, max(0, len(idx) - cap), replace=False).tolist())
    poly_cache = {}
    for k, (clip, rel, png, t0, t1, boxes, ignores) in enumerate(items):
        if k in drop:
            continue
        split = "val" if rel in val else "train"
        w_stats[f"{split}: windows"] += 1
        w_stats[f"{split}: windows with >= 1 box"] += bool(boxes)
        w_stats[f"{split}: boxes"] += len(boxes)
        w_stats[f"{split}: windows with masked (ignore/unlabeled) columns"] += bool(ignores)
        h, w = 257, 487  # gerbil_ssl PNG size; checked against the image below
        ch = int(re.search(r"headmic_(\d+)_", png.name)[1])
        masks = []
        for s, e, lo, hi, src in boxes:
            rle = box_mask(model_polys(src, rel, ch, png.name, poly_cache), to_px(s, e, lo, hi, t0, t1, w, h), h, w)
            masks.append(rle)
            w_stats[f"{split}: masks from {src}" if rle else f"{split}: no mask ({src})"] += 1
        if a.dry_run:
            continue
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        assert img.shape == (h, w), (png, img.shape)
        for s, e in ignores:
            img[:, int(np.floor((s - t0) / (t1 - t0) * w)):int(np.ceil((e - t0) / (t1 - t0) * w))] = 0
        name = f"{rel.replace('/', '_')}_{png.name}"
        out = Path(a.out) / split / name
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        c = coco[split]
        iid = len(c["images"])
        c["images"].append({"id": iid, "file_name": name, "width": w, "height": h, "clip": clip,
                            "window_start_sec": t0, "window_end_sec": t1})
        for (s, e, lo, hi, src), rle in zip(boxes, masks):
            bb = to_px(s, e, lo, hi, t0, t1, w, h)
            an = {"id": len(c["annotations"]), "image_id": iid, "category_id": 1, "bbox": bb,
                  "area": bb[2] * bb[3], "iscrowd": 0, "source": src}
            if rle:
                an["segmentation"] = rle
            c["annotations"].append(an)
    if not a.dry_run:
        for k, c in coco.items():
            json.dump(c, open(Path(a.out) / k / "_annotations.coco.json", "w"))
    print(f"recordings: {len(recs) - len(val)} train, {len(val)} val")
    for k in sorted(stats):
        print(f"{k:58s} {stats[k]}")
    for k in sorted(w_stats):
        print(f"{k:58s} {w_stats[k]}")


if __name__ == "__main__":
    main()
