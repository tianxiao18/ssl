"""
Overlay COCO 'segmented' masks (cyan only) on spectrogram images and produce
summary statistics.

Outputs:
  <out_png>            – grid of all spectrograms in sorted order (empty frames included)
  <out_png stem>_stats.png – summary statistics figure

Usage (single image):
    python scripts/inspect_coco_masks.py \\
        outputs/squeakout_raw/exp384_idx000/_annotations.coco.json \\
        outputs/squeakout_raw/exp384_idx000/spectrograms/headmic_118_file_001_chunk_00000_t0.00-1.00.png \\
        outputs/inspect_masks/headmic_118_gt.png

Usage (full sequence grid + stats):
    python scripts/inspect_coco_masks.py \\
        outputs/squeakout_raw/exp384_idx000/_annotations.coco.json \\
        outputs/squeakout_raw/exp384_idx000/spectrograms/ \\
        outputs/inspect_masks/all_gt.png --cols 12
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask_util

CYAN_BGR = (255, 220, 0)   # BGR
CYAN_RGB = (0, 220, 255)   # RGB for matplotlib
SEGMENTED_CAT_ID = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("coco_json", type=Path)
    p.add_argument("image_path", type=Path,
                   help="Single spectrogram .png OR a directory of spectrograms")
    p.add_argument("out_png", type=Path)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.45,
                   help="Mask fill opacity (0=invisible, 1=opaque)")
    p.add_argument("--gap", type=int, default=20,
                   help="Max frame gap within a cluster (default 20)")
    p.add_argument("--pad", type=int, default=5,
                   help="Padding frames added around each cluster (default 5)")
    return p.parse_args()


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    id2img = {img["id"]: img for img in data["images"]}
    stem2id: dict[str, int] = {}
    for img in data["images"]:
        stem2id[Path(img["file_name"]).stem] = img["id"]
        extra_name = img.get("extra", {}).get("name", "")
        if extra_name:
            stem2id[Path(extra_name).stem] = img["id"]
    id2anns: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        id2anns[ann["image_id"]].append(ann)
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    return id2img, stem2id, id2anns, cat_map


def find_image_id(stem: str, stem2id: dict) -> int | None:
    if stem in stem2id:
        return stem2id[stem]
    norm = stem.replace(".", "-")
    for s, iid in stem2id.items():
        if s.replace(".", "-") == norm:
            return iid
    for s, iid in stem2id.items():
        sn = s.replace(".", "-")
        if sn.startswith(norm) or norm.startswith(sn):
            return iid
    return None


def seg_to_mask(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        return coco_mask_util.decode(seg).astype(np.uint8)
    rles = coco_mask_util.frPyObjects(seg, h, w)
    return coco_mask_util.decode(coco_mask_util.merge(rles)).astype(np.uint8)


def segmented_anns(anns: list) -> list:
    return [a for a in anns if a["category_id"] == SEGMENTED_CAT_ID]


def draw_segmented(img_bgr: np.ndarray, anns: list, alpha: float) -> np.ndarray:
    """Overlay only 'segmented' annotations in cyan."""
    h, w = img_bgr.shape[:2]
    seg_anns = segmented_anns(anns)
    if not seg_anns:
        return img_bgr.copy()
    overlay = img_bgr.copy()
    for ann in seg_anns:
        bmask = seg_to_mask(ann["segmentation"], h, w)
        overlay[bmask > 0] = CYAN_BGR
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    for ann in seg_anns:
        bmask = seg_to_mask(ann["segmentation"], h, w)
        contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, CYAN_BGR, 2)
    return blended


def render_single(spec_path: Path, anns: list, alpha: float) -> np.ndarray:
    img_rgb = np.array(Image.open(spec_path).convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = draw_segmented(img_bgr, anns, alpha)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def mask_coverage(anns: list, h: int, w: int) -> float:
    """Fraction of pixels covered by the union of segmented masks."""
    seg_anns = segmented_anns(anns)
    if not seg_anns:
        return 0.0
    union = np.zeros((h, w), dtype=np.uint8)
    for ann in seg_anns:
        union |= seg_to_mask(ann["segmentation"], h, w)
    return union.sum() / (h * w)


# ── stats figure ───────────────────────────────────────────────────────────────

def save_stats(pairs: list[tuple[Path, list]], out_path: Path, id2img: dict, stem2id: dict):
    """
    pairs: list of (spec_path, all_anns_for_image)
    Produces a stats figure with:
      1. Timeline: coverage % per frame (bar)
      2. Annotation count per frame
      3. Distribution of non-zero coverages (histogram)
    """
    labels, coverages, n_masks = [], [], []
    for sp, anns in pairs:
        iid = find_image_id(sp.stem, stem2id)
        img_meta = id2img.get(iid, {}) if iid is not None else {}
        h = img_meta.get("height", 257)
        w = img_meta.get("width", 487)
        cov = mask_coverage(anns, h, w)
        coverages.append(cov * 100)
        n_masks.append(len(segmented_anns(anns)))
        labels.append(sp.stem)

    coverages = np.array(coverages)
    n_masks = np.array(n_masks)
    n_total = len(pairs)
    n_annotated = (coverages > 0).sum()
    nonzero_cov = coverages[coverages > 0]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Ground-truth 'segmented' mask statistics  —  "
        f"{n_annotated}/{n_total} frames annotated  |  "
        f"{n_masks.sum()} total masks  |  "
        f"mean coverage (annotated frames): {nonzero_cov.mean():.2f}%  "
        f"[median {np.median(nonzero_cov):.2f}%, max {nonzero_cov.max():.2f}%]",
        fontsize=10, y=0.98,
    )

    xs = np.arange(n_total)

    # ── panel 1: coverage timeline ─────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 1, 1)
    colors = [(*[c / 255 for c in CYAN_RGB], 0.8) if c > 0 else (0.25, 0.25, 0.25, 0.6)
              for c in coverages]
    ax1.bar(xs, coverages, color=colors, width=1.0, linewidth=0)
    ax1.set_ylabel("Mask coverage (%)", fontsize=8)
    ax1.set_xlim(-0.5, n_total - 0.5)
    ax1.set_title("Per-frame mask coverage (cyan = annotated, grey = empty)", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.set_xlabel("Frame index (sorted by filename)", fontsize=8)

    # ── panel 2: annotation count timeline ────────────────────────────────────
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.bar(xs, n_masks,
            color=[(*[c / 255 for c in CYAN_RGB], 0.8) if n > 0 else (0.25, 0.25, 0.25, 0.6)
                   for n in n_masks],
            width=1.0, linewidth=0)
    ax2.set_ylabel("# segmented masks", fontsize=8)
    ax2.set_xlim(-0.5, n_total - 0.5)
    ax2.set_title("Number of 'segmented' annotations per frame", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_xlabel("Frame index (sorted by filename)", fontsize=8)
    ax2.yaxis.get_major_locator().set_params(integer=True)

    # ── panel 3: coverage histogram (non-zero frames only) ────────────────────
    ax3 = fig.add_subplot(3, 1, 3)
    if len(nonzero_cov) > 0:
        ax3.hist(nonzero_cov, bins=20, color=[c / 255 for c in CYAN_RGB],
                 edgecolor="white", linewidth=0.4, alpha=0.85)
    ax3.set_xlabel("Mask coverage (%) — annotated frames only", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)
    ax3.set_title(f"Distribution of mask coverage across {n_annotated} annotated frames", fontsize=8)
    ax3.tick_params(labelsize=7)

    # annotate percentiles
    for pct, val in zip([25, 50, 75], np.percentile(nonzero_cov, [25, 50, 75])):
        ax3.axvline(val, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        ax3.text(val, ax3.get_ylim()[1] * 0.95, f"p{pct}={val:.1f}%",
                 ha="center", va="top", fontsize=6, color="white")

    fig.set_facecolor("#111111")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#1a1a1a")
        ax.spines[:].set_color("#444444")
        ax.tick_params(colors="#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#dddddd")
    fig.suptitle(fig.texts[0].get_text(), color="#eeeeee", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved stats: {out_path}")


# ── cluster helpers ────────────────────────────────────────────────────────────

_SEP = object()  # sentinel for separator slots in the grid


def cluster_pairs(pairs: list, gap: int, pad: int) -> list:
    """
    Return a flat list of (spec_path, anns) entries with _SEP sentinels between
    clusters. Only frames within `pad` of an annotated frame are included.
    Clusters are groups of annotated frames separated by > `gap` empty frames.
    """
    n = len(pairs)
    annotated = [i for i, (_, a) in enumerate(pairs) if segmented_anns(a)]
    if not annotated:
        return pairs

    # find cluster boundaries
    clusters: list[tuple[int, int]] = []
    start = annotated[0]
    prev = annotated[0]
    for idx in annotated[1:]:
        if idx - prev > gap:
            clusters.append((start, prev))
            start = idx
        prev = idx
    clusters.append((start, prev))

    # expand by pad, clamp to [0, n-1]
    windows = [(max(0, s - pad), min(n - 1, e + pad)) for s, e in clusters]

    result = []
    for wi, (s, e) in enumerate(windows):
        if wi > 0:
            result.append(_SEP)
        for i in range(s, e + 1):
            result.append(pairs[i])

    total_frames = sum(e - s + 1 for s, e in windows)
    n_clusters = len(clusters)
    print(f"Showing {n_clusters} cluster(s): "
          + ", ".join(f"frames {s}–{e}" for s, e in windows)
          + f"  ({total_frames} frames total)")
    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    id2img, stem2id, id2anns, cat_map = load_coco(args.coco_json)

    if args.image_path.is_dir():
        spec_dir = args.image_path
        candidates = sorted(p for p in spec_dir.glob("*.png")
                            if not p.stem.endswith(("_bbox", "_cropped")))
        # all images in sequence; empty list for unannotated frames
        pairs = []
        for sp in candidates:
            iid = find_image_id(sp.stem, stem2id)
            anns = id2anns[iid] if iid is not None else []
            pairs.append((sp, anns))
        n_ann = sum(1 for _, a in pairs if segmented_anns(a))
        print(f"{len(pairs)} total frames, {n_ann} with 'segmented' annotations")
        all_pairs = pairs  # keep full list for stats
        pairs = cluster_pairs(pairs, gap=args.gap, pad=args.pad)
    else:
        sp = args.image_path
        iid = find_image_id(sp.stem, stem2id)
        anns = id2anns[iid] if iid is not None else []
        seg = segmented_anns(anns)
        print(f"Image id={iid}, {len(seg)} 'segmented' annotations")
        for a in seg:
            print(f"  bbox={a['bbox']}")
        all_pairs = pairs = [(sp, anns)]

    args.out_png.parent.mkdir(parents=True, exist_ok=True)

    # ── single image ────────────────────────────────────────────────────────────
    if len(pairs) == 1:
        sp, anns = pairs[0]
        result = render_single(sp, anns, args.alpha)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(result)
        ax.set_title(sp.name, fontsize=9)
        ax.axis("off")
        ax.legend(handles=[mpatches.Patch(color=[c / 255 for c in CYAN_RGB],
                                          label="segmented")],
                  loc="upper right", fontsize=8, framealpha=0.7)
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.out_png}")
        return

    # ── grid (zoomed to vocalization clusters) ─────────────────────────────────
    cols = args.cols
    # build row-by-row: auto-wrap at `cols`, _SEP inserts a full blank divider row
    rows_data: list[list] = [[]]
    for entry in pairs:
        if entry is _SEP:
            while len(rows_data[-1]) < cols:
                rows_data[-1].append(None)
            rows_data.append([_SEP] * cols)
            rows_data.append([])
        else:
            if len(rows_data[-1]) == cols:
                rows_data.append([])
            rows_data[-1].append(entry)
    while len(rows_data[-1]) < cols:
        rows_data[-1].append(None)

    n_rows = len(rows_data)
    cell_w, cell_h = 3.5, 2.0
    fig, axes = plt.subplots(n_rows, cols,
                             figsize=(cell_w * cols, cell_h * n_rows),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.02})
    axes = np.array(axes).reshape(n_rows, cols)

    for r, row in enumerate(rows_data):
        for c, entry in enumerate(row):
            ax = axes[r, c]
            ax.axis("off")
            if entry is None:
                ax.set_facecolor("#111111")
            elif entry is _SEP:
                ax.set_facecolor("#222222")
            else:
                sp, anns = entry
                result = render_single(sp, anns, args.alpha)
                ax.imshow(result)
                has_seg = bool(segmented_anns(anns))
                title_color = "#00dcff" if has_seg else "#555555"
                ax.set_title(sp.stem[:32], fontsize=5.5, color=title_color, pad=2)

    fig.set_facecolor("#111111")
    fig.legend(
        handles=[mpatches.Patch(color=[c / 255 for c in CYAN_RGB], label="segmented")],
        loc="lower right", fontsize=9, framealpha=0.8,
    )
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    n_frames = sum(1 for e in pairs if e is not _SEP and e is not None)
    print(f"Saved grid: {args.out_png}  ({n_frames} frames shown, {cols} cols)")

    # ── stats ──────────────────────────────────────────────────────────────────
    stats_png = args.out_png.with_name(args.out_png.stem + "_stats.png")
    save_stats(all_pairs, stats_png, id2img, stem2id)


if __name__ == "__main__":
    main()
