"""
Montage comparing ridge filters on raw spectrograms.

Layout:
    columns = contiguous 1-second chunks in chronological order
    rows    = raw | sato | meijering | frangi | hessian (labelled on the left)

A combined mask montage is written to <out_dir>/masks/:
    threshold → morphological close → connected-component filtering
    (removes speckles < min_area and vertical interference lines by aspect ratio)

Filter outputs are rendered with the inferno heatmap.

Sigma choice
------------
All four filters accept a `sigmas` parameter that sets the Gaussian scales (in
pixels) at which ridges are sought.  The default [2,3,4] spans thin to
moderately thick ridges.  Pass --sigmas to override, e.g. --sigmas 1,2,3,4.

COCO JSON output
----------------
One COCO JSON is written per filter per channel:
    <out_dir>/coco_<filter>_ch_<ch>.json
This matches the naming convention of run_squeakout_raw.py / run_yolo_vox.py
and can be consumed directly by combine_spectrogram_views.py.

Usage:
    python scripts/ridge_filter_montage.py <spec_dir>
        [--cols 10] [--sigmas 2,3,4] [--out-dir <path>]

Example:
    python scripts/ridge_filter_montage.py outputs/spectrograms/exp384_idx000
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi, hessian, meijering, sato

CELL_W, CELL_H = 300, 257
LABEL_W = 90

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("spec_dir")
parser.add_argument("--cols", type=int, default=10)
parser.add_argument("--threshold-pct", type=float, default=99.0,
                    help="percentile threshold for segmentation mask (default: 99)")
parser.add_argument("--sigmas", default="2,3,4",
                    help="comma-separated sigma values for all filters (default: 2,3,4)")
parser.add_argument("--out-dir", default=None,
                    help="output directory (default: outputs/ridge_filters/<exp_id>)")
parser.add_argument("--sample-rate", type=int, default=125000,
                    help="recording sample rate in Hz (default: 125000)")
parser.add_argument("--freq-min", type=float, default=20000.0,
                    help="discard mask components entirely below this frequency in Hz (default: 20000)")
args = parser.parse_args()

sigmas = [float(s) for s in args.sigmas.split(",")]

FILTERS = [
    ("sato",      lambda img: sato(img, sigmas=sigmas, black_ridges=False)),
    ("meijering", lambda img: meijering(img, sigmas=sigmas, black_ridges=False)),
    ("frangi",    lambda img: frangi(img, sigmas=sigmas, black_ridges=False)),
    # hessian gives binary output (ridge≈0, background≈1), so invert and
    # multiply by the raw image to suppress the flat-background false positives
    ("hessian",   lambda img: (1.0 - hessian(img, sigmas=sigmas, black_ridges=False)) * img),
]
ROW_LABELS = ["raw"] + [name for name, _ in FILTERS]

spec_dir = Path(args.spec_dir)
out_dir  = Path(args.out_dir) if args.out_dir else \
           spec_dir.parent.parent / "ridge_filters" / spec_dir.name
out_dir.mkdir(parents=True, exist_ok=True)


def parse_fname(fname):
    m = re.match(r'headmic_(\d+)_.*_t([\d.]+)-([\d.]+)\.png$', fname)
    if m:
        return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


def make_label_strip(row_labels, cell_h, label_w=LABEL_W):
    strip = np.zeros((cell_h * len(row_labels), label_w, 3), dtype=np.uint8)
    for i, name in enumerate(row_labels):
        y = i * cell_h + cell_h // 2 + 5
        cv2.putText(strip, name, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def to_cell(arr_u8_bgr, label=None):
    cell = cv2.resize(arr_u8_bgr, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(cell, label, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return cell


def _filter_components(binary, min_area, vert_aspect, horiz_aspect, freq_cutoff_row):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    out = np.zeros_like(binary)
    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        aspect = comp_h / comp_w
        y_top  = stats[lbl, cv2.CC_STAT_TOP]
        if area < min_area:
            continue
        if aspect > vert_aspect:
            continue
        if aspect < horiz_aspect:
            continue
        if y_top > freq_cutoff_row:
            continue
        out[labels == lbl] = 255
    return out


def compute_seg_mask(response, min_area=30, vert_aspect=5.0, horiz_aspect=0.2):
    """threshold → filter lines → close → filter again → binary uint8 mask.

    Lines are removed BEFORE morphological close so the close cannot bridge
    a noise line into an adjacent vocalization blob.
    """
    H = response.shape[0]
    nyquist = args.sample_rate / 2.0
    freq_cutoff_row = int((H - 1) * (1.0 - args.freq_min / nyquist))

    thresh = np.percentile(response, args.threshold_pct)
    binary = (response >= thresh).astype(np.uint8) * 255

    filtered = _filter_components(binary, min_area, vert_aspect, horiz_aspect,
                                  freq_cutoff_row)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    return _filter_components(closed, min_area, vert_aspect, horiz_aspect,
                              freq_cutoff_row)


def filter_to_bgr(result):
    lo, hi = result.min(), result.max()
    if hi > lo:
        result = (result - lo) / (hi - lo)
    return cv2.applyColorMap((result * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


fnames = sorted(p.name for p in spec_dir.glob("*.png"))
if not fnames:
    print(f"No PNG files found in {spec_dir}")
    raise SystemExit(1)

print(f"Rendering {len(fnames)} chunks × {len(ROW_LABELS)} rows")

columns      = []
raw_cells    = []
mask_columns = {name: [] for name, _ in FILTERS}

# COCO accumulation keyed by channel
coco_images_by_ch = defaultdict(list)
coco_annots_by_ch = defaultdict(lambda: {name: [] for name, _ in FILTERS})
image_id_by_ch    = defaultdict(int)
annot_id_by_ch    = defaultdict(lambda: defaultdict(int))

for fname in fnames:
    path = spec_dir / fname
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue

    ch, t0, t1 = parse_fname(fname)
    label = f"t={t0:.1f}-{t1:.1f}s" if t0 is not None else fname

    image_id_by_ch[ch] += 1
    iid = image_id_by_ch[ch]
    H_orig, W_orig = gray.shape
    coco_images_by_ch[ch].append({
        "id": iid,
        "file_name": fname,
        "width": W_orig,
        "height": H_orig,
        "window_start_sec": t0,
        "window_end_sec": t1,
    })

    img_f   = gray.astype(np.float64) / 255.0
    raw_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raw_cells.append(to_cell(raw_bgr, label=label))
    rows = [to_cell(raw_bgr, label=label)]

    for filter_name, fn in FILTERS:
        result = fn(img_f)
        rows.append(to_cell(filter_to_bgr(result)))
        mask = compute_seg_mask(result)
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        mask_columns[filter_name].append(to_cell(overlay))

        for cnt in contours:
            if len(cnt) < 3:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            annot_id_by_ch[ch][filter_name] += 1
            coco_annots_by_ch[ch][filter_name].append({
                "id": annot_id_by_ch[ch][filter_name],
                "image_id": iid,
                "category_id": 1,
                "segmentation": [cnt.flatten().tolist()],
                "area": float(cv2.contourArea(cnt)),
                "bbox": [x, y, w, h],
                "iscrowd": 0,
            })

    columns.append(np.vstack(rows))

# ── main montage ──────────────────────────────────────────────────────────────
label_strip = make_label_strip(ROW_LABELS, CELL_H)
n_pages = -(-len(columns) // args.cols)
for page in range(n_pages):
    page_cols = columns[page * args.cols: (page + 1) * args.cols]
    montage   = np.hstack([label_strip] + page_cols)
    out_path  = out_dir / f"page{page:02d}.png"
    cv2.imwrite(str(out_path), montage)
    print(f"  saved {out_path} ({len(page_cols)} cols)")

# ── mask comparison montage ───────────────────────────────────────────────────
mask_labels      = ["raw"] + [name for name, _ in FILTERS]
mask_label_strip = make_label_strip(mask_labels, CELL_H)
masks_dir = out_dir / "masks"
masks_dir.mkdir(exist_ok=True)
mask_cols_combined = [
    np.vstack([raw_cells[i]] + [mask_columns[name][i] for name, _ in FILTERS])
    for i in range(len(raw_cells))
]
n_pages = -(-len(mask_cols_combined) // args.cols)
for page in range(n_pages):
    page_cols = mask_cols_combined[page * args.cols: (page + 1) * args.cols]
    montage   = np.hstack([mask_label_strip] + page_cols)
    out_path  = masks_dir / f"page{page:02d}.png"
    cv2.imwrite(str(out_path), montage)
print(f"  masks: {n_pages} page(s) → {masks_dir}")

# ── COCO JSON output (per filter × per channel) ───────────────────────────────
categories = [{"id": 1, "name": "vox"}]
for ch in sorted(coco_images_by_ch):
    for filter_name, _ in FILTERS:
        coco = {
            "info": {"description": f"ridge_filter_montage — {filter_name} ch {ch}"},
            "images": coco_images_by_ch[ch],
            "annotations": coco_annots_by_ch[ch][filter_name],
            "categories": categories,
        }
        json_path = out_dir / f"coco_{filter_name}_ch_{ch}.json"
        with open(json_path, "w") as f:
            json.dump(coco, f, indent=2)
        n_ann = len(coco_annots_by_ch[ch][filter_name])
        print(f"  COCO ({filter_name} ch {ch}): {json_path}  ({n_ann} annotations)")
