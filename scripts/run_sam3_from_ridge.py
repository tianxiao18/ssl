"""
For each 1-second spectrogram PNG, run the sato ridge filter to find the best
candidate vocalization trace, then use its bounding box as a positive image
exemplar for SAM3 (Segment Anything with Concepts) to exhaustively segment all
similar traces in the image.

Best candidate selection
------------------------
Hard filters (all must pass):
  - area in [min_area, max_area_ratio * H * W]
  - comp_h / H < 0.6       (skip vertical noise lines)
  - comp_w / W < 0.3       (skip horizontal smears)
  - eccentricity > 0.8     (skip round blobs; keep elongated sweeps)

Soft scoring (multiplicative, higher is better):
  score = mean_ridge_intensity * (1 - extent) * (1 - solidity)
  - mean ridge intensity  → prefer stronger ridges
  - 1 - extent            → prefer sparse/curved traces
  - 1 - solidity          → prefer non-convex (curved) shapes

SAM3 is installed from sam3/ at the repo root (pip install -e sam3/ --no-deps).
Checkpoint: hf auth login && hf download facebook/sam3 --local-dir sam3/

Usage:
    python scripts/run_sam3_from_ridge.py <spec_dir> <out_dir> \\
        [--sam3-checkpoint /path/to/sam3.pt] \\
        [--sigmas 2,3,4] [--threshold-pct 99.0] \\
        [--montage] [--cols 10]

Example (auto-download checkpoint):
    python scripts/run_sam3_from_ridge.py \\
        outputs/spectrograms/exp384_idx000 \\
        outputs/sam3_masks/exp384_idx000 \\
        --montage

Example (local checkpoint):
    python scripts/run_sam3_from_ridge.py \\
        outputs/spectrograms/exp384_idx000 \\
        outputs/sam3_masks/exp384_idx000 \\
        --sam3-checkpoint /mnt/home/the10/ssl/sam3/sam3.pt \\
        --montage
"""
import argparse
import json
import re
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
from skimage.filters import sato

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("spec_dir",  help="directory of 1-sec spectrogram PNGs")
parser.add_argument("out_dir",   help="output directory for COCO JSON (and montage)")
parser.add_argument("--sam3-checkpoint", default=None,
                    help="path to a local SAM3 .pt checkpoint; omit to auto-download "
                         "from HuggingFace (requires 'hf auth login' with approved access)")
parser.add_argument("--sigmas",  default="2,3,4",
                    help="comma-separated sigma values for sato filter (default: 2,3,4)")
parser.add_argument("--threshold-pct", type=float, default=99.0,
                    help="percentile threshold for ridge-filter segmentation (default: 99)")
parser.add_argument("--sample-rate", type=int, default=125000,
                    help="recording sample rate in Hz used for frequency-band cutoff (default: 125000)")
parser.add_argument("--freq-min", type=float, default=20000.0,
                    help="discard components entirely below this frequency in Hz (default: 20000)")
parser.add_argument("--min-area", type=int, default=30,
                    help="minimum component area in pixels (default: 30)")
parser.add_argument("--score-threshold", type=float, default=0.5,
                    help="SAM3 confidence threshold for keeping output masks (default: 0.5)")
parser.add_argument("--montage", action="store_true",
                    help="write a visualization montage alongside the COCO JSON")
parser.add_argument("--cols",    type=int, default=10,
                    help="columns per montage page (default: 10)")
args = parser.parse_args()

sigmas   = [float(s) for s in args.sigmas.split(",")]
spec_dir = Path(args.spec_dir)
out_dir  = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


# ── ridge-filter helpers (adapted from ridge_filter_montage.py) ──────────────

def _freq_cutoff_row(H):
    nyquist = args.sample_rate / 2.0
    return int((H - 1) * (1.0 - args.freq_min / nyquist))


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


def compute_seg_mask(response, H, W, vert_aspect=5.0, horiz_aspect=0.2):
    freq_cutoff_row = _freq_cutoff_row(H)
    thresh   = np.percentile(response, args.threshold_pct)
    binary   = (response >= thresh).astype(np.uint8) * 255
    filtered = _filter_components(binary, args.min_area, vert_aspect, horiz_aspect,
                                  freq_cutoff_row)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    closed   = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    return _filter_components(closed, args.min_area, vert_aspect, horiz_aspect,
                              freq_cutoff_row)


# ── candidate selection ───────────────────────────────────────────────────────

def _eccentricity(comp_mask):
    """Eccentricity of best-fit ellipse via second-order central moments."""
    M = cv2.moments(comp_mask)
    if M["m00"] < 1:
        return 0.0
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    tr   = mu20 + mu02
    disc = max((tr / 2) ** 2 - (mu20 * mu02 - mu11 ** 2), 0.0)
    lam1 = tr / 2 + disc ** 0.5
    lam2 = tr / 2 - disc ** 0.5
    return (1.0 - lam2 / lam1) ** 0.5 if lam1 > 1e-9 else 0.0


def pick_best_candidate(seg_mask, sato_response):
    """Return (x0, y0, x1, y1) of the best candidate component, or None.

    Components in seg_mask are already filtered by compute_seg_mask.
    Here we only add eccentricity (prefer elongated traces over blobs)
    and require score > 0 (positive sato ridge response).
    """
    H, W = seg_mask.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_mask)
    best_box   = None
    best_score = 0.0  # must beat 0 — negative mean_int means no real ridge

    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        x0     = stats[lbl, cv2.CC_STAT_LEFT]
        y0     = stats[lbl, cv2.CC_STAT_TOP]

        if comp_h / H >= 0.35:   # too tall — likely noise stripe, not a sweep
            continue
        if comp_w / W >= 0.30:   # too wide — horizontal smear
            continue

        comp_mask = (labels == lbl).astype(np.uint8)
        if _eccentricity(comp_mask) <= 0.8:  # skip round blobs
            continue

        mean_int = float(sato_response[comp_mask > 0].mean())
        extent   = area / (comp_h * comp_w)

        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        hull      = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 1.0

        score = mean_int * (1.0 - extent) * (1.0 - solidity)
        if score > best_score:
            best_score = score
            best_box   = (x0, y0, x0 + comp_w, y0 + comp_h)

    return best_box


# ── polygon helpers (from run_squeakout_raw.py) ───────────────────────────────

def mask_to_polygons(mask, img_shape):
    mh, mw = mask.shape
    h,  w  = img_shape
    sx, sy = w / mw, h / mh
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) >= 3:
            pts = c.reshape(-1, 2).astype(float)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            polys.append(pts.reshape(-1).tolist())
    return polys


# ── SAM3 setup ────────────────────────────────────────────────────────────────

from sam3.model_builder import build_sam3_image_model          # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor      # noqa: E402

if args.sam3_checkpoint:
    print(f"Loading SAM3 from {args.sam3_checkpoint} …")
    sam3_model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=False,
    )
else:
    print("Downloading SAM3 checkpoint from HuggingFace (requires approved access + hf auth login) …")
    sam3_model = build_sam3_image_model(load_from_HF=True)
processor = Sam3Processor(sam3_model, confidence_threshold=args.score_threshold)
print("SAM3 ready.")


# ── build ordered entry list ──────────────────────────────────────────────────

entries = [p.name for p in sorted(spec_dir.glob("*.png"))]

if not entries:
    print("No PNG files found in", spec_dir)
    raise SystemExit(0)

print(f"Processing {len(entries)} windows …")


# ── per-channel accumulators ──────────────────────────────────────────────────

from collections import defaultdict

def _make_coco():
    return {
        "info": {
            "description":     "SAM3 detections via sato ridge-filter exemplar",
            "spec_dir":        str(spec_dir),
            "sigmas":          sigmas,
            "threshold_pct":   args.threshold_pct,
            "score_threshold": args.score_threshold,
        },
        "licenses":    [],
        "categories":  [{"id": 1, "name": "sam3"}],
        "images":      [],
        "annotations": [],
    }

coco_by_ch      = defaultdict(_make_coco)
img_id_by_ch    = defaultdict(int)
ann_id_by_ch    = defaultdict(int)
n_skipped_by_ch = defaultdict(int)
montage_by_ch   = defaultdict(list) if args.montage else None

CELL_W, CELL_H = 300, 257


# ── montage cell helper ───────────────────────────────────────────────────────

def _cell(bgr, label=None):
    c = cv2.resize(bgr, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(c, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return c


# ── main loop ─────────────────────────────────────────────────────────────────

for fname in entries:
    path = spec_dir / fname
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
    H, W = gray.shape
    img_f = gray.astype(np.float64) / 255.0

    # parse channel and time from filename
    _m = re.match(r'headmic_(\d+)_.*_t([\d.]+)-([\d.]+)\.png$', fname)
    ch       = int(_m.group(1)) if _m else 0
    time_lbl = f"ch{ch} t{_m.group(2)}-{_m.group(3)}" if _m else fname

    img_id = img_id_by_ch[ch]
    ann_id = ann_id_by_ch[ch]
    coco   = coco_by_ch[ch]

    # 1. Sato filter → segmentation mask
    response = sato(img_f, sigmas=sigmas, black_ridges=False)
    seg_mask = compute_seg_mask(response, H, W)

    # 2. Pick best candidate trace
    best_box = pick_best_candidate(seg_mask, response)

    coco["images"].append({
        "id":        img_id,
        "file_name": fname,
        "width":     W,
        "height":    H,
    })

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if best_box is None:
        n_skipped_by_ch[ch] += 1
        if args.montage:
            sato_ov = bgr.copy()
            ctrs, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sato_ov, ctrs, -1, (0, 255, 255), 1)
            montage_by_ch[ch].append(np.vstack([
                _cell(bgr.copy(), label=time_lbl),
                _cell(sato_ov),
                _cell(bgr.copy()),
            ]))
        img_id_by_ch[ch] += 1
        continue

    x0, y0, x1, y1 = best_box

    # 3. Run SAM3 — bounding box as positive image exemplar
    # Box format: normalized [center_x, center_y, width, height] in [0, 1]
    norm_box = [
        (x0 + x1) / 2 / W,
        (y0 + y1) / 2 / H,
        (x1 - x0) / W,
        (y1 - y0) / H,
    ]
    pil_rgb = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state  = processor.set_image(pil_rgb)
        state  = processor.add_geometric_prompt(box=norm_box, label=True, state=state)

    out_masks  = state.get("masks",  [])
    out_scores = state.get("scores", [])

    # 4. Convert SAM3 output masks → COCO annotations
    # out_masks: bool tensor [N, 1, H, W] on CUDA; each slice is [1, H, W]
    for mask_tensor, score in zip(out_masks, out_scores):
        mask_u8 = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255

        for poly in mask_to_polygons(mask_u8, (H, W)):
            pts = np.array(poly).reshape(-1, 2)
            px1, py1 = pts.min(axis=0).tolist()
            px2, py2 = pts.max(axis=0).tolist()
            coco["annotations"].append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  1,
                "segmentation": [poly],
                "bbox":         [px1, py1, px2 - px1, py2 - py1],
                "area":         float(cv2.contourArea(
                                    np.array(poly, dtype=np.float32).reshape(-1, 1, 2))),
                "score":        float(score),
                "iscrowd":      0,
            })
            ann_id += 1
            ann_id_by_ch[ch] = ann_id

    # 5. Visualization column: raw | sato+candidate box | SAM3 masks
    if args.montage:
        raw_cell = _cell(bgr.copy(), label=time_lbl)

        sato_ov = bgr.copy()
        ctrs, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sato_ov, ctrs, -1, (0, 255, 255), 1)
        cv2.rectangle(sato_ov, (x0, y0), (x1, y1), (0, 128, 255), 2)
        sato_cell = _cell(sato_ov)

        sam3_ov = bgr.copy()
        for mask_tensor in out_masks:
            m_u8 = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255
            ctrs2, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sam3_ov, ctrs2, -1, (255, 100, 0), 1)
        sam3_cell = _cell(sam3_ov)

        montage_by_ch[ch].append(np.vstack([raw_cell, sato_cell, sam3_cell]))

    img_id_by_ch[ch] += 1


# ── save per-channel COCO JSON ────────────────────────────────────────────────

exp_id = spec_dir.name
for ch, coco in sorted(coco_by_ch.items()):
    n_ann  = ann_id_by_ch[ch]
    n_win  = img_id_by_ch[ch]
    n_skip = n_skipped_by_ch[ch]
    coco_path = out_dir / f"coco_sam3_{exp_id}_ch_{ch}.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"ch {ch}: {n_ann} annotations across {n_win - n_skip} windows "
          f"({n_skip} skipped) → {coco_path}")


# ── save per-channel montage pages ───────────────────────────────────────────

if args.montage:
    ROW_LABELS = ["raw", "sato", "sam3"]
    LABEL_W    = 60
    strip      = np.zeros((CELL_H * len(ROW_LABELS), LABEL_W, 3), dtype=np.uint8)
    for i, name in enumerate(ROW_LABELS):
        cv2.putText(strip, name,
                    (4, i * CELL_H + CELL_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    for ch, cols in sorted(montage_by_ch.items()):
        if not cols:
            continue
        montage_dir = out_dir / "montage" / f"ch_{ch}"
        montage_dir.mkdir(parents=True, exist_ok=True)
        n_pages = -(-len(cols) // args.cols)
        for page in range(n_pages):
            page_cols = cols[page * args.cols: (page + 1) * args.cols]
            img       = np.hstack([strip] + page_cols)
            cv2.imwrite(str(montage_dir / f"page{page:02d}.png"), img)
        print(f"ch {ch}: montage ({n_pages} page(s)) → {montage_dir}")
