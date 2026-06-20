"""
Build side-by-side comparison montages from the COCO JSONs produced by
run_yolo_vox.py and run_squeakout_raw.py.  For each chunk window that overlaps
at least one DAS-annotated vocalization event, three panels are stacked vertically:

    row 1 -- raw spectrogram
    row 2 -- raw spectrogram + YOLO (vox) bounding boxes
    row 3 -- raw spectrogram + SqueakOut mask contour overlay

Results are paginated chronologically, --cols columns per page, into
<yolo_dir>/montages/.

Usage:
    python scripts/combine_spectrogram_views.py <yolo_dir> <squeakout_dir> \
        [--channels 118,35] [--cols 10] [--all] [--spec-dir <path>]

    <yolo_dir>      output directory passed to run_yolo_vox.py (montages written here)
    <squeakout_dir> output directory passed to run_squeakout_raw.py
    --spec-dir      directory containing the shared spectrogram PNGs
                    (default: outputs/spectrograms/<exp_id>, matching the default
                    used by run_yolo_vox.py and run_squeakout_raw.py)

Example:
    python scripts/combine_spectrogram_views.py \
        outputs/yolo_vox/exp384_idx000 \
        outputs/squeakout_raw/exp384_idx000
"""
import argparse
import json
from pathlib import Path

from spec_utils import default_spec_dir

import cv2
import numpy as np

CELL_W, CELL_H = 300, 257
VOX_COLOR  = (0, 255, 0)    # green, BGR
MASK_COLOR = (0, 255, 255)  # cyan,  BGR

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("yolo_dir", help="output directory from run_yolo_vox.py")
parser.add_argument("squeakout_dir", help="output directory from run_squeakout_raw.py")
parser.add_argument("--channels", default="118,35", help="comma-separated headmic channel numbers")
parser.add_argument("--cols", type=int, default=10, help="chunk columns per montage page")
parser.add_argument("--spec-dir", default=None,
                    help="shared spectrogram directory (default: outputs/spectrograms/<exp_id>)")
args = parser.parse_args()

channels     = [int(c) for c in args.channels.split(",")]
yolo_dir     = Path(args.yolo_dir)
squeakout_dir = Path(args.squeakout_dir)
spec_dir     = Path(args.spec_dir) if args.spec_dir else default_spec_dir(yolo_dir)
montage_dir  = yolo_dir / "montages"
montage_dir.mkdir(exist_ok=True)


def resize_cell(img, label=None):
    cell = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(cell, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return cell


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


for ch in channels:
    yolo_path     = yolo_dir / f"coco_ch_{ch}.json"
    squeakout_path = squeakout_dir / f"coco_ch_{ch}.json"

    for p in (yolo_path, squeakout_path):
        if not p.exists():
            print(f"channel {ch}: {p} not found, skipping")
            break
    else:
        pass  # both exist, fall through
    if not yolo_path.exists() or not squeakout_path.exists():
        continue

    with open(yolo_path) as f:
        yolo_coco = json.load(f)
    with open(squeakout_path) as f:
        sq_coco = json.load(f)

    # Index vox bboxes by image_id from the yolo JSON (category 1 = vox).
    vox_boxes: dict[int, list] = {}
    for ann in yolo_coco["annotations"]:
        if ann["category_id"] == 1:
            vox_boxes.setdefault(ann["image_id"], []).append(ann["bbox"])

    # Index squeakout polygons by file_name so we can match across JSONs.
    # (image IDs are independent between the two files.)
    sq_polys_by_name: dict[str, list] = {}
    sq_id_to_name = {im["id"]: im["file_name"] for im in sq_coco["images"]}
    for ann in sq_coco["annotations"]:
        if ann["category_id"] == 1:
            name = sq_id_to_name[ann["image_id"]]
            sq_polys_by_name.setdefault(name, []).extend(ann["segmentation"])

    images = yolo_coco["images"]
    print(f"channel {ch}: rendering {len(images)} windows")

    columns = []
    for im in images:
        iid   = im["id"]
        t0    = im["window_start_sec"]
        t1    = im["window_end_sec"]
        fname = im["file_name"]
        path  = spec_dir / fname
        if not path.exists():
            continue

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        # Row 1: raw spectrogram.
        row1 = resize_cell(to_bgr(gray), label=f"t={t0:.1f}-{t1:.1f}s")

        # Row 2: YOLO bounding boxes.
        bbox_img = to_bgr(gray)
        for x, y, w, h in vox_boxes.get(iid, []):
            cv2.rectangle(bbox_img, (int(x), int(y)), (int(x + w), int(y + h)), VOX_COLOR, 1)
        row2 = resize_cell(bbox_img)

        # Row 3: SqueakOut mask contours.
        overlay_img = to_bgr(gray)
        contours = [
            np.array(poly, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
            for poly in sq_polys_by_name.get(fname, [])
        ]
        if contours:
            cv2.drawContours(overlay_img, contours, -1, MASK_COLOR, 1)
        row3 = resize_cell(overlay_img)

        columns.append(np.vstack([row1, row2, row3]))

    if not columns:
        print(f"channel {ch}: no spectrogram files found on disk")
        continue

    n_pages = -(-len(columns) // args.cols)
    for page in range(n_pages):
        page_cols = columns[page * args.cols: (page + 1) * args.cols]
        montage   = np.hstack(page_cols)
        out_path  = montage_dir / f"ch_{ch}_page{page:02d}.png"
        cv2.imwrite(str(out_path), montage)
        print(f"  saved {out_path} ({len(page_cols)} cols)")
