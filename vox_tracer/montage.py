"""Montage building utilities for spectrogram visualisation."""
from pathlib import Path

import cv2
import numpy as np

CELL_W, CELL_H = 300, 257
LABEL_W = 90


def make_label_strip(labels, cell_h=CELL_H, label_w=LABEL_W):
    strip = np.zeros((cell_h * len(labels), label_w, 3), dtype=np.uint8)
    for i, name in enumerate(labels):
        y = i * cell_h + cell_h // 2 + 5
        cv2.putText(strip, name, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def resize_cell(img, label=None, cell_w=CELL_W, cell_h=CELL_H):
    cell = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(cell, label, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return cell


def overlay_boxes(bgr, boxes, color=(0, 255, 0)):
    """Draw [x, y, w, h] bounding boxes onto a copy of bgr."""
    out = bgr.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
    return out


def overlay_polygons(bgr, polys, color=(0, 255, 255)):
    """Draw COCO polygon coordinate lists onto a copy of bgr."""
    out = bgr.copy()
    contours = [
        np.array(p, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        for p in polys
    ]
    if contours:
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def viz_session_strip(spec_dir, pred_dir, out_dir, channels=None, cols=10,
                      sigmas=(2, 3, 4), threshold_pct=99.0, sample_rate=125000, freq_min=20000.0):
    """Three-row strip of every chunk in a session: raw | sato+exemplar box | predictions.

    The sato row shows the ridge response (yellow-green) with the pick_best_candidate
    box drawn in orange and its score labelled — useful for diagnosing bad SAM3 prompts.
    """
    import json
    from collections import defaultdict

    import numpy as np
    from skimage.filters import sato as _sato

    from vox_tracer.ridge import compute_seg_mask
    from vox_tracer.sam3_runner import pick_best_candidate
    from vox_tracer.spec import group_specs_by_channel

    COLOR_PRED = (255, 100, 0)   # blue  — predictions
    COLOR_BOX  = (0, 165, 255)   # orange — exemplar box
    sigmas = list(sigmas)
    out_dir = Path(out_dir)

    for ch, entries in group_specs_by_channel(spec_dir, channels).items():
        pred_path = Path(pred_dir) / f"coco_ch_{ch}.json"
        if not pred_path.exists():
            continue
        pred_coco = json.load(open(pred_path))
        img_by_fname = {im["file_name"]: im for im in pred_coco["images"]}
        ann_by_imgid = defaultdict(list)
        for ann in pred_coco["annotations"]:
            ann_by_imgid[ann["image_id"]].append(ann)

        columns = []
        for path, t0, t1 in entries:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            H, W  = gray.shape
            img_f = gray.astype(np.float64) / 255.0

            raw_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            response = _sato(img_f, sigmas=sigmas, black_ridges=False)
            seg_mask = compute_seg_mask(response, H, W, threshold_pct, sample_rate, freq_min)
            best_box, score = pick_best_candidate(seg_mask, response)

            # ridge row: raw spec + seg_mask contours + all component boxes (gray) + selected (orange)
            ridge_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ridge_img, contours, -1, (0, 200, 200), 1)
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(seg_mask)
            for lbl in range(1, n_labels):
                x0 = stats[lbl, cv2.CC_STAT_LEFT];  y0 = stats[lbl, cv2.CC_STAT_TOP]
                w  = stats[lbl, cv2.CC_STAT_WIDTH];  h  = stats[lbl, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(ridge_img, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
            if best_box is not None:
                x0, y0, x1, y1 = best_box
                cv2.rectangle(ridge_img, (x0, y0), (x1, y1), COLOR_BOX, 1)
                cv2.putText(ridge_img, f"{score:.2f}", (x0, max(y0 - 2, 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_BOX, 1, cv2.LINE_AA)

            pred_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if path.name in img_by_fname:
                iid = img_by_fname[path.name]["id"]
                for ann in ann_by_imgid[iid]:
                    polys = ann.get("segmentation", [])
                    pred_img = (overlay_polygons(pred_img, polys, COLOR_PRED) if polys
                                else overlay_boxes(pred_img, [ann["bbox"]], COLOR_PRED))

            columns.append(np.vstack([
                resize_cell(raw_img, label=f"ch{ch} t{t0:.0f}-{t1:.0f}s"),
                resize_cell(ridge_img),
                resize_cell(pred_img),
            ]))

        save_pages(columns, out_dir / f"ch_{ch}", prefix="page",
                   cols_per_page=cols, label_strip=make_label_strip(["raw", "ridge", "pred"]))
        print(f"  session strip ch {ch}: {len(columns)} chunks → {out_dir}/ch_{ch}")


def save_pages(columns, out_dir, prefix="page", cols_per_page=10, label_strip=None):
    """Write paginated montage images; return the number of pages saved."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pages = -(-len(columns) // cols_per_page)
    for page in range(n_pages):
        page_cols = columns[page * cols_per_page: (page + 1) * cols_per_page]
        parts = ([label_strip] if label_strip is not None else []) + page_cols
        img = np.hstack(parts)
        cv2.imwrite(str(out_dir / f"{prefix}{page:02d}.png"), img)
    return n_pages
