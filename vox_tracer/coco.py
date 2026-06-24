"""Shared COCO JSON building utilities."""
import json
from pathlib import Path

import cv2
import numpy as np


def make_coco(description, category_name, extra_info=None):
    info = {"description": description}
    if extra_info:
        info.update(extra_info)
    return {
        "info": info,
        "licenses": [],
        "categories": [{"id": 1, "name": category_name}],
        "images": [],
        "annotations": [],
    }


def image_entry(image_id, file_name, width, height, **kwargs):
    entry = {"id": image_id, "file_name": str(file_name), "width": width, "height": height}
    entry.update(kwargs)
    return entry


def mask_to_polygons(mask, img_shape):
    """Convert a binary mask to a list of COCO polygons scaled to img_shape (h, w)."""
    mh, mw = mask.shape
    h, w = img_shape
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


def poly_annotation(ann_id, image_id, poly, extra=None):
    """Build a COCO annotation dict from a flat polygon coordinate list."""
    pts = np.array(poly).reshape(-1, 2)
    px1, py1 = pts.min(axis=0).tolist()
    px2, py2 = pts.max(axis=0).tolist()
    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [poly],
        "bbox": [px1, py1, px2 - px1, py2 - py1],
        "area": float(cv2.contourArea(np.array(poly, dtype=np.float32).reshape(-1, 1, 2))),
        "iscrowd": 0,
    }
    if extra:
        ann.update(extra)
    return ann


def save_coco_per_channel(coco_by_ch, out_dir):
    """Write coco_ch_{ch}.json for each channel into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch, coco in sorted(coco_by_ch.items()):
        p = out_dir / f"coco_ch_{ch}.json"
        with open(p, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"ch {ch}: {len(coco['annotations'])} annotations → {p}")


def bbox_annotation(ann_id, image_id, x1, y1, x2, y2):
    """Build a COCO annotation dict from an xyxy bounding box."""
    bw, bh = x2 - x1, y2 - y1
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x1, y1, bw, bh],
        "area": bw * bh,
        "iscrowd": 0,
    }
