"""
For each headmic channel, slide a --chunk-sec window across the full recording,
save each window as a raw spectrogram PNG, run SqueakOut on every window, and
write a single COCO JSON file per channel that contains:

  - one image entry per window (file_name, time bounds, any overlapping DAS
    events stored as metadata so crops can be reproduced without saving them)
  - "squeakout" annotations (category 1): mask contour polygons + bounding box

Usage:
    python scripts/run_squeakout_raw.py <recording_dir> <out_dir> [--channels 118,35] [--chunk-sec 1.0]
        [--spec-dir <path>]

    --spec-dir  Directory where spectrogram PNGs are read/written.
                Defaults to outputs/spectrograms/<exp_id> (sibling of out_dir's parent),
                the same default used by run_yolo_vox.py, so spectrograms are shared
                and computed only once.  Pass an explicit path to override.

Example:
    python scripts/run_squeakout_raw.py \
        /mnt/home/the10/ceph/dataset/ssl_gt_data/experiment_384/idx_000 \
        outputs/squeakout_raw/exp384_idx000
"""
import argparse
import json
import math
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor
from squeakout.inference import DEFAULT_MASK_THRESHOLD, logits_to_mask
from spec_utils import NPERSEG, NOVERLAP, SPEC_HI, SPEC_LO, SYNC_PAD, default_spec_dir, write_spectrogram_img

INFERENCE_BATCH_SIZE = 16

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording_dir")
parser.add_argument("out_dir")
parser.add_argument("--channels", default="118,35", help="comma-separated headmic channel numbers")
parser.add_argument("--chunk-sec", type=float, default=1.0, help="window length in seconds")
parser.add_argument("--checkpoint", default=None,
                    help="path to squeakout checkpoint (default: squeakout/squeakout_weights.ckpt)")
parser.add_argument("--spec-dir", default=None,
                    help="shared spectrogram directory (default: outputs/spectrograms/<exp_id>)")
args = parser.parse_args()

channels = [int(c) for c in args.channels.split(",")]
recording_dir = Path(args.recording_dir)
out_dir = Path(args.out_dir)
spec_dir = Path(args.spec_dir) if args.spec_dir else default_spec_dir(out_dir)
spec_dir.mkdir(parents=True, exist_ok=True)

das_matches = sorted(glob(str(recording_dir / "*annotations_gt.csv")))
das_file = Path(das_matches[0])
vox_times = pd.read_csv(das_file)
vox_rows = vox_times[vox_times["name"] == "vox"]


def clean_mask(mask, min_area_ratio=0.1, min_component=100, min_total=300):
    """Remove small connected components; return empty mask if total area < min_total."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        return mask
    rel_threshold = max(areas) * min_area_ratio
    clean = np.zeros_like(mask)
    for label, area in enumerate(areas, start=1):
        if area >= rel_threshold and area >= min_component:
            clean[labels == label] = 255
    if (clean > 0).sum() < min_total:
        return np.zeros_like(mask)
    return clean


def run_squeakout(paths, model, device, batch_size=INFERENCE_BATCH_SIZE):
    masks = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            tensors = torch.stack(
                [load_spectrogram_tensor(p, DEFAULT_IMAGE_SIZE) for p in batch]
            ).to(device)
            logits = model(tensors)
            masks.extend(logits_to_mask(l, threshold=DEFAULT_MASK_THRESHOLD) for l in logits)
    return masks


def mask_to_polygons(mask, img_shape):
    """Return COCO polygons scaled from mask space into image (spectrogram) pixel space."""
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


device = resolve_device()
ckpt = Path(args.checkpoint) if args.checkpoint else Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
squeakout_model = load_model(ckpt, device=device)
squeakout_model.eval()

for ch in channels:
    matches = sorted(glob(str(recording_dir / f"headmic_{ch}_*.wav")))
    if not matches:
        print(f"channel {ch}: no headmic_{ch}_*.wav found in {recording_dir}, skipping")
        continue

    sr, audio = wavfile.read(matches[0])
    base_name = Path(matches[0]).stem

    chunk_len = int(args.chunk_sec * sr)
    n_chunks = len(audio) // chunk_len

    # Map each DAS event (sync-padded) onto the chunk(s) it overlaps (stored as image metadata).
    das_by_chunk: dict[int, list] = {i: [] for i in range(n_chunks)}
    for vox_idx, orig_start, orig_stop in zip(
        vox_rows.index, vox_rows.start_seconds, vox_rows.stop_seconds
    ):
        if math.isnan(orig_start) or math.isnan(orig_stop):
            continue
        das_start = orig_start - SYNC_PAD / 2
        das_stop  = orig_stop  + SYNC_PAD / 2
        for i in range(n_chunks):
            chunk_start = i * args.chunk_sec
            chunk_stop  = (i + 1) * args.chunk_sec
            if das_start < chunk_stop and das_stop > chunk_start:
                das_by_chunk[i].append({
                    "vox_idx":   int(vox_idx),
                    "start_sec": das_start,
                    "stop_sec":  das_stop,
                })

    # Save all chunk spectrograms (skip chunks already on disk).
    paths, shapes = [], []
    for i in range(n_chunks):
        start_i = i * chunk_len
        stop_i  = start_i + chunk_len
        chunk_path = spec_dir / f"{base_name}_chunk_{i:05d}_t{start_i/sr:.2f}-{stop_i/sr:.2f}.png"
        if chunk_path.exists():
            shape = cv2.imread(str(chunk_path), cv2.IMREAD_GRAYSCALE).shape
        else:
            shape = write_spectrogram_img(audio[start_i:stop_i], sr, chunk_path)
        paths.append(chunk_path)
        shapes.append(shape)

    print(f"channel {ch}: {n_chunks} chunks covering {n_chunks * args.chunk_sec:.1f}s "
          f"of a {len(audio)/sr:.1f}s recording")

    # SqueakOut: run on all chunks (DAS-agnostic).
    masks = [clean_mask(m) for m in run_squeakout(paths, squeakout_model, device)]

    # Build COCO JSON.
    coco = {
        "info": {
            "description":   f"SqueakOut detections — ch {ch}",
            "recording_dir": str(recording_dir),
            "chunk_sec":     args.chunk_sec,
            "spec_params":   {"nperseg": NPERSEG, "noverlap": NOVERLAP, "lo": SPEC_LO, "hi": SPEC_HI},
        },
        "licenses":    [],
        "categories":  [{"id": 1, "name": "squeakout"}],
        "images":      [],
        "annotations": [],
    }
    ann_id = 0

    for i, (path, shape, mask) in enumerate(zip(paths, shapes, masks)):
        h, w = shape
        coco["images"].append({
            "id":               i,
            "file_name":        path.name,
            "width":            w,
            "height":           h,
            "window_start_sec": i * args.chunk_sec,
            "window_end_sec":   (i + 1) * args.chunk_sec,
            "das_events":       das_by_chunk[i],
        })

        for poly in mask_to_polygons(mask, shape):
            pts = np.array(poly).reshape(-1, 2)
            px1, py1 = pts.min(axis=0).tolist()
            px2, py2 = pts.max(axis=0).tolist()
            coco["annotations"].append({
                "id":           ann_id,
                "image_id":     i,
                "category_id":  1,
                "segmentation": [poly],
                "bbox":         [px1, py1, px2 - px1, py2 - py1],
                "area":         float(cv2.contourArea(np.array(poly, dtype=np.float32).reshape(-1, 1, 2))),
                "iscrowd":      0,
            })
            ann_id += 1

    coco_path = out_dir / f"coco_ch_{ch}.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"channel {ch}: {ann_id} annotations → {coco_path}")

print(f"Saved spectrograms to {spec_dir}")
