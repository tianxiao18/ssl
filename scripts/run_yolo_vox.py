"""
For each headmic channel, slide a --chunk-sec window across the full recording,
save each window as a raw spectrogram PNG, run YOLO on DAS-cropped sub-windows,
and write a single COCO JSON file per channel that contains:

  - one image entry per window (file_name, time bounds, any overlapping DAS
    events stored as metadata so crops can be reproduced without saving them)
  - "vox" annotations (category 1): YOLO bounding boxes [x, y, w, h] remapped
    from the DAS sub-window into full-chunk pixel coordinates

Usage:
    python scripts/run_yolo_vox.py <recording_dir> <out_dir> [--channels 118,35] [--chunk-sec 1.0]
        [--spec-dir <path>]

    --spec-dir  Directory where spectrogram PNGs are read/written.
                Defaults to outputs/spectrograms/<exp_id> (sibling of out_dir's parent),
                the same default used by run_squeakout_raw.py, so spectrograms are shared
                and computed only once.  Pass an explicit path to override.

Example:
    python scripts/run_yolo_vox.py \
        /mnt/home/the10/ceph/dataset/ssl_gt_data/experiment_384/idx_000 \
        outputs/yolo_vox/exp384_idx000
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
import supervision as sv
from inference import get_model
from scipy.io import wavfile
from scipy.signal import spectrogram
from spec_utils import NPERSEG, NOVERLAP, SPEC_HI, SPEC_LO, SYNC_PAD, default_spec_dir, write_spectrogram_img

YOLO_MODEL_ID = "gerbil-vox-bbox/6"
YOLO_API_KEY = "KmGLPn5Vq1yfYaJ6sRju"
YOLO_NMS_THRESHOLD = 0.3
VOX_MIN_DURATION_SEC = 0.01    # discard detections shorter than 10 ms
VOX_MIN_FREQ_HZ      = 10_000  # discard detections with centre frequency below 10 kHz

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording_dir")
parser.add_argument("out_dir")
parser.add_argument("--channels", default="118,35", help="comma-separated headmic channel numbers")
parser.add_argument("--chunk-sec", type=float, default=1.0, help="window length in seconds")
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


def _merge_overlapping_das_events(events):
    """Merge overlapping DAS time windows so YOLO runs once per contiguous segment."""
    if not events:
        return []
    sorted_evs = sorted(events, key=lambda e: e["start_sec"])
    merged = [dict(sorted_evs[0])]
    for ev in sorted_evs[1:]:
        prev = merged[-1]
        if ev["start_sec"] < prev["stop_sec"]:
            prev["stop_sec"] = max(prev["stop_sec"], ev["stop_sec"])
        else:
            merged.append(dict(ev))
    return merged


def make_spectrogram_array(audio, sr, lo=SPEC_LO, hi=SPEC_HI):
    """Build a normalised dB spectrogram as a BGR uint8 array; return (img, freqs)."""
    freqs, _, Pxx = spectrogram(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Pxx_dB = np.flipud(10 * np.log10(Pxx + 1e-12))
    a = np.clip(Pxx_dB, lo, hi)
    a = (a - lo) / (hi - lo + 1e-12)
    a = (a * 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), freqs


yolo_model = get_model(YOLO_MODEL_ID, api_key=YOLO_API_KEY)

for ch in channels:
    matches = sorted(glob(str(recording_dir / f"headmic_{ch}_*.wav")))
    if not matches:
        print(f"channel {ch}: no headmic_{ch}_*.wav found in {recording_dir}, skipping")
        continue

    sr, audio = wavfile.read(matches[0])
    base_name = Path(matches[0]).stem

    chunk_len = int(args.chunk_sec * sr)
    n_chunks = len(audio) // chunk_len

    # Map each DAS event (sync-padded) onto the chunk(s) it overlaps.
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

    # YOLO: run on the DAS-cropped audio window; remap bbox into full-chunk coordinates.
    yolo_bboxes: dict[int, list] = {i: [] for i in range(n_chunks)}
    for i in range(n_chunks):
        if not das_by_chunk[i]:
            continue
        h_chunk, w_chunk = shapes[i]
        chunk_start_i = i * chunk_len
        for event in _merge_overlapping_das_events(das_by_chunk[i]):
            das_start_g = max(0, int(event["start_sec"] * sr))
            das_stop_g  = min(len(audio), int(event["stop_sec"] * sr))
            if das_stop_g - das_start_g < NPERSEG:
                continue
            das_img, das_freqs = make_spectrogram_array(audio[das_start_g:das_stop_g], sr)
            h_das, w_das = das_img.shape[:2]
            das_samples  = das_stop_g - das_start_g
            offset_samp  = das_start_g - chunk_start_i
            pred = yolo_model.infer(das_img)[0]
            dets = sv.Detections.from_inference(pred).with_nms(
                threshold=YOLO_NMS_THRESHOLD, class_agnostic=True
            )
            n_freqs = len(das_freqs)
            confs = dets.confidence if dets.confidence is not None else np.ones(len(dets))
            for bbox, conf in zip(dets.xyxy, confs):
                x1, y1, x2, y2 = (float(v) for v in bbox)

                # Discard detections shorter than 10 ms.
                if (x2 - x1) / w_das * das_samples / sr < VOX_MIN_DURATION_SEC:
                    continue

                # Discard detections whose centre frequency is below 10 kHz.
                # Spectrogram is vertically flipped: y=0 is highest freq, y=h_das is lowest.
                f_lo = das_freqs[min(max(int((h_das - y2) / h_das * n_freqs), 0), n_freqs - 1)]
                f_hi = das_freqs[min(max(int((h_das - y1) / h_das * n_freqs), 0), n_freqs - 1)]
                if (f_lo + f_hi) / 2 < VOX_MIN_FREQ_HZ:
                    continue

                x1_c = (offset_samp + x1 / w_das * das_samples) / chunk_len * w_chunk
                x2_c = (offset_samp + x2 / w_das * das_samples) / chunk_len * w_chunk
                y1_c = y1 * h_chunk / h_das
                y2_c = y2 * h_chunk / h_das
                yolo_bboxes[i].append([x1_c, y1_c, x2_c, y2_c, float(conf)])

    # Chunk-level NMS: suppress duplicates from overlapping DAS events.
    # Two-pass strategy:
    #   1. Standard IoU NMS (catches boxes with significant 2-D overlap).
    #   2. Containment filter: suppress any box where >50% of its area falls
    #      inside a larger box — handles the case where a tiny box is fully
    #      nested inside a large one (IoU stays low because the union is big).
    for i in range(n_chunks):
        boxes = yolo_bboxes[i]
        if not boxes:
            continue
        if len(boxes) == 1:
            yolo_bboxes[i] = [boxes[0][:4]]
            continue
        xyxy  = np.array([b[:4] for b in boxes], dtype=np.float32)
        confs = np.array([b[4]  for b in boxes], dtype=np.float32)
        # Pass 1: IoU NMS
        after_nms = sv.Detections(xyxy=xyxy, confidence=confs).with_nms(
            threshold=YOLO_NMS_THRESHOLD, class_agnostic=True
        )
        kept = after_nms.xyxy  # shape (K, 4)
        # Pass 2: containment filter
        areas = (kept[:, 2] - kept[:, 0]) * (kept[:, 3] - kept[:, 1])
        survive = []
        for k in range(len(kept)):
            contained = False
            for m in range(len(kept)):
                if k == m:
                    continue
                ix1 = max(kept[k, 0], kept[m, 0])
                iy1 = max(kept[k, 1], kept[m, 1])
                ix2 = min(kept[k, 2], kept[m, 2])
                iy2 = min(kept[k, 3], kept[m, 3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if areas[k] > 0 and inter / areas[k] > 0.5 and areas[m] > areas[k]:
                    contained = True
                    break
            if not contained:
                survive.append(k)
        yolo_bboxes[i] = [[float(v) for v in kept[k]] for k in survive]

    # Build COCO JSON.
    coco = {
        "info": {
            "description":   f"YOLO vox detections — ch {ch}",
            "recording_dir": str(recording_dir),
            "chunk_sec":     args.chunk_sec,
            "spec_params":   {"nperseg": NPERSEG, "noverlap": NOVERLAP, "lo": SPEC_LO, "hi": SPEC_HI},
        },
        "licenses":    [],
        "categories":  [{"id": 1, "name": "vox"}],
        "images":      [],
        "annotations": [],
    }
    ann_id = 0

    for i, (path, shape) in enumerate(zip(paths, shapes)):
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

        for x1, y1, x2, y2 in yolo_bboxes[i]:
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    i,
                "category_id": 1,
                "bbox":        [x1, y1, bw, bh],
                "area":        bw * bh,
                "iscrowd":     0,
            })
            ann_id += 1

    coco_path = out_dir / f"coco_ch_{ch}.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"channel {ch}: {ann_id} annotations → {coco_path}")

print(f"Saved spectrograms to {spec_dir}")
