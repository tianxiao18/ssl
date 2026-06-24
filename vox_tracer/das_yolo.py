"""DAS-guided YOLO vocalization detection over spectrogram chunks."""
import math
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from inference import get_model

from vox_tracer.coco import bbox_annotation, image_entry, make_coco, save_coco_per_channel
from vox_tracer.spec import (
    NPERSEG, NOVERLAP, SPEC_HI, SPEC_LO, SYNC_PAD,
    group_specs_by_channel, load_channel_audio, make_spectrogram_array,
)

YOLO_MODEL_ID      = "gerbil-vox-bbox/6"
YOLO_API_KEY       = "KmGLPn5Vq1yfYaJ6sRju"
YOLO_NMS_THRESHOLD = 0.3
VOX_MIN_DURATION_SEC = 0.01    # discard detections shorter than 10 ms
VOX_MIN_FREQ_HZ      = 10_000  # discard detections with centre frequency below 10 kHz


def load_das_events(recording_dir):
    """Load the DAS ground-truth CSV and return only 'vox' rows."""
    matches = sorted(glob(str(Path(recording_dir) / "*annotations_gt.csv")))
    if not matches:
        raise FileNotFoundError(f"No *annotations_gt.csv found in {recording_dir}")
    vox_times = pd.read_csv(matches[0])
    return vox_times[vox_times["name"] == "vox"]


def map_das_to_chunks(vox_rows, n_chunks, chunk_sec):
    """Return a dict mapping chunk index → list of DAS event dicts (sync-padded)."""
    das_by_chunk = {i: [] for i in range(n_chunks)}
    for vox_idx, orig_start, orig_stop in zip(
        vox_rows.index, vox_rows.start_seconds, vox_rows.stop_seconds
    ):
        if math.isnan(orig_start) or math.isnan(orig_stop):
            continue
        das_start = orig_start - SYNC_PAD / 2
        das_stop  = orig_stop  + SYNC_PAD / 2
        for i in range(n_chunks):
            chunk_start = i * chunk_sec
            chunk_stop  = (i + 1) * chunk_sec
            if das_start < chunk_stop and das_stop > chunk_start:
                das_by_chunk[i].append({
                    "vox_idx":   int(vox_idx),
                    "start_sec": das_start,
                    "stop_sec":  das_stop,
                })
    return das_by_chunk


def _merge_overlapping(events):
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


def _run_yolo_on_chunks(audio, sr, chunk_sec, paths, shapes, das_by_chunk, yolo_model):
    """Run YOLO on DAS-cropped sub-windows and remap detections into full-chunk coordinates."""
    chunk_len = int(chunk_sec * sr)
    n_chunks  = len(paths)
    yolo_bboxes = {i: [] for i in range(n_chunks)}

    for i in range(n_chunks):
        if not das_by_chunk[i]:
            continue
        h_chunk, w_chunk = shapes[i]
        chunk_start_i = i * chunk_len

        for event in _merge_overlapping(das_by_chunk[i]):
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

                if (x2 - x1) / w_das * das_samples / sr < VOX_MIN_DURATION_SEC:
                    continue

                f_lo = das_freqs[min(max(int((h_das - y2) / h_das * n_freqs), 0), n_freqs - 1)]
                f_hi = das_freqs[min(max(int((h_das - y1) / h_das * n_freqs), 0), n_freqs - 1)]
                if (f_lo + f_hi) / 2 < VOX_MIN_FREQ_HZ:
                    continue

                x1_c = (offset_samp + x1 / w_das * das_samples) / chunk_len * w_chunk
                x2_c = (offset_samp + x2 / w_das * das_samples) / chunk_len * w_chunk
                y1_c = y1 * h_chunk / h_das
                y2_c = y2 * h_chunk / h_das
                yolo_bboxes[i].append([x1_c, y1_c, x2_c, y2_c, float(conf)])

    # Chunk-level NMS: IoU pass then containment filter.
    for i in range(n_chunks):
        boxes = yolo_bboxes[i]
        if not boxes:
            continue
        if len(boxes) == 1:
            yolo_bboxes[i] = [boxes[0][:4]]
            continue
        xyxy  = np.array([b[:4] for b in boxes], dtype=np.float32)
        confs = np.array([b[4]  for b in boxes], dtype=np.float32)
        after_nms = sv.Detections(xyxy=xyxy, confidence=confs).with_nms(
            threshold=YOLO_NMS_THRESHOLD, class_agnostic=True
        )
        kept  = after_nms.xyxy
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

    return yolo_bboxes


def run_yolo(recording_dir, spec_dir, out_dir, channels=(118, 35), chunk_sec=1.0):
    """Run DAS-guided YOLO on each channel; write coco_ch_{ch}.json per channel.

    Spectrograms must be pre-generated in spec_dir (run gen_spectrograms.py first).
    Raw audio from recording_dir is still needed for DAS-event sub-window crops.
    """
    recording_dir = Path(recording_dir)
    vox_rows      = load_das_events(recording_dir)
    yolo_model    = get_model(YOLO_MODEL_ID, api_key=YOLO_API_KEY)
    coco_by_ch    = {}

    for ch in channels:
        result = load_channel_audio(recording_dir, ch)
        if result is None:
            print(f"channel {ch}: no headmic_{ch}_*.wav found, skipping")
            continue
        sr, audio, _ = result
        chunk_len = int(chunk_sec * sr)
        n_chunks  = len(audio) // chunk_len

        ch_specs = group_specs_by_channel(spec_dir, {ch}).get(ch, [])
        paths    = [p for p, _, _ in ch_specs]
        if not paths:
            print(f"channel {ch}: no spectrograms in {spec_dir}; run gen_spectrograms.py first")
            continue
        if len(paths) != n_chunks:
            print(f"channel {ch}: expected {n_chunks} chunks, found {len(paths)} in {spec_dir}, skipping")
            continue
        chunk_shape = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE).shape
        shapes      = [chunk_shape] * n_chunks

        das_by_chunk = map_das_to_chunks(vox_rows, n_chunks, chunk_sec)
        print(f"channel {ch}: {n_chunks} chunks, {len(audio) / sr:.1f}s recording")

        yolo_bboxes = _run_yolo_on_chunks(audio, sr, chunk_sec, paths, shapes,
                                           das_by_chunk, yolo_model)

        coco = make_coco(
            f"YOLO vox detections — ch {ch}", "vox",
            extra_info={
                "recording_dir": str(recording_dir),
                "chunk_sec":     chunk_sec,
                "spec_params":   {"nperseg": NPERSEG, "noverlap": NOVERLAP,
                                  "lo": SPEC_LO, "hi": SPEC_HI},
            },
        )
        for i, (path, shape) in enumerate(zip(paths, shapes)):
            h, w = shape
            coco["images"].append(image_entry(
                i, path.name, w, h,
                window_start_sec=i * chunk_sec,
                window_end_sec=(i + 1) * chunk_sec,
                das_events=das_by_chunk[i],
            ))
            for x1, y1, x2, y2 in yolo_bboxes[i]:
                coco["annotations"].append(
                    bbox_annotation(len(coco["annotations"]), i, x1, y1, x2, y2)
                )
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)
