"""SqueakOut inference over pre-generated spectrogram PNGs."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from vox_tracer.coco import image_entry, make_coco, mask_to_polygons, poly_annotation, save_coco_per_channel
from vox_tracer.spec import group_specs_by_channel

INFERENCE_BATCH_SIZE = 16


def clean_mask(mask, min_area_ratio=0.1, min_component=100, min_total=300):
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


def _run_batch(paths, model, device, batch_size=INFERENCE_BATCH_SIZE):
    from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor
    from squeakout.inference import DEFAULT_MASK_THRESHOLD, logits_to_mask

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


def run_squeakout(spec_dir, out_dir, channels=None, checkpoint=None, batch_size=INFERENCE_BATCH_SIZE):
    """Run SqueakOut on pre-generated spectrogram PNGs; write coco_ch_{ch}.json per channel."""
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "squeakout"))
    from squeakout import load_model, resolve_device

    spec_dir = Path(spec_dir)

    ckpt = Path(checkpoint) if checkpoint else repo_root / "squeakout" / "squeakout_weights.ckpt"
    device = resolve_device()
    model  = load_model(ckpt, device=device)
    model.eval()

    by_ch = group_specs_by_channel(spec_dir, channels)
    coco_by_ch = {}

    for ch, entries in by_ch.items():
        paths     = [e[0] for e in entries]
        raw_masks = _run_batch(paths, model, device, batch_size)
        masks     = [clean_mask(m) for m in raw_masks]

        coco = make_coco(f"SqueakOut detections — ch {ch}", "squeakout")
        for (path, t0, t1), mask in zip(entries, masks):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            H, W = gray.shape
            iid  = len(coco["images"])
            coco["images"].append(
                image_entry(iid, path.name, W, H, window_start_sec=t0, window_end_sec=t1)
            )
            for poly in mask_to_polygons(mask, (H, W)):
                coco["annotations"].append(poly_annotation(len(coco["annotations"]), iid, poly))
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)
