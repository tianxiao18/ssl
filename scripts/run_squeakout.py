"""
Run SqueakOut on spectrogram images and save binary masks.

By default uses raw spectrograms (excludes *_bbox and *_cropped). Pass
--cropped to run on the *_cropped images instead.

Usage:
    python scripts/run_squeakout.py <spectrogram_dir> <mask_dir> [--cropped]

Example:
    python scripts/run_squeakout.py \
        /mnt/home/the10/ceph/results/ssl/experiment_384/idx_000/spectrograms \
        outputs/squeakout_masks/exp384_idx000 \
        --cropped
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor
from squeakout.inference import DEFAULT_MASK_THRESHOLD, logits_to_mask

use_cropped = "--cropped" in sys.argv[3:]
positional = [a for a in sys.argv[1:] if a != "--cropped"]
src_dir = Path(positional[0].strip())
out_dir = Path(positional[1].strip())
out_dir.mkdir(parents=True, exist_ok=True)

device = resolve_device()
ckpt = Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
model = load_model(ckpt, device=device)
model.eval()

if use_cropped:
    paths = sorted(p for p in src_dir.glob("*_cropped.png"))
    print(f"{len(paths)} cropped spectrograms found")
else:
    paths = sorted(
        p for p in src_dir.glob("*.png")
        if not p.stem.endswith("_bbox") and not p.stem.endswith("_cropped")
    )
    print(f"{len(paths)} raw spectrograms found")

tensors = torch.stack([load_spectrogram_tensor(p, DEFAULT_IMAGE_SIZE) for p in paths]).to(device)

with torch.inference_mode():
    logits = model(tensors)

def clean_mask(mask, min_area_ratio=0.1, min_component=100, min_total=300):
    """Remove small CCs; return empty mask if total area below min_total.

    A component is kept only if it passes BOTH:
      - relative: area >= min_area_ratio * largest component area
      - absolute: area >= min_component pixels
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
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


for path, logit in zip(paths, logits):
    mask = logits_to_mask(logit, threshold=DEFAULT_MASK_THRESHOLD)
    mask = clean_mask(mask)
    Image.fromarray(mask, mode="L").save(out_dir / f"{path.stem}_mask.png")

print(f"Saved {len(paths)} masks to {out_dir}")
