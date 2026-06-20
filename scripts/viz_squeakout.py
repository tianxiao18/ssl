"""
Visualize SqueakOut masks: one overlay per image, uniform crop size, all in one figure.
Expects masks named <stem>_mask.png (as produced by run_squeakout.py).

By default visualizes raw spectrograms (excludes *_bbox and *_cropped). Pass
--cropped to visualize the *_cropped images and their masks instead.

Usage:
    python scripts/viz_squeakout.py <spectrogram_dir> <mask_dir> <out_png> [cols=12] [--cropped]

Example:
    python scripts/viz_squeakout.py \
        /mnt/home/the10/ceph/results/ssl/experiment_384/idx_000/spectrograms \
        outputs/squeakout_masks/exp384_idx000 \
        outputs/squeakout_viz/exp384_idx000_overlay.png \
        12 --cropped
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

use_cropped = "--cropped" in sys.argv[1:]
positional  = [a for a in sys.argv[1:] if a != "--cropped"]
spec_dir = Path(positional[0])
mask_dir = Path(positional[1])
out_png  = Path(positional[2])
cols     = int(positional[3]) if len(positional) > 3 else 12

if use_cropped:
    pairs = sorted([
        (p, mask_dir / f"{p.stem}_mask.png")
        for p in spec_dir.glob("*_cropped.png")
        if (mask_dir / f"{p.stem}_mask.png").exists()
    ])
else:
    pairs = sorted([
        (p, mask_dir / f"{p.stem}_mask.png")
        for p in spec_dir.glob("*.png")
        if not p.stem.endswith("_bbox") and not p.stem.endswith("_cropped")
        and (mask_dir / f"{p.stem}_mask.png").exists()
    ])


def mask_bbox_half(gray, mask):
    """Return (cy, cx, half) — centroid and half-size of the mask bounding box."""
    ys, xs = np.where(mask > 0)
    if not len(ys):
        h, w = gray.shape
        return h // 2, w // 2, max(h, w) // 2
    cy = (ys.min() + ys.max()) // 2
    cx = (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2
    return cy, cx, half


def load_pair(sp, mp):
    gray = np.array(Image.open(sp).convert("L"))
    mask = cv2.resize(
        np.array(Image.open(mp).convert("L")),
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return gray, mask


# --- pass 1: determine uniform half-size (95th percentile to ignore outliers) ---
halves = []
for sp, mp in pairs:
    gray, mask = load_pair(sp, mp)
    _, _, h = mask_bbox_half(gray, mask)
    halves.append(h)

half = min(int(np.percentile(halves, 95)) + 6, 80)

# --- pass 2: render ---
rows = -(-len(pairs) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows))
axes = np.array(axes).reshape(rows, cols)

for i, (sp, mp) in enumerate(pairs):
    gray, mask = load_pair(sp, mp)
    cy, cx, _ = mask_bbox_half(gray, mask)

    h, w = gray.shape
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    gray = gray[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]

    # pad to exactly (2*half) × (2*half), centered
    size = 2 * half
    ph = size - gray.shape[0]
    pw = size - gray.shape[1]
    if ph > 0 or pw > 0:
        pt, pb = ph // 2, ph - ph // 2
        pl, pr = pw // 2, pw - pw // 2
        gray = np.pad(gray, ((pt, pb), (pl, pr)))
        mask = np.pad(mask, ((pt, pb), (pl, pr)))

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 255), 1)

    ax = axes[i // cols, i % cols]
    ax.imshow(rgb)
    ax.axis("off")

for j in range(len(pairs), rows * cols):
    axes[j // cols, j % cols].axis("off")

plt.tight_layout(pad=0.2)
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_png}  (crop half={half}px, {len(pairs)} images, {cols} cols)")
