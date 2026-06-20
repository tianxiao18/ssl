"""
Build comparison montages for the fine-tuning experiment.  For every
GT-annotated chunk (category-1 RLE masks from _annotations.coco.json) the
script stacks four rows per chunk column:

    row 1 — raw spectrogram
    row 2 — ground-truth segmentation mask (decoded from RLE)
    row 3 — pretrained SqueakOut prediction
    row 4 — fine-tuned SqueakOut prediction

Chunks are ordered chronologically and split into separate train / val pages
using the same seed used during fine-tuning.

Usage:
    python scripts/visualize_finetune.py \
        <coco_json> <spec_dir> <out_dir> \
        [--pretrained squeakout/squeakout_weights.ckpt] \
        [--finetuned  outputs/squeakout_finetuned/.../squeakout_finetuned.ckpt] \
        [--val-frac 0.2] [--seed 42] [--cols 8]

Example:
    python scripts/visualize_finetune.py \
        outputs/squeakout_raw/exp384_idx000/_annotations.coco.json \
        outputs/squeakout_raw/exp384_idx000/spectrograms \
        outputs/squeakout_finetuned/exp384_idx000/montages_finetune \
        --finetuned outputs/squeakout_finetuned/exp384_idx000/squeakout_finetuned.ckpt
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import json
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("coco_json")
parser.add_argument("spec_dir")
parser.add_argument("out_dir")
parser.add_argument("--pretrained", default=None,
                    help="pretrained checkpoint (default: squeakout/squeakout_weights.ckpt)")
parser.add_argument("--finetuned",  default=None,
                    help="finetuned checkpoint")
parser.add_argument("--val-frac",   type=float, default=0.2)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--cols",       type=int,   default=8)
args = parser.parse_args()

random.seed(args.seed)

out_dir  = Path(args.out_dir)
spec_dir = Path(args.spec_dir)
out_dir.mkdir(parents=True, exist_ok=True)

CELL_W, CELL_H = 300, 257
LABEL_COLOR = (0, 255, 0)

# ── Load models ───────────────────────────────────────────────────────────────

device = resolve_device()
pre_ckpt = Path(args.pretrained) if args.pretrained else \
           Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
pretrained = load_model(pre_ckpt, device=device)

finetuned = None
if args.finetuned:
    finetuned = load_model(Path(args.finetuned), device=device)

# ── Load GT items ─────────────────────────────────────────────────────────────

def norm(name: str) -> str:
    return Path(name).stem.replace(".", "-")

with open(args.coco_json) as f:
    coco = json.load(f)

disk = {norm(p.name): p for p in spec_dir.iterdir() if p.suffix == ".png"}

rles_by_id: dict[int, list] = {}
for ann in coco["annotations"]:
    if ann["category_id"] == 1 and isinstance(ann["segmentation"], dict):
        rles_by_id.setdefault(ann["image_id"], []).append(ann["segmentation"])

# Collect items in original COCO order (chronological within each channel).
all_items: list[tuple[Path, list, int, int, str]] = []
for im in coco["images"]:
    if im["id"] not in rles_by_id:
        continue
    orig = im.get("extra", {}).get("name", im["file_name"])
    path = disk.get(norm(orig))
    if path is None:
        continue
    all_items.append((path, rles_by_id[im["id"]], im["height"], im["width"], orig))

# Reproduce the same train/val split as finetune_squeakout.py.
indices = list(range(len(all_items)))
random.shuffle(indices)
n_val   = max(1, int(len(all_items) * args.val_frac))
val_set = set(indices[:n_val])

train_items = [all_items[i] for i in range(len(all_items)) if i not in val_set]
val_items   = [all_items[i] for i in range(len(all_items)) if i     in val_set]

print(f"GT items: {len(all_items)} total → {len(train_items)} train / {len(val_items)} val")

# ── Rendering helpers ─────────────────────────────────────────────────────────

def cell(img_gray: np.ndarray, label: str) -> np.ndarray:
    c = cv2.resize(img_gray, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, LABEL_COLOR, 1, cv2.LINE_AA)
    return bgr


def predict_gray(model, img_tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        logits = model(img_tensor.to(device))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob * 255).clip(0, 255).astype(np.uint8)


def build_column(path: Path, rles: list, h: int, w: int, name: str) -> np.ndarray:
    img_tensor = load_spectrogram_tensor(path, DEFAULT_IMAGE_SIZE).unsqueeze(0)
    img_gray   = (img_tensor.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)

    # GT mask: merge all RLE masks.
    combined = np.zeros((h, w), dtype=np.uint8)
    for rle in rles:
        combined = np.maximum(combined, mask_util.decode(rle))
    gt_gray = cv2.resize((combined * 255).astype(np.uint8),
                          DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    short = name[:28]
    rows = [
        cell(img_gray,            short),
        cell(gt_gray,             "GT mask"),
        cell(predict_gray(pretrained, img_tensor), "pretrained"),
    ]
    if finetuned is not None:
        rows.append(cell(predict_gray(finetuned, img_tensor), "finetuned"))

    return np.vstack(rows)


def save_pages(items: list, tag: str) -> None:
    if not items:
        return
    columns = []
    for path, rles, h, w, name in items:
        columns.append(build_column(path, rles, h, w, name))

    n_pages = -(-len(columns) // args.cols)
    for page in range(n_pages):
        page_cols = columns[page * args.cols: (page + 1) * args.cols]
        montage   = np.hstack(page_cols)
        out_path  = out_dir / f"{tag}_page{page:02d}.png"
        cv2.imwrite(str(out_path), montage)
        print(f"  saved {out_path}  ({len(page_cols)} chunks)")


# ── Generate montages ─────────────────────────────────────────────────────────

print("Rendering train set…")
save_pages(train_items, "train")
print("Rendering val set…")
save_pages(val_items,   "val")
print(f"\nDone → {out_dir}")
