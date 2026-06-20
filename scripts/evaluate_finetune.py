"""
Compute segmentation metrics (Dice, IoU, Precision, Recall) for the pretrained
and fine-tuned SqueakOut models on the GT-masked images, and plot:

  Figure 1 — per-image metrics: box plots split by model × train/val
  Figure 2 — training loss curves: total, focal, and dice components

Usage:
    python scripts/evaluate_finetune.py \
        <coco_json> <spec_dir> <train_log_csv> \
        [--pretrained squeakout/squeakout_weights.ckpt] \
        [--finetuned  outputs/squeakout_finetuned/.../squeakout_finetuned.ckpt] \
        [--out-dir    outputs/squeakout_finetuned/.../plots] \
        [--threshold 0.5] [--val-frac 0.2] [--seed 42]

Example:
    python scripts/evaluate_finetune.py \
        outputs/squeakout_raw/exp384_idx000/_annotations.coco.json \
        outputs/squeakout_raw/exp384_idx000/spectrograms \
        outputs/squeakout_finetuned/exp384_idx000/train_log.csv \
        --finetuned outputs/squeakout_finetuned/exp384_idx000/squeakout_finetuned.ckpt \
        --out-dir   outputs/squeakout_finetuned/exp384_idx000/plots
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("coco_json")
parser.add_argument("spec_dir")
parser.add_argument("train_log_csv")
parser.add_argument("--pretrained", default=None)
parser.add_argument("--finetuned",  default=None)
parser.add_argument("--out-dir",    default="outputs/plots")
parser.add_argument("--threshold",  type=float, default=0.5)
parser.add_argument("--val-frac",   type=float, default=0.2)
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── Load models ───────────────────────────────────────────────────────────────

device = resolve_device()
pre_ckpt = Path(args.pretrained) if args.pretrained else \
           Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
pretrained = load_model(pre_ckpt, device=device)
finetuned  = load_model(Path(args.finetuned), device=device) if args.finetuned else None

# ── Load GT items (same split as finetune_squeakout.py) ──────────────────────

def norm(name: str) -> str:
    return Path(name).stem.replace(".", "-")

with open(args.coco_json) as f:
    coco = json.load(f)

spec_dir = Path(args.spec_dir)
disk = {norm(p.name): p for p in spec_dir.iterdir() if p.suffix == ".png"}

rles_by_id: dict[int, list] = {}
for ann in coco["annotations"]:
    if ann["category_id"] == 1 and isinstance(ann["segmentation"], dict):
        rles_by_id.setdefault(ann["image_id"], []).append(ann["segmentation"])

all_items = []
for im in coco["images"]:
    if im["id"] not in rles_by_id:
        continue
    orig = im.get("extra", {}).get("name", im["file_name"])
    path = disk.get(norm(orig))
    if path:
        all_items.append((path, rles_by_id[im["id"]], im["height"], im["width"]))

indices = list(range(len(all_items)))
random.shuffle(indices)
n_val   = max(1, int(len(all_items) * args.val_frac))
val_idx = set(indices[:n_val])
splits  = {i: ("val" if i in val_idx else "train") for i in range(len(all_items))}

print(f"GT items: {len(all_items)}  ({len(all_items)-n_val} train / {n_val} val)")

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_prob: np.ndarray, gt: np.ndarray,
                    threshold: float) -> dict[str, float]:
    pred = (pred_prob >= threshold).astype(np.uint8)
    tp   = (pred & gt).sum()
    fp   = (pred & ~gt).sum()
    fn   = (~pred & gt).sum()
    eps  = 1e-6
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def eval_model(model, label: str) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.inference_mode():
        for i, (path, rles, h, w) in enumerate(all_items):
            img_t = load_spectrogram_tensor(path, DEFAULT_IMAGE_SIZE).unsqueeze(0).to(device)
            prob  = torch.sigmoid(model(img_t)).squeeze().cpu().numpy()

            combined = np.zeros((h, w), dtype=np.uint8)
            for rle in rles:
                combined = np.maximum(combined, mask_util.decode(rle))
            gt = cv2.resize(combined, DEFAULT_IMAGE_SIZE,
                            interpolation=cv2.INTER_NEAREST).astype(bool)

            m = compute_metrics(prob, gt, args.threshold)
            m["model"] = label
            m["split"] = splits[i]
            rows.append(m)
    return pd.DataFrame(rows)


print("Evaluating pretrained model…")
df_pre = eval_model(pretrained, "pretrained")
if finetuned is not None:
    print("Evaluating finetuned model…")
    df_ft = eval_model(finetuned, "finetuned")
    df = pd.concat([df_pre, df_ft], ignore_index=True)
else:
    df = df_pre

# Print summary table
for split in ("train", "val"):
    print(f"\n── {split} ──")
    g = df[df["split"] == split].groupby("model")[["dice","iou","precision","recall"]].mean()
    print(g.to_string(float_format="{:.3f}".format))

# ── Figure 1: per-image metrics box plots ────────────────────────────────────

metrics   = ["dice", "iou", "precision", "recall"]
models    = df["model"].unique().tolist()
split_ord = ["train", "val"]
colors    = {"pretrained": "#4C72B0", "finetuned": "#DD8452"}

fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
fig.suptitle("Segmentation metrics — pretrained vs finetuned", fontsize=13, fontweight="bold")

for ax, metric in zip(axes, metrics):
    positions, tick_pos, tick_labels = [], [], []
    pos = 0
    for split in split_ord:
        group_center = pos + (len(models) - 1) / 2
        tick_pos.append(group_center)
        tick_labels.append(split)
        for model in models:
            vals = df[(df["split"] == split) & (df["model"] == model)][metric].values
            bp = ax.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            bp["boxes"][0].set_facecolor(colors.get(model, "grey"))
            positions.append(pos)
            pos += 1
        pos += 0.5  # gap between train/val groups

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_title(metric.capitalize(), fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=colors.get(m, "grey"), label=m) for m in models]
fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.8)

fig.tight_layout()
p1 = out_dir / "metrics_boxplot.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
print(f"\nSaved {p1}")

# ── Figure 2: training loss curves ───────────────────────────────────────────

log = pd.read_csv(args.train_log_csv)
has_components = "train_focal" in log.columns

fig2, axes2 = plt.subplots(1, 3 if has_components else 1,
                            figsize=(14 if has_components else 6, 4))
fig2.suptitle("Training loss curves", fontsize=13, fontweight="bold")

if not has_components:
    axes2 = [axes2]

# Panel 1: total loss
ax = axes2[0]
ax.plot(log["epoch"], log["train_loss"], label="train total", color="#4C72B0")
ax.plot(log["epoch"], log["val_loss"],   label="val total",   color="#4C72B0",
        linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Total loss (0.3·FL + 0.7·DL)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

if has_components:
    # Panel 2: focal loss
    ax = axes2[1]
    ax.plot(log["epoch"], log["train_focal"], label="train focal", color="#DD8452")
    ax.plot(log["epoch"], log["val_focal"],   label="val focal",   color="#DD8452",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_title("Focal loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # Panel 3: dice loss
    ax = axes2[2]
    ax.plot(log["epoch"], log["train_dice"], label="train dice", color="#55A868")
    ax.plot(log["epoch"], log["val_dice"],   label="val dice",   color="#55A868",
            linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_title("Dice loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

fig2.tight_layout()
p2 = out_dir / "loss_curves.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"Saved {p2}")
