"""
Fine-tune a pretrained SqueakOut checkpoint on ground-truth segmentation masks,
then run inference on a recording and write COCO JSONs (same format as
run_squeakout_raw.py) so the output can be fed directly to
combine_spectrogram_views.py.

Loss: α·FocalLoss + (1-α)·DiceLoss  (α=0.3, γ=2)
Augmentations (per-batch, independent probabilities):
  (a) 75%  — random affine
  (b) 25%  — contrast normalisation
  (c) 25%  — Gaussian blur
  (d) 33%  — one of: additive noise / Gaussian noise / frequency noise / elastic

Usage:
    python scripts/finetune_squeakout.py \
        <recording_dir> <coco_json> <out_dir> \
        [--gt-spec-dir DIR] [--channels 118,35] [--chunk-sec 1.0] \
        [--epochs 50] [--batch-size 8] [--lr 1e-5] \
        [--val-frac 0.2] [--freeze-backbone] [--seed 42]

Example:
    python scripts/finetune_squeakout.py \
        /mnt/home/the10/ceph/dataset/ssl_gt_data/experiment_384/idx_000 \
        outputs/squeakout_raw/exp384_idx000/_annotations.coco.json \
        outputs/squeakout_finetuned/exp384_idx000 \
        --gt-spec-dir outputs/squeakout_raw/exp384_idx000/spectrograms
"""
import argparse
import json
import math
import random
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram as scipy_spectrogram
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor
from squeakout.inference import DEFAULT_MASK_THRESHOLD, logits_to_mask

# ── Constants (shared with run_squeakout_raw.py) ─────────────────────────────

SPEC_LO, SPEC_HI = -70, 0
NPERSEG, NOVERLAP = 512, 256
INFERENCE_BATCH_SIZE = 16
SYNC_PAD = 0.05

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recording_dir", help="directory with headmic WAVs and annotations_gt.csv")
parser.add_argument("coco_json",     help="path to _annotations.coco.json (GT for fine-tuning)")
parser.add_argument("out_dir",       help="output: checkpoint + spectrograms/ + coco_ch_*.json")
parser.add_argument("--gt-spec-dir", default=None,
                    help="directory of GT spectrogram PNGs for training "
                         "(default: out_dir/spectrograms, i.e. reuse inference spectrograms)")
parser.add_argument("--channels",        default="118,35")
parser.add_argument("--chunk-sec",       type=float, default=1.0)
parser.add_argument("--epochs",          type=int,   default=50)
parser.add_argument("--batch-size",      type=int,   default=8)
parser.add_argument("--lr",              type=float, default=1e-5)
parser.add_argument("--val-frac",        type=float, default=0.2)
parser.add_argument("--freeze-backbone", action="store_true", default=True,
                    help="freeze backbone, fine-tune decoder only (default: True)")
parser.add_argument("--seed",            type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

channels      = [int(c) for c in args.channels.split(",")]
recording_dir = Path(args.recording_dir)
out_dir       = Path(args.out_dir)
spec_dir      = out_dir / "spectrograms"
spec_dir.mkdir(parents=True, exist_ok=True)

gt_spec_dir = Path(args.gt_spec_dir) if args.gt_spec_dir else spec_dir

# ── Loss ─────────────────────────────────────────────────────────────────────

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, pos_weight: float = 1.0) -> torch.Tensor:
    pw    = torch.tensor(pos_weight, device=logits.device)
    bce   = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    p_t   = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    # Per-image Dice, then average — prevents sparse foreground from being diluted batch-wide.
    num = 2 * (probs * targets).sum(dim=(1, 2, 3)) + eps
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.3, gamma: float = 2.0,
                  pos_weight: float = 1.0) -> torch.Tensor:
    return (alpha * focal_loss(logits, targets, gamma, pos_weight)
            + (1 - alpha) * dice_loss(logits, targets))


# ── Augmentations ────────────────────────────────────────────────────────────
# All ops accept/return float32 (H,W) numpy arrays in [0,1].

def _random_affine(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w  = img.shape
    angle = random.uniform(-10, 10)
    tx    = random.uniform(-0.05, 0.05) * w
    ty    = random.uniform(-0.05, 0.05) * h
    scale = random.uniform(0.9, 1.1)
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img, mask


def _contrast_normalize(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)
    return img


def _gaussian_blur(img: np.ndarray) -> np.ndarray:
    ksize = random.choice([3, 5])
    sigma = random.uniform(0.5, 2.0)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _additive_noise(img: np.ndarray) -> np.ndarray:
    return np.clip(img + np.random.uniform(-0.05, 0.05, img.shape).astype(np.float32), 0.0, 1.0)


def _gaussian_noise(img: np.ndarray) -> np.ndarray:
    return np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0.0, 1.0)


def _frequency_noise(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
    fft     = np.fft.fft2(img)
    noise   = (np.random.randn(*img.shape) + 1j * np.random.randn(*img.shape)) * strength
    noisy   = np.real(np.fft.ifft2(fft + noise * np.abs(fft))).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def _elastic(img: np.ndarray, mask: np.ndarray,
             alpha: float = 30.0, sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    h, w  = img.shape
    dx    = gaussian_filter(np.random.randn(h, w), sigma).astype(np.float32) * alpha
    dy    = gaussian_filter(np.random.randn(h, w), sigma).astype(np.float32) * alpha
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(xs + dx, 0, w - 1)
    map_y = np.clip(ys + dy, 0, h - 1)
    img  = cv2.remap(img,  map_x, map_y, cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return img, mask


def augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.75:                        # (a) affine
        img, mask = _random_affine(img, mask)
    if random.random() < 0.25:                        # (b) contrast
        img = _contrast_normalize(img)
    if random.random() < 0.25:                        # (c) blur
        img = _gaussian_blur(img)
    if random.random() < 0.33:                        # (d) noise / elastic
        choice = random.randint(0, 3)
        if choice == 0:
            img = _additive_noise(img)
        elif choice == 1:
            img = _gaussian_noise(img)
        elif choice == 2:
            img = _frequency_noise(img)
        else:
            img, mask = _elastic(img, mask)
    return img.astype(np.float32), mask.astype(np.float32)


# ── Dataset ──────────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    return Path(name).stem.replace(".", "-")


def _build_mask(polys: list, img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polys:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


class CocoSegDataset(Dataset):
    def __init__(self, coco_json, spec_dir, indices, image_size=DEFAULT_IMAGE_SIZE, augment_fn=None):
        self.image_size = image_size
        self.augment_fn = augment_fn

        with open(coco_json) as f:
            coco = json.load(f)

        disk: dict[str, Path] = {
            _norm_name(p.name): p
            for p in Path(spec_dir).iterdir()
            if p.suffix.lower() == ".png"
        }

        # category_id 1 ("segmented") uses RLE encoding — decode with pycocotools.
        rles_by_id: dict[int, list] = {}
        for ann in coco["annotations"]:
            if ann["category_id"] == 1 and isinstance(ann["segmentation"], dict):
                rles_by_id.setdefault(ann["image_id"], []).append(ann["segmentation"])

        # Only keep images that have at least one segmentation mask on disk.
        items: list[tuple[Path, list, int, int]] = []
        for im in coco["images"]:
            if im["id"] not in rles_by_id:
                continue
            orig = im.get("extra", {}).get("name", im["file_name"])
            path = disk.get(_norm_name(orig))
            if path is None:
                continue
            items.append((path, rles_by_id[im["id"]], im["height"], im["width"]))

        self._items = [items[i] for i in indices if i < len(items)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        path, polys, h, w = self._items[idx]
        img_np = load_spectrogram_tensor(path, self.image_size).squeeze(0).numpy()
        # Merge all RLE masks for this image into one binary mask, then resize.
        combined = np.zeros((h, w), dtype=np.uint8)
        for rle in polys:  # variable reused: now holds list of RLE dicts
            combined = np.maximum(combined, mask_util.decode(rle))
        mask_np = cv2.resize(combined, self.image_size,
                             interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if self.augment_fn is not None:
            img_np, mask_np = self.augment_fn(img_np, mask_np)
        return (torch.from_numpy(img_np).unsqueeze(0),
                torch.from_numpy(mask_np).unsqueeze(0))


# ── Inference helpers (shared with run_squeakout_raw.py) ─────────────────────

def write_spectrogram_img(audio, sr, path):
    _, _, Pxx = scipy_spectrogram(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Pxx_dB    = np.flipud(10 * np.log10(Pxx + 1e-12))
    a = np.clip(Pxx_dB, SPEC_LO, SPEC_HI)
    a = (a - SPEC_LO) / (SPEC_HI - SPEC_LO + 1e-12)
    a = (a * 255).astype(np.uint8)
    cv2.imwrite(str(path), a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return a.shape  # (h, w)


def clean_mask(mask, min_area_ratio=0.1, min_component=100, min_total=300):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas  = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        return mask
    rel_thr = max(areas) * min_area_ratio
    clean   = np.zeros_like(mask)
    for label, area in enumerate(areas, start=1):
        if area >= rel_thr and area >= min_component:
            clean[labels == label] = 255
    return clean if (clean > 0).sum() >= min_total else np.zeros_like(mask)


def run_inference(paths, model, device, batch_size=INFERENCE_BATCH_SIZE):
    masks = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch   = paths[i:i + batch_size]
            tensors = torch.stack(
                [load_spectrogram_tensor(p, DEFAULT_IMAGE_SIZE) for p in batch]
            ).to(device)
            logits  = model(tensors)
            masks.extend(logits_to_mask(l, threshold=DEFAULT_MASK_THRESHOLD) for l in logits)
    return masks


def mask_to_polygons(mask, img_shape):
    mh, mw = mask.shape
    h,  w  = img_shape
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


# ── Build dataset splits ──────────────────────────────────────────────────────

with open(args.coco_json) as f:
    _meta = json.load(f)

# Count only images that have cat-1 RLE masks (the dataset filters internally).
n_masked = sum(
    1 for im in _meta["images"]
    if any(a["category_id"] == 1 and isinstance(a["segmentation"], dict)
           for a in _meta["annotations"] if a["image_id"] == im["id"])
)
# Build split indices over the masked-only pool.
indices = list(range(n_masked))
random.shuffle(indices)
n_val     = max(1, int(n_masked * args.val_frac))
train_idx = indices[n_val:]
val_idx   = indices[:n_val]
print(f"Dataset: {n_masked} masked images → {len(train_idx)} train / {len(val_idx)} val")

train_ds = CocoSegDataset(args.coco_json, gt_spec_dir, train_idx, augment_fn=augment)
val_ds   = CocoSegDataset(args.coco_json, gt_spec_dir, val_idx,   augment_fn=None)

# Estimate pos_weight from training masks, capped at 20 to avoid over-penalising negatives.
# The focal loss already handles within-image imbalance; pos_weight corrects the fg/bg ratio.
_pos = _neg = 0
for _, mask in train_ds:
    _pos += mask.sum().item()
    _neg += (1 - mask).sum().item()
pos_weight = min(_neg / max(_pos, 1), 20.0)
print(f"pos_weight={pos_weight:.1f}  (fg={_pos:.0f} px, bg={_neg:.0f} px, raw ratio={_neg/max(_pos,1):.0f})")

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────

device = resolve_device()
ckpt_pretrained = Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
model  = load_model(ckpt_pretrained, device=device)

if args.freeze_backbone:
    for p in model.backbone.parameters():
        p.requires_grad = False
    print("Backbone frozen — fine-tuning decoder only")

model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.lr / 100
)

# ── Training loop ─────────────────────────────────────────────────────────────

best_val_loss  = float("inf")
best_ckpt_path = out_dir / "squeakout_finetuned.ckpt"
log_path       = out_dir / "train_log.csv"

with open(log_path, "w") as f:
    f.write("epoch,train_loss,train_focal,train_dice,val_loss,val_focal,val_dice,lr\n")

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = train_fl = train_dl = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        fl = focal_loss(logits, masks, pos_weight=pos_weight)
        dl = dice_loss(logits, masks)
        loss = 0.3 * fl + 0.7 * dl
        loss.backward()
        optimizer.step()
        n = len(imgs)
        train_loss += loss.item() * n
        train_fl   += fl.item()   * n
        train_dl   += dl.item()   * n
    train_loss /= len(train_ds)
    train_fl   /= len(train_ds)
    train_dl   /= len(train_ds)

    model.eval()
    val_loss = val_fl = val_dl = 0.0
    with torch.inference_mode():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            fl = focal_loss(logits, masks, pos_weight=pos_weight)
            dl = dice_loss(logits, masks)
            n = len(imgs)
            val_loss += (0.3 * fl + 0.7 * dl).item() * n
            val_fl   += fl.item() * n
            val_dl   += dl.item() * n
    val_loss /= len(val_ds)
    val_fl   /= len(val_ds)
    val_dl   /= len(val_ds)

    lr_now = scheduler.get_last_lr()[0]
    scheduler.step()

    print(f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f} (fl={train_fl:.3f} dl={train_dl:.3f})"
          f"  val={val_loss:.4f} (fl={val_fl:.3f} dl={val_dl:.3f})  lr={lr_now:.2e}")
    with open(log_path, "a") as f:
        f.write(f"{epoch},{train_loss:.6f},{train_fl:.6f},{train_dl:.6f},"
                f"{val_loss:.6f},{val_fl:.6f},{val_dl:.6f},{lr_now:.2e}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                   best_ckpt_path)
        print(f"  ✓ checkpoint (val={val_loss:.4f}) → {best_ckpt_path}")

print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")

# ── Inference: write spectrograms + COCO JSONs ───────────────────────────────

print("\nRunning inference on recording…")
model.eval()

das_matches = sorted(glob(str(recording_dir / "*annotations_gt.csv")))
vox_rows = pd.read_csv(das_matches[0])
vox_rows = vox_rows[vox_rows["name"] == "vox"]

for ch in channels:
    wav_matches = sorted(glob(str(recording_dir / f"headmic_{ch}_*.wav")))
    if not wav_matches:
        print(f"channel {ch}: no WAV found, skipping")
        continue

    sr, audio = wavfile.read(wav_matches[0])
    base_name = Path(wav_matches[0]).stem
    chunk_len = int(args.chunk_sec * sr)
    n_chunks  = len(audio) // chunk_len

    das_by_chunk: dict[int, list] = {i: [] for i in range(n_chunks)}
    for vox_idx, orig_start, orig_stop in zip(
        vox_rows.index, vox_rows.start_seconds, vox_rows.stop_seconds
    ):
        if math.isnan(orig_start) or math.isnan(orig_stop):
            continue
        das_start = orig_start - SYNC_PAD / 2
        das_stop  = orig_stop  + SYNC_PAD / 2
        for i in range(n_chunks):
            if das_start < (i + 1) * args.chunk_sec and das_stop > i * args.chunk_sec:
                das_by_chunk[i].append({
                    "vox_idx":   int(vox_idx),
                    "start_sec": das_start,
                    "stop_sec":  das_stop,
                })

    paths, shapes = [], []
    for i in range(n_chunks):
        start_i = i * chunk_len
        stop_i  = start_i + chunk_len
        p = spec_dir / f"{base_name}_chunk_{i:05d}_t{start_i/sr:.2f}-{stop_i/sr:.2f}.png"
        shapes.append(write_spectrogram_img(audio[start_i:stop_i], sr, p))
        paths.append(p)

    print(f"channel {ch}: {n_chunks} chunks")
    masks = [clean_mask(m) for m in run_inference(paths, model, device)]

    coco = {
        "info": {
            "description":   f"SqueakOut finetuned detections — ch {ch}",
            "recording_dir": str(recording_dir),
            "chunk_sec":     args.chunk_sec,
            "spec_params":   {"nperseg": NPERSEG, "noverlap": NOVERLAP,
                              "lo": SPEC_LO, "hi": SPEC_HI},
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
            "id": i, "file_name": path.name,
            "width": w, "height": h,
            "window_start_sec": i * args.chunk_sec,
            "window_end_sec":   (i + 1) * args.chunk_sec,
            "das_events":       das_by_chunk[i],
        })
        for poly in mask_to_polygons(mask, shape):
            pts = np.array(poly).reshape(-1, 2)
            px1, py1 = pts.min(axis=0).tolist()
            px2, py2 = pts.max(axis=0).tolist()
            coco["annotations"].append({
                "id": ann_id, "image_id": i, "category_id": 1,
                "segmentation": [poly],
                "bbox":  [px1, py1, px2 - px1, py2 - py1],
                "area":  float(cv2.contourArea(
                    np.array(poly, dtype=np.float32).reshape(-1, 1, 2))),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_path = out_dir / f"coco_ch_{ch}.json"
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"channel {ch}: {ann_id} annotations → {coco_path}")

print(f"Saved spectrograms to {spec_dir}")
