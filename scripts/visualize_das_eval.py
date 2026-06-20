"""
Evaluate the finetuned SqueakOut on a full session and build comparison montages.
For every 1-second spectrogram chunk the script stacks three rows:

    row 1 — raw spectrogram
    row 2 — DAS annotation mask  (vertical bands at vocalization times)
    row 3 — finetuned SqueakOut prediction

Usage:
    python scripts/visualize_das_eval.py \
        <spec_dir> <das_csv> <out_dir> \
        [--finetuned  outputs/squeakout_finetuned/.../squeakout_finetuned.ckpt] \
        [--channels   118,35] \
        [--chunk-sec  1.0] \
        [--cols       10] \
        [--threshold  0.3]

Example:
    python scripts/visualize_das_eval.py \
        outputs/squeakout_finetuned/exp384_idx000/spectrograms \
        data/experiment_384/idx_000/exp_384_idx_001_ch_all_annotations_gt.csv \
        outputs/squeakout_finetuned/exp384_idx000/montages_das \
        --finetuned outputs/squeakout_finetuned/exp384_idx000/squeakout_finetuned.ckpt
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("spec_dir",   help="directory containing chunk PNG spectrograms")
parser.add_argument("das_csv",    help="DAS annotations CSV (start_seconds, stop_seconds)")
parser.add_argument("out_dir",    help="output directory for montage PNGs")
parser.add_argument("--finetuned",  default=None)
parser.add_argument("--channels",   default="118,35")
parser.add_argument("--chunk-sec",  type=float, default=1.0)
parser.add_argument("--cols",       type=int,   default=10)
parser.add_argument("--threshold",  type=float, default=0.3)
args = parser.parse_args()

channels  = [int(c) for c in args.channels.split(",")]
spec_dir  = Path(args.spec_dir)
out_dir   = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

CELL_W, CELL_H = 300, 257
LABEL_COLOR = (0, 255, 0)   # green label text
DAS_COLOR   = (0, 200, 255) # orange-yellow dashed lines for DAS events
DASH_LEN    = 8             # pixels on
GAP_LEN     = 5             # pixels off

# ── Load model ────────────────────────────────────────────────────────────────

device = resolve_device()

pre_ckpt   = Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
pretrained = load_model(pre_ckpt, device=device)
pretrained.eval()

ft_ckpt  = Path(args.finetuned) if args.finetuned else pre_ckpt
finetuned = load_model(ft_ckpt, device=device)
finetuned.eval()

# ── Load DAS events ───────────────────────────────────────────────────────────

das_df   = pd.read_csv(args.das_csv)
das_vox  = das_df[das_df["name"] == "vox"].copy()
das_vox  = das_vox.sort_values("start_seconds").reset_index(drop=True)

print(f"DAS events: {len(das_vox)} vox annotations  "
      f"({das_vox.start_seconds.min():.1f}–{das_vox.stop_seconds.max():.1f} s)")

# ── Parse spectrogram filenames ───────────────────────────────────────────────
# Expected pattern: headmic_{ch}_file_{n}_chunk_{idx}_t{t0:.2f}-{t1:.2f}.png

CHUNK_RE = re.compile(
    r"headmic_(\d+)_file_\d+_chunk_(\d+)_t([\d.]+)-([\d.]+)\.png"
)

def parse_chunk(path: Path):
    m = CHUNK_RE.match(path.name)
    if not m:
        return None
    ch, idx, t0, t1 = int(m[1]), int(m[2]), float(m[3]), float(m[4])
    return ch, idx, t0, t1


# ── Draw DAS dashed lines on a BGR cell ──────────────────────────────────────

def draw_das_lines(bgr: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    """Draw vertical dashed lines at DAS event start/end times onto bgr (in-place copy)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    dur  = t_end - t_start
    if dur <= 0:
        return out
    events = das_vox[
        (das_vox["start_seconds"] < t_end) &
        (das_vox["stop_seconds"]  > t_start)
    ]
    for _, row in events.iterrows():
        for t_sec in (row.start_seconds, row.stop_seconds):
            if t_sec < t_start or t_sec > t_end:
                continue
            x = int((t_sec - t_start) / dur * w)
            x = max(0, min(w - 1, x))
            y = 0
            while y < h:
                y1 = min(y + DASH_LEN, h)
                out[y:y1, x] = DAS_COLOR
                y += DASH_LEN + GAP_LEN
    return out


# ── Rendering helpers ─────────────────────────────────────────────────────────

def cell(img_gray: np.ndarray, label: str) -> np.ndarray:
    c   = cv2.resize(img_gray, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                LABEL_COLOR, 1, cv2.LINE_AA)
    return bgr


def predict_gray(m, img_tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        logits = m(img_tensor.to(device))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob * 255).clip(0, 255).astype(np.uint8)


def build_column(path: Path, t_start: float, t_end: float, label: str) -> np.ndarray:
    img_tensor = load_spectrogram_tensor(path, DEFAULT_IMAGE_SIZE).unsqueeze(0)
    img_gray   = (img_tensor.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)

    spec_bgr = cv2.cvtColor(
        cv2.resize(img_gray, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA),
        cv2.COLOR_GRAY2BGR,
    )
    spec_bgr = draw_das_lines(spec_bgr, t_start, t_end)
    cv2.putText(spec_bgr, label[:28], (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, LABEL_COLOR, 1, cv2.LINE_AA)

    return np.vstack([
        spec_bgr,
        cell(predict_gray(pretrained, img_tensor), "pretrained"),
        cell(predict_gray(finetuned,  img_tensor), "finetuned"),
    ])


# ── Process each channel ──────────────────────────────────────────────────────

all_chunks = [p for p in spec_dir.iterdir() if p.suffix == ".png"]

for ch in channels:
    ch_chunks = sorted(
        [p for p in all_chunks if parse_chunk(p) and parse_chunk(p)[0] == ch],
        key=lambda p: parse_chunk(p)[1]
    )
    if not ch_chunks:
        print(f"channel {ch}: no spectrograms found, skipping")
        continue

    print(f"channel {ch}: {len(ch_chunks)} chunks → building montages…")

    columns = []
    for path in ch_chunks:
        _, idx, t0, t1 = parse_chunk(path)
        label = f"ch{ch} t{t0:.0f}-{t1:.0f}"
        columns.append(build_column(path, t0, t1, label))

    n_pages = -(-len(columns) // args.cols)
    for page in range(n_pages):
        page_cols = columns[page * args.cols : (page + 1) * args.cols]
        montage   = np.hstack(page_cols)
        out_path  = out_dir / f"ch{ch}_page{page:02d}.png"
        cv2.imwrite(str(out_path), montage)
        print(f"  saved {out_path}  ({len(page_cols)} chunks)")

print(f"\nDone → {out_dir}")
