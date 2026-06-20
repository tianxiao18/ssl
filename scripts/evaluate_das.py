"""
Compare SqueakOut predictions against DAS ground-truth annotations.

Metrics computed at two levels:

  Event-level  — for each DAS vocalization event, did SqueakOut fire in the
                 corresponding time columns? → Detection Rate (recall)

  Chunk-level  — for chunks with *no* DAS overlap, what fraction does SqueakOut
                 activate on? → False Positive Rate

Reports for pretrained and finetuned models, per channel and pooled.
Also produces a ROC-style Precision–Recall / Det-Rate vs FPR plot swept over thresholds.

Usage:
    python scripts/evaluate_das.py \
        <spec_dir> <das_csv> \
        [--finetuned  outputs/.../squeakout_finetuned.ckpt] \
        [--channels   118,35] \
        [--threshold  0.3] \
        [--out-dir    outputs/.../das_eval]
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "squeakout"))
from squeakout import load_model, resolve_device
from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("spec_dir")
parser.add_argument("das_csv")
parser.add_argument("--finetuned",  default=None)
parser.add_argument("--channels",   default="118,35")
parser.add_argument("--threshold",  type=float, default=0.3)
parser.add_argument("--out-dir",    default="outputs/das_eval")
args = parser.parse_args()

channels = [int(c) for c in args.channels.split(",")]
spec_dir = Path(args.spec_dir)
out_dir  = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

W = DEFAULT_IMAGE_SIZE[1]   # 512 — pixel width of model input

# ── Load models ───────────────────────────────────────────────────────────────

device = resolve_device()

pre_ckpt   = Path(__file__).resolve().parents[1] / "squeakout" / "squeakout_weights.ckpt"
pretrained = load_model(pre_ckpt, device=device)
pretrained.eval()

ft_ckpt   = Path(args.finetuned) if args.finetuned else pre_ckpt
finetuned = load_model(ft_ckpt, device=device)
finetuned.eval()

# ── Load DAS events ───────────────────────────────────────────────────────────

das_df  = pd.read_csv(args.das_csv)
das_vox = das_df[das_df["name"] == "vox"].copy()
das_vox = das_vox.sort_values("start_seconds").reset_index(drop=True)
print(f"DAS events: {len(das_vox)}  ({das_vox.start_seconds.min():.1f}–"
      f"{das_vox.stop_seconds.max():.1f} s)")

# ── Parse chunk filenames ─────────────────────────────────────────────────────

CHUNK_RE = re.compile(
    r"headmic_(\d+)_file_\d+_chunk_(\d+)_t([\d.]+)-([\d.]+)\.png"
)

def parse_chunk(path: Path):
    m = CHUNK_RE.match(path.name)
    if not m:
        return None
    return int(m[1]), int(m[2]), float(m[3]), float(m[4])


# ── Run inference and collect column-level scores ─────────────────────────────
# For each chunk: run both models → 512-wide probability vectors (max over freq axis)

def col_scores(model, img_tensor: torch.Tensor) -> np.ndarray:
    """Return (W,) array of max-over-frequency probabilities."""
    with torch.inference_mode():
        logits = model(img_tensor.to(device))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()   # (H, W)
    return prob.max(axis=0)                                 # (W,)


def das_col_mask(t_start: float, t_end: float) -> np.ndarray:
    """Boolean (W,) mask — True where any DAS event is active."""
    dur  = t_end - t_start
    mask = np.zeros(W, dtype=bool)
    events = das_vox[
        (das_vox["start_seconds"] < t_end) &
        (das_vox["stop_seconds"]  > t_start)
    ]
    for _, row in events.iterrows():
        x0 = int((max(row.start_seconds, t_start) - t_start) / dur * W)
        x1 = int((min(row.stop_seconds,  t_end)   - t_start) / dur * W)
        x0 = max(0, x0); x1 = min(W, max(x1, x0 + 1))
        mask[x0:x1] = True
    return mask


all_chunks = [p for p in spec_dir.iterdir() if p.suffix == ".png" and parse_chunk(p)]

# Collect per-channel results: list of (das_mask, pre_scores, ft_scores) per chunk
results_by_ch: dict[int, list] = {ch: [] for ch in channels}

print("Running inference…")
for path in sorted(all_chunks, key=lambda p: (parse_chunk(p)[0], parse_chunk(p)[1])):
    ch, _, t0, t1 = parse_chunk(path)
    if ch not in results_by_ch:
        continue
    img_t   = load_spectrogram_tensor(path, DEFAULT_IMAGE_SIZE).unsqueeze(0)
    das_m   = das_col_mask(t0, t1)
    pre_s   = col_scores(pretrained, img_t)
    ft_s    = col_scores(finetuned,  img_t)
    results_by_ch[ch].append((das_m, pre_s, ft_s, t0, t1))

# ── Compute metrics at a given threshold ──────────────────────────────────────

def metrics_at_threshold(results, thr):
    """
    Column-level metrics.
    Returns dict with detection_rate (recall), fpr, precision, f1.
    """
    das_pos  = np.concatenate([r[0]           for r in results])   # bool (N_cols,)
    pre_pred = np.concatenate([r[1] >= thr     for r in results])
    ft_pred  = np.concatenate([r[2] >= thr     for r in results])

    rows = {}
    for name, pred in [("pretrained", pre_pred), ("finetuned", ft_pred)]:
        tp = ( das_pos &  pred).sum()
        fp = (~das_pos &  pred).sum()
        fn = ( das_pos & ~pred).sum()
        tn = (~das_pos & ~pred).sum()
        dr   = tp / (tp + fn + 1e-9)
        fpr  = fp / (fp + tn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1   = 2 * prec * dr / (prec + dr + 1e-9)
        rows[name] = dict(detection_rate=dr, fpr=fpr, precision=prec, f1=f1,
                          tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    return rows


# ── Event-level detection rate ────────────────────────────────────────────────

def event_detection_rate(results, thr):
    """
    Per-DAS-event: detected if any column in the event's pixel range exceeds thr.
    Returns (pretrained_rate, finetuned_rate, n_events).
    """
    # pool all chunks into time-indexed score arrays
    # build t->scores mapping from results
    chunk_data = [(r[3], r[4], r[1], r[2]) for r in results]  # t0, t1, pre, ft

    n_events = 0
    pre_hits = ft_hits = 0

    for _, row in das_vox.iterrows():
        ev_start, ev_stop = row.start_seconds, row.stop_seconds
        hit_pre = hit_ft = False
        for t0, t1, pre_s, ft_s in chunk_data:
            if ev_start >= t1 or ev_stop <= t0:
                continue
            dur = t1 - t0
            x0 = int((max(ev_start, t0) - t0) / dur * W)
            x1 = int((min(ev_stop,  t1) - t0) / dur * W)
            x0 = max(0, x0); x1 = min(W, max(x1, x0 + 1))
            if pre_s[x0:x1].max() >= thr:
                hit_pre = True
            if ft_s[x0:x1].max() >= thr:
                hit_ft = True
        n_events += 1
        pre_hits += int(hit_pre)
        ft_hits  += int(hit_ft)

    return pre_hits / max(n_events, 1), ft_hits / max(n_events, 1), n_events


# ── Print summary ─────────────────────────────────────────────────────────────

thr = args.threshold
all_results = [r for ch in channels for r in results_by_ch[ch]]

print(f"\n{'─'*60}")
print(f"Threshold = {thr}")
print(f"{'─'*60}")

print("\n── Column-level metrics (pooled across channels) ──")
col_m = metrics_at_threshold(all_results, thr)
for model_name, m in col_m.items():
    print(f"  {model_name:<12}  DR={m['detection_rate']:.3f}  FPR={m['fpr']:.3f}  "
          f"Prec={m['precision']:.3f}  F1={m['f1']:.3f}"
          f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

print("\n── Per-channel column-level metrics ──")
for ch in channels:
    m = metrics_at_threshold(results_by_ch[ch], thr)
    for mn, mv in m.items():
        print(f"  ch{ch}  {mn:<12}  DR={mv['detection_rate']:.3f}  FPR={mv['fpr']:.3f}  "
              f"Prec={mv['precision']:.3f}  F1={mv['f1']:.3f}")

print("\n── Event-level detection rate ──")
pre_dr, ft_dr, n_ev = event_detection_rate(all_results, thr)
print(f"  {n_ev} DAS events total")
print(f"  pretrained  detected {pre_dr*100:.1f}%")
print(f"  finetuned   detected {ft_dr*100:.1f}%")

# ── ROC-style curve: Detection Rate vs FPR across thresholds ─────────────────

thresholds = np.linspace(0.0, 1.0, 101)
pre_dr_v, ft_dr_v, pre_fpr_v, ft_fpr_v = [], [], [], []
pre_prec_v, ft_prec_v = [], []

for t in thresholds:
    m = metrics_at_threshold(all_results, t)
    pre_dr_v.append(m["pretrained"]["detection_rate"])
    ft_dr_v.append( m["finetuned"]["detection_rate"])
    pre_fpr_v.append(m["pretrained"]["fpr"])
    ft_fpr_v.append( m["finetuned"]["fpr"])
    pre_prec_v.append(m["pretrained"]["precision"])
    ft_prec_v.append( m["finetuned"]["precision"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
session = Path(args.spec_dir).resolve().parents[0].name
fig.suptitle(f"SqueakOut vs DAS — column-level detection\n"
             f"{session}  ·  ch {','.join(str(c) for c in channels)}",
             fontsize=12, fontweight="bold")

# Panel 1: ROC (Detection Rate vs FPR)
ax = axes[0]
ax.plot(pre_fpr_v, pre_dr_v, color="#4C72B0", label="pretrained", linewidth=2)
ax.plot(ft_fpr_v,  ft_dr_v,  color="#DD8452", label="finetuned",  linewidth=2)
ax.axvline(args.threshold, color="gray", linestyle=":", linewidth=1, label=f"thr={thr}")
# mark operating point
m_op = metrics_at_threshold(all_results, thr)
ax.scatter([m_op["pretrained"]["fpr"]], [m_op["pretrained"]["detection_rate"]],
           color="#4C72B0", s=80, zorder=5)
ax.scatter([m_op["finetuned"]["fpr"]],  [m_op["finetuned"]["detection_rate"]],
           color="#DD8452", s=80, zorder=5)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("Detection Rate (Recall)")
ax.set_title("ROC curve")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

# Panel 2: Precision–Recall
ax = axes[1]
ax.plot(pre_dr_v, pre_prec_v, color="#4C72B0", label="pretrained", linewidth=2)
ax.plot(ft_dr_v,  ft_prec_v,  color="#DD8452", label="finetuned",  linewidth=2)
ax.scatter([m_op["pretrained"]["detection_rate"]], [m_op["pretrained"]["precision"]],
           color="#4C72B0", s=80, zorder=5)
ax.scatter([m_op["finetuned"]["detection_rate"]],  [m_op["finetuned"]["precision"]],
           color="#DD8452", s=80, zorder=5)
ax.set_xlabel("Recall (Detection Rate)")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall curve")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

fig.tight_layout()
out_path = out_dir / "das_eval.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved {out_path}")
