"""
Diagnostic: intensity distribution for a spectrogram image.

Usage:
    python scripts/diagnose_column_threshold.py <spectrogram.png>

Saves to /mnt/home/the10/ssl/outputs/:
  *_column_dists.png   — full-image intensity histogram
"""

import sys
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/home/the10/ssl/outputs"


def diagnose(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    gray = gray.astype(np.float32)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, os.path.basename(img_path).rsplit(".", 1)[0])

    all_vals = gray.ravel()
    mean_val = np.mean(all_vals)
    thresh_96 = np.percentile(all_vals, 96)
    bright_vals = all_vals[all_vals > np.percentile(all_vals, 99)]
    bright_lo, bright_hi = bright_vals.min(), bright_vals.max()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_vals, bins=80, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(mean_val,  color="orange", lw=1.5, linestyle="--", label=f"mean={mean_val:.1f}")
    ax.axvline(thresh_96, color="red",    lw=1.5, linestyle="--", label=f"96th pct={thresh_96:.1f}")
    ax.axvspan(bright_lo, bright_hi, color="yellow", alpha=0.25,
               label=f"top-1% range [{bright_lo:.0f}–{bright_hi:.0f}]")
    ax.legend(fontsize=8)
    ax.set_xlabel("pixel intensity")
    ax.set_ylabel("count")
    ax.set_title("Intensity distribution (full image)")
    plt.tight_layout()
    out1 = stem + "_column_dists.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")
    print(f"Image: {gray.shape}")


def diagnose_columns(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    gray = gray.astype(np.float32)
    H, W = gray.shape

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, os.path.basename(img_path).rsplit(".", 1)[0])

    # --- Plot 1: per-column intensity distribution ---
    n_bins = 64
    bin_edges = np.linspace(0, 255, n_bins + 1)
    hist2d = np.zeros((n_bins, W), dtype=np.float32)
    for col in range(W):
        counts, _ = np.histogram(gray[:, col], bins=bin_edges)
        hist2d[:, col] = counts

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        np.flipud(hist2d),
        aspect="auto",
        extent=[0, W, bin_edges[0], bin_edges[-1]],
        norm=matplotlib.colors.LogNorm(vmin=1),
        cmap="magma",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="pixel count (log)")
    ax.plot(np.arange(W) + 0.5, np.median(gray, axis=0), color="cyan",   lw=1,              label="median")
    ax.plot(np.arange(W) + 0.5, np.mean(gray,   axis=0), color="yellow", lw=1, linestyle="--", label="mean")
    ax.set_xlabel("time (col index)")
    ax.set_ylabel("intensity (0–255)")
    ax.set_title("Per-column intensity distribution")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    out1 = stem + "_col_dist.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # --- Plot 2: intensity by frequency row, one line per column ---
    fig, ax = plt.subplots(figsize=(8, 6))
    rows = np.arange(H)
    cmap = plt.cm.plasma
    for col in range(W):
        color = cmap(col / max(W - 1, 1))
        ax.plot(gray[:, col], rows, color=color, alpha=0.4, lw=0.8)
        peak_row = np.argmax(gray[:, col])
        peak_val = gray[peak_row, col]
        ax.scatter(peak_val, peak_row, color=color, s=20, zorder=3, edgecolors="white", linewidths=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, W - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="col index (time)")
    ax.invert_yaxis()
    ax.set_xlabel("intensity")
    ax.set_ylabel("freq (row index, 0=top)")
    ax.set_title("Intensity by frequency row — each line = one timestamp")
    plt.tight_layout()
    out2 = stem + "_col_freq.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    print(f"Image shape: {gray.shape}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    diagnose(sys.argv[1])
    diagnose_columns(sys.argv[1])
