"""Mongolia AudioMoth call-type figure from labeled candidates of mongolia_audiomoth_k4_hp15k (whiteboard layout).

call_types.{png,pdf}: putative call types, two examples each (campaign label = call).
background_patterns.{png,pdf}: common not-a-call patterns, two examples each, named neutrally for review.
examples.csv: source WAV, times, campaign label, detectors, shape basis and flag per panel.

Call types are proposed from contour shape only (no call-type annotations exist). Every panel is a WIN-second window
centred on the candidate, same STFT and dB scale throughout.

    python scripts/evaluation/plot_call_type_figure.py --out outputs/eval/mongolia_call_types
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "outputs/label_campaigns/mongolia_audiomoth_k4_hp15k"
SRC = "/mnt/ceph/users/neurostatslab/saneslab_data/mongolia/Audiomoth_Files/Colony_Vox/"
NPS, NOV, VMIN, VMAX, WIN, FHI = 256, 224, -110, -72, 0.25, 60

# (figure, row, group, cand_id, basis, flag)
PICKS = [
    ("calls", 0, "Alarm", 27158, "arch 27-32-27 kHz", "putative type"),
    ("calls", 0, "Alarm", 26148, "low arch 25-28-24 kHz, harmonic ~53 kHz", "putative type; flatter top than 27158"),
    ("calls", 0, "UFM", 5256, "flat ~29 kHz then upward sweep to ~34 kHz", "putative type"),
    ("calls", 0, "UFM", 20737, "upward sweep ~28-33 kHz", "putative type"),
    ("calls", 0, "Warble", 3134, "sinusoidal FM ~6 cycles, rising 33-43 kHz", "putative type; bout of similar calls 5553-5558 s"),
    ("calls", 0, "Warble", 23191, "sinusoidal FM, 34-40 kHz", "putative type"),
    ("calls", 0, "Down FM", 1979, "downward sweep ~32-25 kHz", "putative type"),
    ("calls", 0, "Down FM", 8361, "shallow downward sweep ~29-26 kHz", "putative type; faint"),
    ("calls", 1, "U-shape", 9240, "U contour 35-29-35 kHz", "putative type"),
    ("calls", 1, "U-shape", 5863, "U contour 36-32-34 kHz", "putative type"),
    ("calls", 1, "Short flat", 25687, "flat ~40 kHz, 34 ms", "putative type"),
    ("calls", 1, "Short flat", 26996, "flat ~37 kHz, 51 ms", "putative type"),
    ("calls", 1, "Long flat", 14238, "flat ~25 kHz, 213 ms, harmonic ~51 kHz", "putative type"),
    ("calls", 1, "Long flat", 26625, "flat ~25 kHz, 158 ms, harmonic ~50 kHz", "putative type"),
    ("calls", 1, "Step-down", 26422, "flat ~31 kHz, steps down to ~29 then falls to ~26 kHz", "putative type; loud"),
    ("calls", 1, "Step-down", 26427, "flat ~31 kHz, steps down to ~29 then falls to ~27 kHz", "putative type; loud"),
    ("calls", 2, "Short loud", 27038, "loud 33 ms block ~22-30 kHz, arched harmonic ~48-51 kHz", "putative type"),
    ("calls", 2, "Short loud", 14570, "loud 34 ms block ~20-28 kHz, arched harmonic ~46-48 kHz", "putative type"),
    ("calls", 2, "23 kHz tone", 5318, "short flat ~24 kHz inside the noise band, 33 ms", "putative type; shape close to not-a-call pattern B"),
    ("calls", 2, "23 kHz tone", 19575, "short flat ~23 kHz inside the noise band, 27 ms", "putative type; shape close to not-a-call pattern B"),
    ("calls", 2, "Long high arch", 2590, "long arch ~33-38-33 kHz, 165 ms, over broadband noise", "putative type; rare (3 found); noisy"),
    ("calls", 2, "Long high arch", 14733, "long arch/rise ~34-45 kHz, 330 ms (centre 250 ms shown), over broadband noise", "putative type; rare (3 found); noisy"),
    ("background", 0, "A", 22308, "stack of parallel tones ~8-40 kHz, ~150 ms, hooked ends", "unnamed; same hour on 2 recorders"),
    ("background", 0, "A", 24273, "stack of parallel tones ~8-40 kHz, ~150 ms, hooked ends", "unnamed; same hour on 2 recorders"),
    ("background", 0, "B", 7976, "brief blip ~20-23 kHz, < 30 ms", "unnamed"),
    ("background", 0, "B", 21597, "brief blip ~20-23 kHz, < 50 ms", "unnamed"),
    ("background", 0, "C", 23066, "steep downward line ~35 to 5 kHz", "unnamed"),
    ("background", 0, "C", 21683, "steep downward line ~30 to 5 kHz", "unnamed"),
    ("background", 1, "D", 20852, "train of harmonic arches, fundamental ~7 kHz", "unnamed"),
    ("background", 1, "D", 19330, "train of harmonic arches, fundamental ~6 kHz", "unnamed"),
    ("background", 1, "E", 7644, "dense harmonic bursts, fundamental < 10 kHz", "unnamed"),
    ("background", 1, "E", 15036, "dense harmonic bursts, fundamental < 10 kHz", "unnamed"),
    ("background", 1, "F", 10592, "single broadband click, < 10 ms", "unnamed"),
    ("background", 1, "F", 10524, "single broadband click, < 10 ms", "unnamed"),
    ("background", 2, "G", 6104, "regular train of faint ticks", "unnamed"),
    ("background", 2, "G", 6100, "regular train of faint ticks", "unnamed"),
    ("background", 2, "H", 20011, "loud broadband burst, banded, ~60 ms", "unnamed"),
    ("background", 2, "H", 21919, "loud broadband burst with sweeping bands, ~80 ms", "unnamed"),
    ("background", 2, "I", 19635, "low flat tone with harmonics, fundamental < 10 kHz", "unnamed"),
    ("background", 2, "I", 12547, "low flat tone with harmonics, fundamental < 10 kHz", "unnamed"),
]


def labels():
    lab = {}
    for line in open(CAMPAIGN / "labels.jsonl"):
        r = json.loads(line)
        if r.get("retracted"):
            lab.pop(r["cand_id"], None)
        else:
            lab[r["cand_id"]] = (int(r["label"]), r["annotator"])
    return lab


def spec(clip, t0, t1):
    wav = ROOT / "data/mongolia_audiomoth" / clip / f"mic_0_{clip.split('/')[-1]}.wav"
    with sf.SoundFile(wav) as fh:
        sr = fh.samplerate
        fh.seek(int(t0 * sr))
        x = fh.read(int((t1 - t0) * sr), dtype="float32")
    f, _, P = spectrogram(x, fs=sr, nperseg=NPS, noverlap=NOV)
    return f / 1000, 10 * np.log10(P + 1e-14), os.readlink(wav).removeprefix(SRC), float(np.abs(x).max())


def draw(panels, n_rows, path, title_fn):
    """panels: [(row, group, cid, S, f, (s, e) call span in ms)]; equal-width panels, a gap column between groups."""
    rows = [[p for p in panels if p[0] == r] for r in range(n_rows)]
    layout = []
    for P in rows:
        cols = []
        for k, p in enumerate(P):
            if k and p[1] != P[k - 1][1]:
                cols.append(None)
            cols.append(p)
        layout.append(cols)
    ncol = max(len(c) for c in layout)
    widths = [0.25 if any(i < len(c) and c[i] is None for c in layout) else 1 for i in range(ncol)]
    fig, axes = plt.subplots(n_rows, ncol, figsize=(1.35 * sum(widths) + 0.8, 2.1 * n_rows), facecolor="#ffffff",
                             squeeze=False, gridspec_kw={"width_ratios": widths, "wspace": 0.08, "hspace": 0.5})
    im = None
    for r, cols in enumerate(layout):
        for i in range(ncol):
            ax = axes[r, i]
            p = cols[i] if i < len(cols) else None
            if p is None:
                ax.axis("off")
                continue
            _, group, cid, S, f, (s, e) = p
            im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=VMIN, vmax=VMAX,
                           extent=[0, WIN * 1000, f[0], f[-1]], interpolation="nearest")
            ax.axhline(24, color="#ffffff", lw=0.5, ls=(0, (3, 3)), alpha=0.6)
            ax.plot([max(s, 0), min(e, WIN * 1000)], [1.2, 1.2], color="#22d3ee", lw=2.2, solid_capstyle="butt")  # candidate span
            ax.set_xlim(0, WIN * 1000)
            ax.set_ylim(0, FHI)
            ax.set_xticks([0, 100, 200])
            ax.tick_params(axis="x", labelsize=6, length=2, pad=1)
            ax.set_yticks([0, 24, 48] if i == 0 else [])
            if i == 0:
                ax.set_ylabel("kHz")
            if i == 0 or cols[i - 1] is None:
                ax.text(0, 1.09, title_fn(group), transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
            if r == n_rows - 1:
                ax.set_xlabel("Time (ms)", fontsize=7, labelpad=1)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.set_label("dB", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(CAMPAIGN / "candidates.csv").set_index("cand_id")
    lab = labels()
    zcols = [c for c in pool.columns if c.startswith("z_")]
    rows, panels = [], {"calls": [], "background": []}
    for fig_name, row, group, cid, basis, flag in PICKS:
        r = pool.loc[cid]
        clip = r["clip"]
        t0 = max(0.0, (r["t_start"] + r["t_end"]) / 2 - WIN / 2)
        f, S, src, peak = spec(clip, t0, t0 + WIN)
        panels[fig_name].append((row, group, cid, S, f, ((r["t_start"] - t0) * 1000, (r["t_end"] - t0) * 1000)))
        y, who = lab.get(cid, (None, None))
        rows.append({"figure": fig_name, "group": group, "cand_id": cid, "source_wav": src,
                     "t_start_s": round(r["t_start"], 4), "t_end_s": round(r["t_end"], 4),
                     "duration_ms": round((r["t_end"] - r["t_start"]) * 1000, 1),
                     "window_s": f"{t0:.4f}-{t0 + WIN:.4f}",
                     "campaign_label": {1: "call", 0: "not a call", None: "unlabeled"}[y], "annotator": who,
                     "detectors": "+".join(z[2:] for z in zcols if r[z]), "peak_abs_sample": round(peak, 3),
                     "shape": basis, "flag": flag})
    pd.DataFrame(rows).to_csv(out / "examples.csv", index=False)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    draw(panels["calls"], 3, out / "call_types", lambda g: g)
    draw(panels["background"], 3, out / "background_patterns", lambda g: f"Pattern {g}")
    print("->", out)


if __name__ == "__main__":
    main()
