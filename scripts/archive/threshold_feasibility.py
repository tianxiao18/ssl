"""
Feasibility prototype for the "low volume threshold + manual verify" gold-standard plan.

Question it answers, on real data, for one recording:
    As you lower a volume-signal threshold, how does
        (a) labeling cost  = number of candidate events a human must verify,
        (b) recall ceiling = fraction of true calls the threshold can ever surface,
        (c) candidate junk = fraction of candidates that are NOT real calls,
    trade off against each other?

The "volume signal" is band-limited energy above `freq_min` (USVs are ultrasonic),
pooled across the two head-mic channels (a call loud in EITHER mic is a candidate) —
i.e. the best case for a naive energy detector. Reference truth for recall is the
curated `vox` events in the per-recording *annotations_gt.csv.

CAVEAT (read before trusting the recall axis): the reference `vox` events are
themselves model-seeded (das + segmentation), so this measures recall of the volume
gate *relative to that set*, not against a from-scratch exhaustive sweep. The real
protocol still needs a small exhaustive-from-scratch subset (see the discussion).
This prototype shows the machinery and the SHAPE of the cost/recall/junk curves.

Usage:
    python scripts/threshold_feasibility.py \
        --recording data/experiment_384/idx_000 \
        --gt-csv    data/experiment_384/idx_000/exp_384_idx_001_ch_all_annotations_gt.csv \
        --out       outputs/eval/threshold_feasibility
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── volume signal ──────────────────────────────────────────────────────────────

def band_energy_envelope(wav_path, freq_min, nperseg=256, hop=128):
    """Per-frame log energy in the band above freq_min. Returns (t_centers, env_db)."""
    x, sr = sf.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f, t, Sxx = spectrogram(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
                            scaling="spectrum", mode="psd")
    band = f >= freq_min
    power = Sxx[band, :].sum(axis=0)
    env_db = 10.0 * np.log10(power + 1e-20)
    return t, env_db


# ── candidate extraction ─────────────────────────────────────────────────────────

def extract_events(above, t, hop_s, merge_gap_s, min_dur_s):
    """Boolean per-frame mask -> merged candidate (start, stop) intervals."""
    if not above.any():
        return []
    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        stops = np.r_[stops, len(above)]
    ivals = [(t[s], t[min(e, len(t) - 1)]) for s, e in zip(starts, stops)]
    # merge across small gaps
    merged = []
    for s, e in ivals:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) >= min_dur_s]


def overlaps(a, intervals):
    a0, a1 = a
    return any(min(a1, b1) > max(a0, b0) for b0, b1 in intervals)


def load_gt(gt_csv):
    """Curated vox events [(start, stop)] from a *annotations_gt.csv."""
    df = pd.read_csv(gt_csv)
    return [(float(r.start_seconds), float(r.stop_seconds))
            for r in df[df.name == "vox"].itertuples()
            if r.stop_seconds > r.start_seconds]


def find_gt_csv(rec):
    """Locate the lone *annotations_gt.csv inside a recording dir, or None."""
    hits = sorted(Path(rec).glob("*annotations_gt.csv"))
    return hits[0] if hits else None


def resolve_channel(rec, ch):
    """Resolve a channel prefix (e.g. 'headmic_118') to its wav in rec (suffix varies), or None."""
    hits = sorted(Path(rec).glob(f"{ch}*.wav"))
    return hits[0] if hits else None


def sweep_recording(rec, gt, channels, freq_min, merge_gap_ms, min_dur_ms, n_thresholds):
    """Run the threshold sweep for one recording. Returns (sweep_df, dur_seconds)."""
    # Pooled volume envelope = elementwise max over channels (call loud in EITHER mic)
    envs, t_ref = [], None
    for ch in channels:
        wav = resolve_channel(rec, ch)
        if wav is None:
            continue
        t, env = band_energy_envelope(wav, freq_min)
        if t_ref is None:
            t_ref = t
        n = min(len(t_ref), len(env))
        envs.append(env[:n])
        t_ref = t_ref[:n]
    env = np.maximum.reduce([e[:len(t_ref)] for e in envs])
    hop_s = float(np.median(np.diff(t_ref)))
    dur = float(t_ref[-1])

    lo, hi = np.percentile(env, 50), env.max()
    thresholds = np.linspace(hi, lo, n_thresholds)
    merge_gap_s, min_dur_s = merge_gap_ms / 1e3, min_dur_ms / 1e3

    rows = []
    for thr in thresholds:
        cand = extract_events(env >= thr, t_ref, hop_s, merge_gap_s, min_dur_s)
        n_cand = len(cand)
        recovered = sum(1 for g in gt if overlaps(g, cand))
        recall = recovered / len(gt) if gt else float("nan")
        cand_tp = sum(1 for c in cand if overlaps(c, gt))
        precision = cand_tp / n_cand if n_cand else float("nan")
        rows.append(dict(threshold_db=thr, n_candidates=n_cand, recall=recall,
                         precision=precision, n_recovered=recovered, n_gt=len(gt)))
    return pd.DataFrame(rows), dur


def operating_points(sweep, n_gt, targets=(0.90, 0.95, 0.99, 1.00)):
    """For each target recall, the min-cost row that reaches it (or None)."""
    out = {}
    for tgt in targets:
        ok = sweep[sweep.recall >= tgt]
        out[tgt] = ok.loc[ok.n_candidates.idxmin()] if len(ok) else None
    return out


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--recording", help="single recording dir")
    grp.add_argument("--session", help="experiment dir; sweeps every idx_* with GT + audio")
    ap.add_argument("--gt-csv", help="GT csv for --recording (auto-discovered if omitted)")
    ap.add_argument("--channels", default="headmic_118,headmic_35",
                    help="channel wav prefixes; the file_NNN suffix is resolved per recording")
    ap.add_argument("--freq-min", type=float, default=20000.0)
    ap.add_argument("--merge-gap-ms", type=float, default=15.0)
    ap.add_argument("--min-dur-ms", type=float, default=5.0)
    ap.add_argument("--n-thresholds", type=int, default=60)
    ap.add_argument("--out", default="outputs/eval/threshold_feasibility")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    channels = args.channels.split(",")

    # Build the list of (recording_dir, gt_csv) to sweep.
    if args.recording:
        gt_csv = args.gt_csv or find_gt_csv(args.recording)
        if gt_csv is None:
            raise SystemExit(f"no *annotations_gt.csv in {args.recording}")
        recs = [(Path(args.recording), Path(gt_csv))]
    else:
        recs = []
        for idx in sorted(Path(args.session).glob("idx_*")):
            gt_csv = find_gt_csv(idx)
            has_audio = resolve_channel(idx, channels[0]) is not None
            if gt_csv is not None and has_audio:
                recs.append((idx, gt_csv))
        if not recs:
            raise SystemExit(f"no recordings with GT + audio under {args.session}")

    # Sweep each recording.
    results = []  # (name, sweep_df, n_gt, dur)
    for rec, gt_csv in recs:
        gt = load_gt(gt_csv)
        name = f"{rec.parent.name}/{rec.name}"
        print(f"\n=== {name} ===\nGT vox events: {len(gt)}")
        sweep, dur = sweep_recording(rec, gt, channels, args.freq_min,
                                     args.merge_gap_ms, args.min_dur_ms, args.n_thresholds)
        sweep.insert(0, "recording", name)
        results.append((name, sweep, len(gt), dur))

        print("Target recall  ->  candidates to label   (precision of that pile)")
        for tgt, r in operating_points(sweep, len(gt)).items():
            if r is not None:
                print(f"  recall >= {tgt:.2f}  ->  {int(r.n_candidates):5d} candidates "
                      f"({int(r.n_recovered)}/{len(gt)} real)  cand-precision {r.precision:.2f}  "
                      f"(thr {r.threshold_db:.1f} dB)")
            else:
                print(f"  recall >= {tgt:.2f}  ->  NOT REACHABLE "
                      f"(max recall {sweep.recall.max():.2f})")

    # Persist all sweeps.
    all_sweeps = pd.concat([s for _, s, _, _ in results], ignore_index=True)
    all_sweeps.to_csv(out / "sweep.csv", index=False)

    # ── plots ── (each recording overlaid; single recording is just one curve)
    single = len(results) == 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
    for (name, sweep, n_gt, dur), col in zip(results, cmap):
        lbl = f"{name.split('/')[-1]} ({n_gt})"
        axes[0].plot(sweep.threshold_db, sweep.n_candidates, color=col, label=lbl)
        axes[1].plot(sweep.threshold_db, sweep.recall, color=col, label=lbl)
        if single:
            axes[1].plot(sweep.threshold_db, sweep.precision, color="C3", ls="--",
                         label="cand. precision")
        # money plot: normalize cost by #GT so recordings are comparable
        xcost = sweep.n_candidates if single else sweep.n_candidates / max(n_gt, 1)
        axes[2].plot(xcost, sweep.recall, marker=".", ms=3, color=col, label=lbl)

    axes[0].set(xlabel="threshold (dB)", ylabel="# candidate events (labeling cost)",
                title="Labeling cost vs. threshold")
    axes[1].set(xlabel="threshold (dB)", ylabel="recall ceiling",
                title="Recall ceiling vs. threshold", ylim=(0, 1.05))
    axes[1].axhline(1.0, color="gray", lw=0.6)
    axes[2].set(xlabel=("# candidates to label" if single else "candidates per true call (cost ratio)"),
                ylabel="recall ceiling", title="Recall ceiling vs. labeling cost", ylim=(0, 1.05))
    axes[2].axhline(1.0, color="gray", lw=0.6)
    if single:
        n_gt = results[0][2]
        axes[2].axvline(n_gt, color="k", ls=":", lw=0.8)
        axes[2].text(n_gt, 0.05, f" {n_gt} true", fontsize=7)
    else:
        axes[2].axvline(1.0, color="k", ls=":", lw=0.8)
        axes[2].text(1.0, 0.05, " 1:1", fontsize=7)
    for ax in axes:
        ax.invert_xaxis() if ax in (axes[0], axes[1]) else None
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "feasibility.png", dpi=140, bbox_inches="tight")
    print(f"\nsaved  {out/'feasibility.png'}\nsaved  {out/'sweep.csv'}")


if __name__ == "__main__":
    main()
