"""
Ablation: rectangular-bbox band vs. exact-mask ("zero everything else") for the
stage-2 spectral-flatness gate on ridge detections.

The pipeline (vox_tracer/ridge.detection_band_features) computes flatness on the
STFT sliced to the detection's rectangular [f_low, f_high] band. The question is
whether restricting instead to the *segmentation polygon* -- zeroing every STFT
bin outside the detected mask -- gives a more accurate (more TP/FP separable,
higher event-F1) flatness. This isolates the choice by computing flatness FOUR
ways from ONE shared STFT per detection, changing only which bins enter it:

  rect          bins in [f_low, f_high], every frame        (current pipeline)
  mask_kept     per-frame: only bins under the mask at that time column,
                averaged over the kept bins only            (the fair per-frame band)
  mask_zero_bd  same mask, but out-of-mask bins ZEROED and INCLUDED in the
                per-frame average, denominator = band bins  (literal "zero it out")
  mask_zero_full same, denominator = the full 0..Nyquist spectrum

Comparing rect vs mask_kept answers "does the mask shape help?"; comparing
mask_kept vs mask_zero_bd answers "does zeroing-and-including-zeros hurt?" (it
does: flatness is a geometric/arithmetic-mean ratio, so inserted ~0 bins drag
the geometric mean down and collapse flatness toward 0 regardless of tonality).

Each variant is scored by (a) TP-vs-FP ROC-AUC and (b) event-level F1 under
grouped 4-fold CV, with the flatness threshold chosen on the training folds each
fold (no in-sample threshold peeking). Mirrors ridge_stage2_fromscratch.py.

    source venv/bin/activate
    python scripts/viz/ridge_stage2_masking_ablation.py --n-sessions 8
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))

import librosa  # noqa: E402
import ridge_pipeline_viz as rpv  # noqa: E402
from spectral_features import N_FFT, HOP, EPS, bbox_to_band, bbox_to_time  # noqa: E402
from vox_tracer.spec import load_channel_audio, parse_spec_fname  # noqa: E402
from vox_tracer.scoring import score_combined  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

VARIANTS = ["rect", "mask_kept", "mask_zero_bd", "mask_zero_full"]
LABELS = {
    "rect": "rect band\n(current)",
    "mask_kept": "mask, kept\nbins only",
    "mask_zero_bd": "mask, zeros in\navg (band denom)",
    "mask_zero_full": "mask, zeros in\navg (full denom)",
}


def stft_payload(audio, sr, comp, st, t0, t1, H, W, nyquist):
    """STFT of one detection's audio segment plus the rect-band / mask bin maps.

    ``comp`` is the H x W segmentation mask (0/1) on the spectrogram grid; ``st``
    holds the bbox. Returns a dict with the log-magnitude STFT (``logS``), the
    frequency axis (``freqs``), the rectangular band mask (``band``, over bins),
    the per-frame polygon mask (``keep``, bins x frames), and ``flat`` = the four
    flatness variants. Returns None for an empty segment.
    """
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(audio), int(round(t1 * sr)))
    if i1 <= i0:
        return None
    seg = audio[i0:i1].astype(np.float32)

    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP)) + EPS  # (n_bins, n_frames)
    logS = np.log(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    n_bins, n_frames = S.shape

    # bin -> spectrogram row (image is flipud: row 0 = Nyquist, row H-1 = 0 Hz)
    rows = np.clip(np.round((H - 1) * (1.0 - freqs / nyquist)).astype(int), 0, H - 1)

    # rectangular band bins (same for every frame) -- the current pipeline's gate
    f_low, f_high = bbox_to_band(st["by"], st["bh"], H, nyquist)
    band = (freqs >= f_low) & (freqs <= f_high)
    if not band.any():
        band[np.argmin(np.abs(freqs - 0.5 * (f_low + f_high)))] = True

    # frame -> mask column: the segment spans bbox cols [bx, bx+bw], so frame j
    # (centered at j*HOP samples) maps to bx + (j*HOP/len)*bw.
    bx, bw = st["bx"], max(st["bw"], 1)
    if n_frames > 1:
        frac = (np.arange(n_frames) * HOP) / max(len(seg), 1)
    else:
        frac = np.full(1, 0.5)
    cols = np.clip((bx + frac * bw).round().astype(int), 0, W - 1)

    # keep[bin, frame] = mask set at (row(bin), col(frame))
    keep = comp[rows[:, None], cols[None, :]] > 0

    def _flat_per_frame(mask_bf, denom_bins):
        """Mean over frames of gmean/amean, restricted per frame to mask_bf & denom_bins."""
        vals = []
        for j in range(n_frames):
            sel = mask_bf[:, j] & denom_bins
            if not sel.any():
                continue
            gm = np.exp(logS[sel, j].mean())
            am = S[sel, j].mean()
            vals.append(gm / am)
        return float(np.mean(vals)) if vals else np.nan

    def _flat_zeroed(denom_bins):
        """Zero out-of-mask bins (-> EPS) and INCLUDE them in the per-frame average."""
        Sz = np.where(keep, S, EPS)
        d = denom_bins
        gm = np.exp(np.log(Sz)[d].mean(axis=0))
        am = Sz[d].mean(axis=0)
        return float(np.mean(gm / am))

    band_bf = np.tile(band[:, None], (1, n_frames))
    flat = {
        "rect": _flat_per_frame(band_bf, np.ones(n_bins, bool)),  # band, all band bins
        "mask_kept": _flat_per_frame(keep, band),                 # mask ∩ band, kept only
        "mask_zero_bd": _flat_zeroed(band),                       # mask, zeros included, band denom
        "mask_zero_full": _flat_zeroed(np.ones(n_bins, bool)),    # mask, zeros included, full denom
    }
    return dict(logS=logS, freqs=freqs, band=band, keep=keep, dur_ms=(t1 - t0) * 1e3, flat=flat)


def build(sessions, channels=(118, 35), min_area=50, limit_pngs=None, reservoir_cap=400):
    """Rerun raw ridge; one row per stage-2 detection with the 4 flatness variants.

    Also returns a reservoir of STFT payloads (capped) for the before/after render.
    """
    rows, reservoir = [], []
    p = dict(rpv.DEFAULTS)
    p["min_area"] = min_area
    for session, spec_dir, data_dir, gt_csv, vox in sessions:
        for ch in channels:
            loaded = load_channel_audio(data_dir, ch)
            if loaded is None:
                continue
            sr, audio, _ = loaded
            nyq = sr / 2.0
            pngs = sorted(spec_dir.glob(f"headmic_{ch}_*.png"))
            if limit_pngs:
                pngs = pngs[:limit_pngs]
            for png in pngs:
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                H, W = gray.shape
                pc = dict(p)
                pc["freq_cutoff_row"] = int((H - 1) * (1.0 - p["freq_min"] / nyq))
                res = rpv.run_pipeline(gray, pc)
                _, wt0, wt1 = parse_spec_fname(png.name)
                for comp, fate, st in res["s2"]:
                    t0, t1 = bbox_to_time(st["bx"], st["bw"], wt0, wt1, W)
                    pay = stft_payload(audio, sr, comp, st, t0, t1, H, W, nyq)
                    if pay is None:
                        continue
                    label = rpv.label_tpfp(st["x0"], st["x1"], wt0, wt1, W, vox)
                    bmask = comp > 0
                    row = dict(
                        session=session, t0=t0, t1=t1, label=label,
                        ridge_score=float(res["response"][bmask].mean()) if bmask.any() else 0.0,
                    )
                    row.update(pay["flat"])
                    rows.append(row)
                    # reservoir: keep well-resolved detections for the render panel
                    if len(reservoir) < reservoir_cap and pay["keep"].shape[1] >= 5 \
                            and pay["band"].sum() >= 4:
                        pay["nyq"] = nyq; pay["label"] = label; pay["session"] = session
                        reservoir.append(pay)
        print(f"[{session}] cumulative detections: {len(rows)}")
    return pd.DataFrame(rows), reservoir


def event_f1(det, keep, sess_subset):
    """Event-level F1 over sessions for the kept detections (uses vox_tracer.scoring)."""
    kept = det[keep]
    tp = fn = pred_tp = fp = 0
    vox_by = {s: v for s, _, _, _, v in sess_subset}
    for session in vox_by:
        sub = kept[kept.session == session]
        boxes = list(zip(sub.t0, sub.t1, sub.ridge_score))
        r = score_combined(vox_by[session], boxes, 0.0)
        tp += r["tp"]; fn += r["fn"]; pred_tp += r["pred_tp"]; fp += r["fp"]
    prec = pred_tp / (pred_tp + fp) if (pred_tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return rec, prec, f1


def cv_best_threshold(det, col, sessions, folds, thr_grid):
    """Grouped CV: pick the flatness<=thr on train folds, evaluate on the held-out fold."""
    sess_by = {s[0]: s for s in sessions}
    thr_grid = list(thr_grid)
    R, P, F, chosen = [], [], [], []
    for te_names in folds:
        te_sub = [sess_by[n] for n in te_names]
        if not te_sub:  # skip empty folds (e.g. n_sessions < n_folds)
            continue
        tr_sub = [sess_by[s[0]] for s in sessions if s[0] not in te_names]
        best_thr, best_f1 = thr_grid[-1], -1.0  # default: most permissive gate
        for thr in thr_grid:
            keep = (det[col] <= thr).fillna(False)
            _, _, f1 = event_f1(det, keep, tr_sub)
            if np.isfinite(f1) and f1 > best_f1:
                best_f1, best_thr = f1, thr
        keep = (det[col] <= best_thr).fillna(False)
        r, p, f1 = event_f1(det, keep, te_sub)
        R.append(r); P.append(p); F.append(f1); chosen.append(best_thr)
    return dict(recall=np.nanmean(R), recall_std=np.nanstd(R),
                precision=np.nanmean(P), precision_std=np.nanstd(P),
                f1=np.nanmean(F), f1_std=np.nanstd(F),
                thr_med=float(np.median(chosen)))


def auc_tpfp(det, col):
    y = (det.label == "tp").astype(int).values
    v = det[col].values.astype(float)
    ok = np.isfinite(v)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return np.nan
    a = roc_auc_score(y[ok], v[ok])
    return max(a, 1 - a)  # oriented; flatness is TP-low so raw a < 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sessions", type=int, default=8)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=50, help="best stage-1 min_area")
    ap.add_argument("--limit-pngs", type=int, default=None, help="cap PNGs/channel (debug)")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs/eval/ridge_stage2_masking_ablation")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = rpv_discover(args.n_sessions)
    print(f"=== {len(sessions)} sessions (stage-1 min_area={args.min_area}) ===")
    det, reservoir = build(sessions, min_area=args.min_area, limit_pngs=args.limit_pngs)
    det.to_csv(args.out_dir / "detections_flatness.csv", index=False)
    n_tp = (det.label == "tp").sum(); n_fp = (det.label == "fp").sum()
    print(f"\n{len(det)} detections ({n_tp} TP / {n_fp} FP)\n")

    names = [s[0] for s in sessions]
    folds = [names[i::args.folds] for i in range(args.folds)]

    rows = []
    for v in VARIANTS:
        vals = det[v].dropna()
        thr_grid = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 40))) if len(vals) else [0.2]
        m = cv_best_threshold(det, v, sessions, folds, thr_grid)
        m["variant"] = v
        m["auc"] = auc_tpfp(det, v)
        m["tp_med"] = float(det.loc[det.label == "tp", v].median())
        m["fp_med"] = float(det.loc[det.label == "fp", v].median())
        rows.append(m)

    S = pd.DataFrame(rows)[["variant", "auc", "f1", "f1_std", "recall", "precision",
                            "thr_med", "tp_med", "fp_med"]]
    S = S.sort_values("f1", ascending=False).reset_index(drop=True)
    S.to_csv(args.out_dir / "masking_ablation.csv", index=False)

    print("=== flatness gate: rectangular band vs. exact mask (zeroing) ===")
    print(f"  {'variant':16s} {'AUC':>6} {'F1(cv)':>12} {'thr':>7} "
          f"{'TPmed':>7} {'FPmed':>7}")
    for r in S.itertuples():
        print(f"  {r.variant:16s} {r.auc:6.3f} {r.f1:6.3f}±{r.f1_std:.3f} "
              f"{r.thr_med:7.3f} {r.tp_med:7.3f} {r.fp_med:7.3f}")

    make_figure(det, S, args.out_dir / "masking_ablation.png")
    make_examples_figure(reservoir, args.out_dir / "masking_examples.png")
    print(f"\nwrote {args.out_dir}/detections_flatness.csv, masking_ablation.csv, "
          "masking_ablation.png, masking_examples.png")


def rpv_discover(n, min_vox=15):
    """Up to n sessions (one per experiment) with GT vox + audio. Same rule as
    ridge_filter_experiment.discover_sessions, but rooted at the repo (that
    module's ROOT is off by one now that it lives under scripts/viz/)."""
    picks, seen_exp = [], set()
    for exp in sorted((ROOT / "outputs/spectrograms").glob("experiment_*")):
        if exp.name in seen_exp:
            continue
        for idx in sorted(exp.glob("idx_*")):
            data_dir = ROOT / "data" / exp.name / idx.name
            gt = list(data_dir.glob("*annotations_gt.csv"))
            wav = list(data_dir.glob("headmic_*_*.wav"))
            if not (gt and wav):
                continue
            vox = rpv.load_gt_vox(gt[0])
            if len(vox) < min_vox:
                continue
            picks.append((f"{exp.name}/{idx.name}", idx, data_dir, gt[0], vox))
            seen_exp.add(exp.name)
            break
        if len(picks) >= n:
            break
    return picks


RECT_C = "#e08a1e"   # rectangular band
MASK_C = "#2a9d3a"   # exact segmentation mask


def region_color(variant):
    """Orange = rectangular-band region; green = exact-mask region."""
    return RECT_C if variant == "rect" else MASK_C


def make_figure(det, S, out_path):
    """Left: CV event-F1 per variant, colored by region (rect vs mask). Right: TP/FP
    flatness distributions for the rect band vs. the (kept-bins) mask."""
    order = list(S.variant)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    axL.bar(range(len(order)), S.f1, yerr=S.f1_std,
            color=[region_color(v) for v in order], capsize=3,
            error_kw=dict(lw=1, ecolor="#333"))
    for i, (f, s) in enumerate(zip(S.f1, S.f1_std)):
        axL.text(i, f + s + 0.008, f"{f:.2f}", ha="center", fontsize=9, fontweight="bold")
    axL.set_xticks(range(len(order)))
    axL.set_xticklabels([LABELS[v] for v in order], fontsize=8)
    axL.set_ylabel("event-level F1 (grouped 4-fold CV, mean ± std)")
    axL.set_ylim(0, 1.0)
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(color=RECT_C, label="rectangular band"),
                        Patch(color=MASK_C, label="segmentation mask")], loc="lower left")
    axL.set_title("Flatness gate F1 by region type", fontsize=10)
    axL.grid(axis="y", alpha=0.3); axL.spines[["top", "right"]].set_visible(False)

    # rect band vs. the exact mask (kept-bins version), colored by region
    for v, c in [("rect", RECT_C), ("mask_kept", MASK_C)]:
        for lab, ls in [("tp", "-"), ("fp", "--")]:
            x = det.loc[det.label == lab, v].dropna()
            if len(x):
                axR.hist(x, bins=40, range=(0, 1), density=True, histtype="step",
                         color=c, ls=ls, lw=1.6,
                         label=f"{'rect band' if v=='rect' else 'mask'} — {lab.upper()}")
    axR.set_xlabel("spectral flatness")
    axR.set_ylabel("density")
    axR.set_title("TP vs FP flatness: rect band (orange) vs mask (green)", fontsize=10)
    axR.legend(fontsize=7.5, ncol=2)
    axR.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _select_examples(reservoir):
    """Pick a few illustrative detections: tonal TP, swept TP (mask departs most from
    the box), a broadband-noise FP, and a tonal-looking FP (hard case). Silent /
    degenerate segments (no dynamic range) are excluded."""
    rend = [r for r in reservoir
            if np.isfinite(r["flat"]["rect"])
            and (np.percentile(r["logS"], 99) - np.percentile(r["logS"], 5)) > 1.5]
    if not rend:
        return []
    for r in rend:  # off-box fraction: band bins the mask drops, averaged over frames
        band_bf = np.tile(r["band"][:, None], (1, r["keep"].shape[1]))
        denom = band_bf.sum()
        r["_offbox"] = float((band_bf & ~r["keep"]).sum() / denom) if denom else 0.0
    tp = [r for r in rend if r["label"] == "tp"]
    fp = [r for r in rend if r["label"] == "fp"]
    picks = []
    if tp:  # most tonal TP (lowest rect flatness)
        picks.append(min(tp, key=lambda r: r["flat"]["rect"]))
    if tp:  # TP whose mask departs most from its bounding box
        swept = max(tp, key=lambda r: r["_offbox"])
        if all(swept is not q for q in picks):
            picks.append(swept)
    fp_noisy = [r for r in fp if r["flat"]["rect"] < 0.99]  # exclude flatness==1 degenerates
    if fp_noisy:  # most noise-like FP
        picks.append(max(fp_noisy, key=lambda r: r["flat"]["rect"]))
    if fp:  # a tonal-looking FP (the hard case the gate can miss)
        alt = min(fp, key=lambda r: r["flat"]["rect"])
        if all(alt is not q for q in picks):
            picks.append(alt)
    return picks[:4]


def make_examples_figure(reservoir, out_path):
    """Per example detection: full STFT -> rect-band-zeroed -> mask-zeroed, with the
    spectral-flatness score annotated on each and colored by region."""
    picks = _select_examples(reservoir)
    if not picks:
        print("  (no reservoir examples to render)")
        return
    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.9 * n), constrained_layout=True,
                             squeeze=False)
    for i, r in enumerate(picks):
        logS, freqs, band, keep = r["logS"], r["freqs"], r["band"], r["keep"]
        floor = logS.min()
        khz = freqs / 1e3
        extent = [0, r["dur_ms"], khz[0], khz[-1]]
        vmin, vmax = np.percentile(logS, [5, 99])

        rect_img = np.where(np.tile(band[:, None], (1, logS.shape[1])), logS, floor)
        mask_img = np.where(keep, logS, floor)
        panels = [
            (logS, "STFT segment", "#333", None),
            (rect_img, f"rect band  ·  flatness={r['flat']['rect']:.3f}", RECT_C, "rect"),
            (mask_img, f"mask (bg zeroed)  ·  flatness={r['flat']['mask_zero_bd']:.3f}",
             MASK_C, "mask"),
        ]
        for j, (img, title, tc, kind) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(img, origin="lower", aspect="auto", extent=extent,
                      cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9, color=tc,
                         fontweight="bold" if kind else "normal")
            if j == 0:
                ax.set_ylabel(f"{r['label'].upper()}  ·  {r['session']}\nfreq (kHz)",
                              fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel("time (ms)", fontsize=8)
            ax.tick_params(labelsize=7)
            # draw the kept-region outline in its colour
            if kind == "rect":
                for f in (freqs[band].min() / 1e3, freqs[band].max() / 1e3):
                    ax.axhline(f, color=tc, lw=1.1, ls="--")
            elif kind == "mask":
                ax.contour(keep.astype(float), levels=[0.5], colors=[tc],
                           linewidths=1.1, extent=extent, origin="lower")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
