"""
Visualize the ridge (sato) segmentation pipeline stage by stage, so every
threshold in vox_tracer/ridge.py can be judged against the data it acts on.

Two figures are produced.

1. ridge_pipeline_<chunk>.png  — one row of spatial panels per example chunk:
       raw spectrogram (+GT band) | sato response | percentile threshold |
       component filter (color = why each blob was kept/killed) |
       morph-close + refilter (final mask) | stage-2 per-detection filter
   This shows *what* each threshold does to a real 1-second window.

2. ridge_criteria_hist.png — one panel per scalar threshold, each a histogram of
   the quantity the threshold compares against (sato response value, component
   area, aspect ratio, top-row frequency, stage-2 mask-area fraction, column
   span, frequency-sweep fraction) with the current cut drawn as a line. This
   shows *whether* each threshold sits in a sensible place in the distribution.

Stages and thresholds mirror vox_tracer.ridge.compute_seg_mask /
passes_mask_filters exactly (imported where possible).

Usage
-----
    python scripts/viz/ridge_pipeline_viz.py \
        --spec-dir outputs/spectrograms/experiment_384/idx_000 \
        --gt-csv   data/experiment_384/idx_000/exp_384_idx_001_ch_all_annotations_gt.csv \
        --channel 118 --chunks 16 17 18 \
        --out-dir  outputs/eval/ridge_pipeline
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
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from vox_tracer.ridge import _build_filter_fn, passes_mask_filters  # noqa: E402
from vox_tracer.spec import load_channel_audio, parse_spec_fname  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from spectral_features import bbox_to_band, bbox_to_time, event_features  # noqa: E402

# Defaults copied from vox_tracer.ridge / scripts.run so the picture matches the run.
DEFAULTS = dict(
    filter_name="sato", sigmas=(2, 3, 4), threshold_pct=99.0,
    sample_rate=125000, freq_min=20000.0,
    min_area=30, vert_aspect=5.0, horiz_aspect=0.2, close_kernel=(7, 3),
    max_mask_area_frac=0.15, min_freq_sweep_frac=0.04, min_mask_cols=5,
)

# Stage-1 component fates (priority order matches _filter_components).
FATE_COLORS = {
    "keep":     "#0ca30c",   # survives every stage-1 test
    "area":     "#8a8984",   # too small
    "vertical": "#d03b3b",   # aspect too tall/narrow (vertical noise burst)
    "horizontal": "#e08a1e",  # aspect too wide/flat
    "lowfreq":  "#2a78d6",   # top row below freq_min cutoff
}
# Stage-2 per-detection fates.
S2_COLORS = {
    "keep":      "#0ca30c",
    "too_big":   "#d03b3b",
    "too_short": "#e08a1e",
    "flat":      "#8e5fd3",
}


def component_fate(area, comp_h, comp_w, y_top, p):
    """Reproduce _filter_components' accept/reject decision, returning the reason."""
    aspect = comp_h / max(comp_w, 1)
    if area < p["min_area"]:
        return "area"
    if aspect > p["vert_aspect"]:
        return "vertical"
    if aspect < p["horiz_aspect"]:
        return "horizontal"
    if y_top > p["freq_cutoff_row"]:
        return "lowfreq"
    return "keep"


def stage2_fate(mask_u8, H, W, p):
    """Reproduce passes_mask_filters' decision, returning the reason it was cut."""
    if int((mask_u8 > 0).sum()) > p["max_mask_area_frac"] * H * W:
        return "too_big"
    ys, xs = np.where(mask_u8 > 0)
    unique_xs = np.unique(xs)
    if len(unique_xs) < p["min_mask_cols"]:
        return "too_short"
    col_meds = np.array([np.median(ys[xs == x]) for x in unique_xs])
    if (col_meds.max() - col_meds.min()) / H < p["min_freq_sweep_frac"]:
        return "flat"
    return "keep"


def run_pipeline(gray, p, stage1=True, response=None):
    """Run every stage; return a dict of intermediate arrays + per-component stats.

    Pass a precomputed ``response`` (the sato filter output) to skip recomputing
    it when sweeping many stage-1 configs over the same chunk.

    Mirrors compute_seg_mask: threshold -> stage-1 component filter -> morph-close
    -> stage-1 again -> (caller applies stage-2 per contour).

    stage1=False ablates the stage-1 component filter entirely: the thresholded
    binary goes straight into the morph-close and every resulting contour becomes
    a detection (speckle and all), so stage-2 alone must do the cleaning.
    """
    H, W = gray.shape
    if response is None:
        fn = _build_filter_fn(p["filter_name"], list(p["sigmas"]))
        response = fn(gray.astype(np.float64) / 255.0)

    thresh = np.percentile(response, p["threshold_pct"])
    binary = (response >= thresh).astype(np.uint8) * 255

    # stage-1 component labelling with fate per blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    fate_img = np.zeros((H, W), np.uint8)   # 0 = background
    kept = np.zeros((H, W), np.uint8)
    comp_rows = []
    fate_idx = {k: i + 1 for i, k in enumerate(FATE_COLORS)}
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = stats[lbl, cv2.CC_STAT_WIDTH]
        y_top = stats[lbl, cv2.CC_STAT_TOP]
        x0 = stats[lbl, cv2.CC_STAT_LEFT]
        fate = component_fate(area, comp_h, comp_w, y_top, p) if stage1 else "keep"
        fate_img[labels == lbl] = fate_idx[fate]
        comp_rows.append(dict(area=area, aspect=comp_h / max(comp_w, 1),
                              y_top=y_top, fate=fate, x0=int(x0), x1=int(x0 + comp_w)))
        if fate == "keep":
            kept[labels == lbl] = 255

    # morph-close then stage-1 filter again == compute_seg_mask's final mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(p["close_kernel"]))
    closed = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(closed)
    final = np.zeros((H, W), np.uint8)
    for lbl in range(1, n2):
        area = stats2[lbl, cv2.CC_STAT_AREA]
        comp_h = stats2[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = stats2[lbl, cv2.CC_STAT_WIDTH]
        y_top = stats2[lbl, cv2.CC_STAT_TOP]
        if not stage1 or component_fate(area, comp_h, comp_w, y_top, p) == "keep":
            final[labels2 == lbl] = 255

    # stage-2 per detection (per contour of the final mask)
    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s2 = []   # (comp_mask, fate, stats dict)
    for cnt in contours:
        if len(cnt) < 3:
            continue
        comp = np.zeros((H, W), np.uint8)
        cv2.drawContours(comp, [cnt], -1, 1, thickness=cv2.FILLED)
        fate = stage2_fate(comp, H, W, p)
        ys, xs = np.where(comp > 0)
        ux = np.unique(xs)
        col_meds = np.array([np.median(ys[xs == x]) for x in ux]) if len(ux) else np.array([0])
        bx, by, bw, bh = cv2.boundingRect(cnt)
        s2.append((comp, fate, dict(
            area_frac=comp.sum() / (H * W),
            n_cols=len(ux),
            sweep_frac=(col_meds.max() - col_meds.min()) / H,
            x0=int(ux.min()) if len(ux) else 0,
            x1=int(ux.max()) + 1 if len(ux) else 0,
            bx=int(bx), by=int(by), bw=int(bw), bh=int(bh),
        )))

    return dict(response=response, thresh=thresh, binary=binary, fate_img=fate_img,
                final=final, comp_rows=comp_rows, s2=s2, H=H, W=W)


def load_gt_vox(gt_csv):
    """[(start, stop)] of GT vox events for the recording, or [] if no csv."""
    if gt_csv is None or not Path(gt_csv).exists():
        return []
    df = pd.read_csv(gt_csv)
    return [(float(r.start_seconds), float(r.stop_seconds))
            for r in df[df.name == "vox"].itertuples()
            if r.stop_seconds > r.start_seconds]


def label_tpfp(x0, x1, wt0, wt1, W, gt):
    """'tp' if the component's time span overlaps any GT vox event, else 'fp'."""
    dur = wt1 - wt0
    ta = wt0 + (x0 / W) * dur
    tb = wt0 + (x1 / W) * dur
    return "tp" if any(min(tb, e) > max(ta, s) for s, e in gt) else "fp"


def gt_bands_for_chunk(gt_csv, t0, t1):
    """[(x_frac0, x_frac1)] of GT vox events overlapping window [t0, t1]."""
    if gt_csv is None or not Path(gt_csv).exists():
        return []
    df = pd.read_csv(gt_csv)
    dur = t1 - t0
    bands = []
    for r in df[df.name == "vox"].itertuples():
        s, e = float(r.start_seconds), float(r.stop_seconds)
        if e <= t0 or s >= t1:
            continue
        bands.append((max(0.0, (s - t0) / dur), min(1.0, (e - t0) / dur)))
    return bands


def draw_gt(ax, bands, W, H):
    for a, b in bands:
        ax.axvspan(a * W, b * W, color="#00a000", alpha=0.12, lw=0)
        for x in (a * W, b * W):
            ax.axvline(x, color="#00a000", ls="--", lw=0.8, alpha=0.7)


def fate_cmap(color_map):
    """ListedColormap with index 0 transparent, then one color per fate key."""
    cols = [(0, 0, 0, 0)] + [matplotlib.colors.to_rgba(c) for c in color_map.values()]
    return ListedColormap(cols)


def pipeline_figure(gray, res, bands, title, out_path):
    H, W = res["H"], res["W"]
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.4), constrained_layout=True)

    # 1 raw + GT
    axes[0].imshow(gray, cmap="magma", aspect="auto")
    draw_gt(axes[0], bands, W, H)
    axes[0].set_title("1. spectrogram (+GT band)", fontsize=10)

    # 2 sato response
    im = axes[1].imshow(res["response"], cmap="viridis", aspect="auto")
    axes[1].set_title(f"2. sato response\n(p99 = {res['thresh']:.3f})", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

    # 3 threshold binary
    axes[2].imshow(res["binary"], cmap="gray", aspect="auto")
    axes[2].set_title("3. >= p99 threshold", fontsize=10)

    # 4 component fates
    axes[3].imshow(gray, cmap="gray", aspect="auto")
    axes[3].imshow(res["fate_img"], cmap=fate_cmap(FATE_COLORS), aspect="auto",
                   vmin=0, vmax=len(FATE_COLORS), alpha=0.85, interpolation="nearest")
    axes[3].set_title("4. stage-1 component filter", fontsize=10)

    # 5 final mask overlay
    axes[4].imshow(gray, cmap="gray", aspect="auto")
    overlay = np.zeros((H, W, 4))
    overlay[res["final"] > 0] = matplotlib.colors.to_rgba("#0ca30c", 0.55)
    axes[4].imshow(overlay, aspect="auto")
    draw_gt(axes[4], bands, W, H)
    axes[4].set_title("5. morph-close -> final mask", fontsize=10)

    # 6 stage-2 per detection
    axes[5].imshow(gray, cmap="gray", aspect="auto")
    s2_overlay = np.zeros((H, W, 4))
    for comp, fate, _ in res["s2"]:
        s2_overlay[comp > 0] = matplotlib.colors.to_rgba(S2_COLORS[fate], 0.6)
    axes[5].imshow(s2_overlay, aspect="auto")
    draw_gt(axes[5], bands, W, H)
    n_keep = sum(1 for _, f, _ in res["s2"] if f == "keep")
    axes[5].set_title(f"6. stage-2 filter ({n_keep}/{len(res['s2'])} kept)", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    # legends
    s1_handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=c,
                             label={"keep": "kept", "area": "too small",
                                    "vertical": "aspect>vert (vertical burst)",
                                    "horizontal": "aspect<horiz (flat)",
                                    "lowfreq": "below freq_min"}[k])
                  for k, c in FATE_COLORS.items()]
    axes[3].legend(handles=s1_handles, fontsize=6.5, loc="lower center",
                   bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)
    s2_handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=c,
                             label={"keep": "kept", "too_big": "area>15%",
                                    "too_short": "<5 cols", "flat": "sweep<4%"}[k])
                  for k, c in S2_COLORS.items()]
    axes[5].legend(handles=s2_handles, fontsize=6.5, loc="lower center",
                   bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


TP_COLOR, FP_COLOR, CUT_COLOR = "#0ca30c", "#d03b3b", "#111111"


def criteria_histograms(all_comp, all_s2, p, out_path):
    """Per-threshold histograms colored by TP/FP, one red-dashed cut line each.

    Each row is one class (TP = overlaps a GT vox call, FP = does not), plotted
    as a density (area-normalised) so the two shapes are comparable despite very
    different counts. The single dashed line is the current threshold; the panel
    subtitle names which side is rejected.
    """
    comp = pd.DataFrame(all_comp)
    s2 = pd.DataFrame(all_s2)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    axes = axes.ravel()

    def panel(ax, df, col, cut, title, reject, xlabel, log=False):
        vals_all = np.asarray(df[col], float)
        finite = np.isfinite(vals_all)
        if log:
            finite &= vals_all > 0
        lo, hi = np.nanpercentile(vals_all[finite], [0.5, 99.5])
        if log:
            bins = np.logspace(np.log10(max(lo, 1e-9)), np.log10(hi), 50)
            ax.set_xscale("log")
        else:
            bins = np.linspace(lo, hi, 50)
        for lab, color in (("tp", TP_COLOR), ("fp", FP_COLOR)):
            v = vals_all[finite & (df["label"].values == lab)]
            if len(v) < 2:
                continue
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.45,
                    label=f"{lab.upper()} (n={len(v):,})")
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=1.6)
        ax.axvline(cut, color=CUT_COLOR, lw=2.2, ls="--", label=f"cut = {cut:g}")
        ax.set_title(f"{title}\nreject {reject}", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_yticks([])
        ax.legend(fontsize=7.5)
        ax.spines[["top", "right", "left"]].set_visible(False)

    panel(axes[0], comp, "area", p["min_area"], "Stage-1: component area",
          f"area < {p['min_area']}", "pixels", log=True)
    panel(axes[1], comp, "aspect", p["vert_aspect"], "Stage-1: aspect — vertical cut",
          f"aspect > {p['vert_aspect']:g} (vertical burst)", "height / width", log=True)
    panel(axes[2], comp, "aspect", p["horiz_aspect"], "Stage-1: aspect — horizontal cut",
          f"aspect < {p['horiz_aspect']:g} (flat line)", "height / width", log=True)
    panel(axes[3], comp, "y_top", p["freq_cutoff_row"], "Stage-1: component top row",
          f"row > {p['freq_cutoff_row']} (< {p['freq_min']/1e3:.0f} kHz)", "top row (0 = Nyquist)")

    if len(s2):
        panel(axes[4], s2, "area_frac", p["max_mask_area_frac"],
              "Stage-2: mask area fraction", f"frac > {p['max_mask_area_frac']:g}",
              "fraction of image")
        panel(axes[5], s2, "n_cols", p["min_mask_cols"], "Stage-2: column span",
              f"cols < {p['min_mask_cols']}", "# time columns")
        panel(axes[6], s2, "sweep_frac", p["min_freq_sweep_frac"],
              "Stage-2: frequency-sweep fraction", f"sweep < {p['min_freq_sweep_frac']:g}",
              "median-freq sweep / H")
    axes[7].axis("off")

    fig.suptitle("Ridge thresholds vs. the data — colored by TP (overlaps a GT call) "
                 f"vs FP  ·  full session  ·  {len(comp):,} stage-1 components, "
                 f"{len(s2):,} stage-2 detections",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# (column, title, unit, scale) for the raw-detection spectral-feature grid.
SPEC_PANELS = [
    ("centroid_mean",  "Spectral centroid",  "kHz", 1e-3),
    ("bandwidth_mean", "Spectral bandwidth", "kHz", 1e-3),
    ("rolloff_mean",   "Spectral rolloff",   "kHz", 1e-3),
    ("flatness_mean",  "Spectral flatness",  "",    1.0),
    ("rms_mean",       "In-band RMS",        "",    1.0),
    ("zcr_mean",       "Zero-crossing rate", "",    1.0),
    ("peak_freq_mean", "Peak frequency",     "kHz", 1e-3),
    ("f_band_hz",      "Detection band",     "kHz", 1e-3),
    ("duration_sec",   "Event duration",     "ms",  1e3),
]


def spectral_tpfp_figure(spec_rows, out_path):
    """KDE (area-normalised) spectral-feature distributions of RAW ridge
    detections, one curve per class (TP overlaps a GT call, FP does not)."""
    from scipy.stats import gaussian_kde
    if not spec_rows:
        print("no spectral rows (missing audio?) — skipping spectral figure")
        return
    df = pd.DataFrame(spec_rows)
    n_tp = int((df["label"] == "tp").sum())
    n_fp = int((df["label"] == "fp").sum())

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    axes = axes.ravel()

    def kde(vals, grid):
        v = vals[np.isfinite(vals)]
        if len(v) < 5 or np.ptp(v) == 0:
            return None
        return gaussian_kde(v)(grid)

    for ax, (col, title, unit, scale) in zip(axes, SPEC_PANELS):
        pooled = df[col].dropna().values * scale
        if len(pooled) == 0:
            ax.axis("off"); continue
        lo, hi = np.percentile(pooled, [1, 99])
        grid = np.linspace(lo, hi, 400)
        for lab, color in (("tp", TP_COLOR), ("fp", FP_COLOR)):
            vals = df.loc[df["label"] == lab, col].dropna().values * scale
            dens = kde(vals, grid)
            if dens is not None:
                ax.plot(grid, dens, color=color, lw=1.9, solid_capstyle="round")
                ax.fill_between(grid, dens, color=color, alpha=0.12)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        if unit:
            ax.set_xlabel(unit, fontsize=9, color="#8a8984")
        ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [plt.Line2D([], [], color=TP_COLOR, lw=2.6, label=f"True positive (n={n_tp:,})"),
               plt.Line2D([], [], color=FP_COLOR, lw=2.6, label=f"False positive (n={n_fp:,})")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=11)
    fig.suptitle("Spectral features of RAW ridge detections (pre-filter), "
                 "TP vs FP  ·  full session  ·  density-normalised", fontsize=13,
                 fontweight="bold", y=1.03)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}  (TP {n_tp:,}, FP {n_fp:,})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--gt-csv", default=None)
    ap.add_argument("--channel", type=int, default=118,
                    help="channel used for the per-chunk pipeline panels")
    ap.add_argument("--channels", default="118,35",
                    help="channels aggregated for the TP/FP criteria histograms (full session)")
    ap.add_argument("--recording-dir", default=None,
                    help="dir with the headmic wavs (default: data/<exp>/<idx> from spec-dir)")
    ap.add_argument("--chunks", type=int, nargs="*", default=[16, 17, 18],
                    help="chunk indices to render as full pipeline panels")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval/ridge_pipeline"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p = dict(DEFAULTS)
    # freq_cutoff_row depends on H; set per image below, but stash freq params.
    spec_dir = Path(args.spec_dir)
    pngs = sorted(spec_dir.glob(f"headmic_{args.channel}_*.png"))
    by_chunk = {}
    for png in pngs:
        stem = png.stem
        ci = int(stem.split("chunk_")[1].split("_")[0])
        by_chunk[ci] = png

    # 1. per-chunk pipeline panels
    for ci in args.chunks:
        png = by_chunk.get(ci)
        if png is None:
            print(f"chunk {ci}: no PNG, skipping")
            continue
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        H = gray.shape[0]
        pc = dict(p)
        pc["freq_cutoff_row"] = int((H - 1) * (1.0 - p["freq_min"] / (p["sample_rate"] / 2.0)))
        res = run_pipeline(gray, pc)
        _, t0, t1 = parse_spec_fname(png.name)
        bands = gt_bands_for_chunk(args.gt_csv, t0, t1)
        pipeline_figure(gray, res, bands,
                        f"ridge pipeline  ch{args.channel}  chunk {ci}  t={t0:.0f}-{t1:.0f}s",
                        args.out_dir / f"ridge_pipeline_ch{args.channel}_chunk{ci:05d}.png")

    # 2. aggregate TP/FP criteria histograms + RAW-detection spectral features
    #    over the FULL session (all channels).  "Raw" = every contour of the final
    #    mask, i.e. before passes_mask_filters — not the on-disk filtered output.
    gt = load_gt_vox(args.gt_csv)
    print(f"GT vox events: {len(gt)}")
    rec_dir = Path(args.recording_dir) if args.recording_dir else \
        ROOT / "data" / spec_dir.parent.name / spec_dir.name
    hist_channels = [int(c) for c in args.channels.split(",")]
    all_comp, all_s2, spec_rows, pc = [], [], [], p
    for ch in hist_channels:
        ch_pngs = sorted(spec_dir.glob(f"headmic_{ch}_*.png"))
        loaded = load_channel_audio(rec_dir, ch)          # for spectral features
        sr = audio = None
        if loaded is not None:
            sr, audio, _ = loaded
        print(f"  ch {ch}: {len(ch_pngs)} chunks, audio={'yes' if audio is not None else 'MISSING'}")
        for png in ch_pngs:
            gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            H, W = gray.shape
            pc = dict(p)
            pc["freq_cutoff_row"] = int((H - 1) * (1.0 - p["freq_min"] / (p["sample_rate"] / 2.0)))
            res = run_pipeline(gray, pc)
            _, wt0, wt1 = parse_spec_fname(png.name)
            for row in res["comp_rows"]:
                row["label"] = label_tpfp(row["x0"], row["x1"], wt0, wt1, W, gt)
                all_comp.append(row)
            for _, f, st in res["s2"]:
                st = dict(fate=f, **st)
                st["label"] = label_tpfp(st["x0"], st["x1"], wt0, wt1, W, gt)
                all_s2.append(st)
                if audio is None:
                    continue
                # raw-detection spectral features (band-restricted, same mapping
                # as scripts/spectral_features.py) — every contour, filtered or not.
                t0, t1 = bbox_to_time(st["bx"], st["bw"], wt0, wt1, W)
                f_low, f_high = bbox_to_band(st["by"], st["bh"], H, sr / 2.0)
                feats = event_features(audio, sr, t0, t1, f_low, f_high)
                if feats is None:
                    continue
                srow = {"label": st["label"], "duration_sec": t1 - t0,
                        "f_band_hz": f_high - f_low, "fate": f}
                for name, (m, _) in feats.items():
                    srow[f"{name}_mean"] = m
                spec_rows.append(srow)
    criteria_histograms(all_comp, all_s2, pc, args.out_dir / "ridge_criteria_hist.png")
    spectral_tpfp_figure(spec_rows, args.out_dir / "ridge_spectral_tpfp.png")


if __name__ == "__main__":
    main()
