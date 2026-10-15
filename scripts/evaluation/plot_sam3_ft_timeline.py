"""Contiguous stretches of test recordings: spectrogram + each model's masks, with event-level outcomes.

Rows: GT, then one per model. Each row is the --channel spectrogram over [t0, t0 + --seconds] with the
model's masks on that mic; the strip above it has two lanes, scored exactly as compare_sam3_ft.py
(both mics pooled, any overlap):
  top     predicted events: filled = matches a GT call, hollow = false positive
  bottom  GT calls: filled = detected by this model, hollow = missed
Only the call band (--rows) is shown, contrast-stretched. Segments: --n hardest (most errors summed over
models, of --focus only, or GT calls missed by every model with --missed-by-all; >= 3 GT calls) and --n random, one PNG each.

GT is either the gerbil_ssl corrected GT (--split-json, --annotator) or, with --gt-dir, one
*annotations_gt.csv per recording covering the whole recording (--recordings-file); --extra-gt adds a row
with another GT's intervals. Spectrograms come from PNGs or HDF5 (--spec-root, --prefix):

    python scripts/evaluation/plot_sam3_ft_timeline.py --gt-dir outputs/dryad_full_eval/gt \
        --recordings-file outputs/dryad_full_eval/recordings.txt --extra-gt segments=data/dryad_gerbil_full \
        --spec-root outputs/spectrograms_h5/dryad_gerbil_full --prefix mic --channel 0 --channels 0 --rows 0,257 \
        --det sam3-text-boxft=outputs/sam3_text_boxft_dryad/dryad_gerbil_full --focus sam3-text-boxft --out ...

    python scripts/evaluation/plot_sam3_ft_timeline.py --split-json outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json \
        --det zero-shot=outputs/sam3_zeroshot_trace/gerbil_ssl@0.2 box2mask=outputs/sam3_ft_smoke_box2mask/gerbil_ssl \
              seg=outputs/sam3_ft_smoke_head/gerbil_ssl sam3_best=outputs/sam3_best/gerbil_ssl@0 \
        --out outputs/eval/sam3_ft_smoke_test/timeline
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.compare_sam3_ft import COLORS, GT_ROOT, load_gt, rel_of, spec  # noqa: E402
from vox_tracer.scoring import load_gt_from_csv, read_pred_boxes, score_combined  # noqa: E402
from vox_tracer.spec import WindowReader, spec_windows_by_channel  # noqa: E402

GT_GRAY = "#555555"


def outcomes(base, thr, events, spans, channels, rel):
    """(pred events [(t0, t1, is_tp)], GT [(t0, t1, detected)]) for one recording."""
    boxes = [b for b in read_pred_boxes(Path(base) / rel, channels)
             if (b[2] is None or b[2] >= thr) and any(s <= (b[0] + b[1]) / 2 <= e for s, e in spans)]
    r = score_combined(events, boxes)
    tp = set(r["tp_events"])
    return ([(s, e, (s, e) in tp) for s, e in r["pred_events"]],
            [(s, e, i in r["matched_gt"]) for i, (s, e) in enumerate(events)])


def in_seg(iv, t0, t1):
    return [x for x in iv if x[1] > t0 and x[0] < t1]


def overlap_counts(iv, t0s, seconds):
    """For each window [t0, t0 + seconds), how many intervals of iv overlap it."""
    st, en = np.sort([x[0] for x in iv]), np.sort([x[1] for x in iv])
    return np.searchsorted(st, t0s + seconds, "left") - np.searchsorted(en, t0s, "right")


def strip_canvas(spec_root, prefix, rel, ch, t0, seconds):
    """(concatenated 1 s spectrogram windows (gray), column offset of each window) for [t0, t0 + seconds),
    or None if any window is missing."""
    wins = {round(w0): (src, w0, w1) for src, w0, w1 in
            spec_windows_by_channel(Path(spec_root) / rel, [ch], prefix=prefix).get(ch, [])}
    if any(t0 + k not in wins for k in range(seconds)):
        return None
    read = WindowReader()
    tiles = [read(*wins[t0 + k]) for k in range(seconds)]
    read.close()
    return np.hstack(tiles), np.cumsum([0] + [t.shape[1] for t in tiles])


def mask_canvas(base, rel, ch, thr, t0, seconds, h, offs):
    """Union of the model's masks on mic `ch` over the segment, same layout as strip_canvas."""
    p = Path(base) / rel / f"coco_ch_{ch}.json"
    out = np.zeros((h, offs[-1]), bool)
    if not p.exists():
        return out
    c = json.load(open(p))
    ims = {i["id"]: i for i in c["images"]}
    for a in c["annotations"]:
        im = ims[a["image_id"]]
        k = int(round(im["window_start_sec"] - t0))
        polys = [q for q in a.get("segmentation", []) if len(q) >= 6]
        if not 0 <= k < seconds or a.get("score", 1.0) < thr or not polys:
            continue
        m = mask_util.decode(mask_util.merge(mask_util.frPyObjects(polys, im["height"], im["width"]))).astype(bool)
        n = min(m.shape[1], offs[k + 1] - offs[k])
        out[:, offs[k]:offs[k] + n] |= m[:h, :n]
    return out


def draw_lanes(ax, preds, gts, color, t0, t1):
    for s, e, ok in preds:
        ax.add_patch(plt.Rectangle((s, 0.55), e - s, 0.4, fc=color if ok else "none", ec=color, lw=1))
    for s, e, ok in gts:
        ax.add_patch(plt.Rectangle((s, 0.05), e - s, 0.4, fc=GT_GRAY if ok else "none", ec=GT_GRAY, lw=1,
                                   ls="-" if ok else "--"))
    ax.set(xlim=(t0, t1), ylim=(0, 1), xticks=[], yticks=[])
    ax.spines[:].set_visible(False)


def plot_segment(strip, rel, t0, seconds, ch, models, res, gt_events, extra, rows_px, out):
    img, offs = strip
    img = img.astype(float)
    h = img.shape[0]
    y0, y1 = rows_px
    img = np.clip(img / max(np.percentile(img[y0:y1], 99.97), 1), 0, 1)
    t1 = t0 + seconds
    rows = [("GT", None)] + [(nm, ("extra", ev)) for nm, ev in extra] + models
    fig = plt.figure(figsize=(3.2 * seconds, 1.55 * len(rows)), facecolor="#ffffff")
    gs = fig.add_gridspec(2 * len(rows), 1, height_ratios=[0.22, 1] * len(rows), hspace=0.08)
    ext = (t0, t1, y1, y0)
    for i, (nm, m) in enumerate(rows):
        lane = fig.add_subplot(gs[2 * i])
        ax = fig.add_subplot(gs[2 * i + 1])
        rgb = np.repeat(img[..., None], 3, axis=2)
        if m is None or m[0] == "extra":
            ev = gt_events if m is None else m[1]
            draw_lanes(lane, [], [(s, e, True) for s, e in in_seg(ev, t0, t1)], GT_GRAY, t0, t1)
        else:
            base, thr, color = m
            mk = mask_canvas(base, rel, ch, thr, t0, seconds, h, offs)
            rgb[mk] = 0.2 * rgb[mk] + 0.8 * np.array(matplotlib.colors.to_rgb(color))
            preds, gts = res[nm]
            draw_lanes(lane, in_seg(preds, t0, t1), in_seg(gts, t0, t1), color, t0, t1)
        ax.imshow(rgb[y0:y1], aspect="auto", extent=ext, interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(nm, fontsize=9)
        if i < len(rows) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("time (s)")
    fig.savefig(out, dpi=130, facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-json", help="gerbil_ssl: dataset split whose recordings are shown")
    ap.add_argument("--gt-dir", help="per-recording <gt-dir>/<rel>/*annotations_gt.csv (whole recording)")
    ap.add_argument("--recordings-file", help="with --gt-dir: recordings shown, one per line")
    ap.add_argument("--extra-gt", nargs="*", default=[], help="name=<dir>: extra GT row, <dir>/<rel>/*annotations_gt.csv")
    ap.add_argument("--spec-root", default="outputs/spectrograms/gerbil_ssl")
    ap.add_argument("--prefix", default="headmic")
    ap.add_argument("--focus", help="rank hard segments by this model's errors only")
    ap.add_argument("--missed-by-all", action="store_true",
                    help="rank hard segments by GT calls that no model detects")
    ap.add_argument("--det", nargs="+", required=True, help="name=<recording pred base>[@score] ...")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--annotator", default="niegil")
    ap.add_argument("--channel", type=int, default=118, help="mic shown (the annotator's mic)")
    ap.add_argument("--channels", default="118,35", help="mics pooled for scoring")
    ap.add_argument("--gt-root", default=GT_ROOT)
    ap.add_argument("--coverage", default="outputs/pseudo_boxes/gerbil_ssl_corrected_gt_coverage.csv")
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--n", type=int, default=4, help="segments per kind (hardest, random)")
    ap.add_argument("--rows", default="85,180", help="spectrogram pixel rows shown (the call band)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.gt_dir:
        recs = [x.strip() for x in open(a.recordings_file) if x.strip()]
        gt = {r: (load_gt_from_csv(Path(a.gt_dir) / r)[0], [(0.0, float("inf"))]) for r in recs}
    else:
        recs = {rel_of(im["clip"]) for im in json.load(open(a.split_json))["images"]}
        gt = load_gt(a.gt_root, a.annotator, recs, pd.read_csv(a.coverage))
    extra = {nm: {r: load_gt_from_csv(Path(d) / r)[0] for r in gt} for nm, d in (x.split("=", 1) for x in a.extra_gt)}
    channels = [int(c) for c in a.channels.split(",")]
    dets = [spec(s, a.score) for s in a.det]
    models = [(nm, (base, thr, COLORS[k % len(COLORS)])) for k, (nm, base, thr) in enumerate(dets)]
    res = {rel: {nm: outcomes(base, thr, ev, spans, channels, rel) for nm, base, thr in dets}
           for rel, (ev, spans) in gt.items()}

    cands = []  # (errors, rel, t0)
    for rel, (ev, spans) in gt.items():
        for s, e in spans:
            e = min(e, max((x[1] for x in ev), default=0) + a.seconds)
            t0s = np.arange(int(np.ceil(s)), int(np.floor(e)) - a.seconds + 1)
            if a.missed_by_all:
                errs = [x for x, *rest in zip(*(g for _, g in res[rel].values()))
                        if not x[2] and not any(y[2] for y in rest)]
            else:
                errs = [x for nm, (p, g) in res[rel].items() if a.focus in (None, nm) for x in p + g if not x[2]]
            n_gt, n_err = overlap_counts(ev, t0s, a.seconds), overlap_counts(errs, t0s, a.seconds)
            cands += [(int(n), rel, int(t)) for t, n, g in zip(t0s, n_err, n_gt) if g >= 3]
    rng = np.random.default_rng(a.seed)
    hard, used = [], set()
    for err, rel, t0 in sorted(cands, key=lambda c: -c[0]):
        if rel not in used and len(hard) < a.n:
            hard.append((rel, t0))
            used.add(rel)
    rand = [cands[i][1:] for i in rng.choice(len(cands), min(a.n, len(cands)), replace=False)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind, segs in (("hard", hard), ("random", rand)):
        for k, (rel, t0) in enumerate(segs):
            img = strip_canvas(a.spec_root, a.prefix, rel, a.channel, t0, a.seconds)
            if img is None:
                print(f"skip {rel} t={t0}: missing spectrogram windows")
                continue
            f = out / f"{kind}_{k:02d}_{rel.replace('/', '_')}_t{t0}.png"
            plot_segment(img, rel, t0, a.seconds, a.channel, models,
                         {nm: res[rel][nm] for nm, _ in models}, gt[rel][0],
                         [(nm, ev[rel]) for nm, ev in extra.items()],
                         tuple(int(v) for v in a.rows.split(",")), f)
            print(f"-> {f}")


if __name__ == "__main__":
    main()
