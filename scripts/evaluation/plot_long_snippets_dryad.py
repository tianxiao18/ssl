"""Dryad / Mongolia version of plot_long_snippets.py: 3 s of mic 0, spectrogram then one row per detector.

Neither has dense human GT, so the reference is the vox labeler: every labeled candidate of the dataset's campaigns
in view is marked on top (white = real call, grey = not a call). Windows are centred on labeled candidates,
the busiest ones (most detections over all detectors): --n-pos on real calls, --n-neg on non-calls.

    python scripts/evaluation/plot_long_snippets_dryad.py --out outputs/eval/dryad_7det/long
    python scripts/evaluation/plot_long_snippets_dryad.py --dataset mongolia_audiomoth --out outputs/eval/mongolia_k6/long
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vox_tracer.spec import read_h5_window  # noqa: E402

PRESETS = {  # dataset: (campaigns, [(model, prediction dir)])
    "dryad_gerbil_full": (("dryad_gerbil_full_k7_traces", "dryad_gerbil_full_sub100_k7"),
                          [("squeakout", "squeakout"), ("ridge", "ridge_raw"), ("das", "das"),
                           ("sam3 zero-shot (box)", "sam3_h5"), ("sam3 zero-shot (text)", "sam3_traces_dryad_full"),
                           ("sam3 gold ft", "sam3_ft_traces_gold_v2_dryad"),
                           ("sam3 das_yolo ft", "sam3_ft_traces_dasyolo_labeled_v2_dryad")]),
    "mongolia_audiomoth": (("mongolia_audiomoth_k6",),
                           [("squeakout", "squeakout"), ("ridge", "ridge"), ("sam3 zero-shot (box)", "sam3_best"),
                            ("sam3 zero-shot (text)", "sam3_traces_t037"), ("sam3 gold ft", "sam3_ft_traces_gold_v2"),
                            ("sam3 das_yolo ft", "sam3_ft_traces_dasyolo_labeled_v2")]),
}
DATASET = "dryad_gerbil_full"
SPEC = ROOT / "outputs/spectrograms_h5" / DATASET
CAMPAIGNS, PRED = PRESETS[DATASET]
MASK, H = "#22d3ee", 257


def labeled():
    """[(clip, t_start, t_end, label)] over the dataset's campaigns, last write winning, retractions dropped."""
    out = []
    for c in CAMPAIGNS:
        d = ROOT / "outputs/label_campaigns" / c
        lab = {}
        for line in open(d / "labels.jsonl"):
            r = json.loads(line)
            if r.get("retracted"):
                lab.pop(r["cand_id"], None)
            else:
                lab[r["cand_id"]] = int(r["label"])
        with open(d / "candidates.csv", newline="") as fh:  # streamed: the k7 pool is 2.4 GB
            out += [(r["clip"], float(r["t_start"]), float(r["t_end"]), lab[int(r["cand_id"])])
                    for r in csv.DictReader(fh) if int(r["cand_id"]) in lab]
    return out


def preds(clip):
    """{model: [(t0, t1, annotation, window_start)]} on mic 0; das is time-only."""
    out = {}
    for name, d in PRED:
        j = json.load(open(ROOT / "outputs" / d / DATASET / clip / "coco_ch_0.json"))
        ws = {im["id"]: (im["window_start_sec"], im["width"]) for im in j["images"]}
        rows = []
        for a in j["annotations"]:
            w0, width = ws[a["image_id"]]
            if name == "das":  # bbox in ms on one whole-recording image
                rows.append((a["bbox"][0] / 1000, (a["bbox"][0] + a["bbox"][2]) / 1000, None, 0.0))
            else:
                pps = width  # 1 s windows
                rows.append((w0 + a["bbox"][0] / pps, w0 + (a["bbox"][0] + a["bbox"][2]) / pps, a, (w0, pps)))
        out[name] = rows
    return out


def busy(P, t0, t1):
    return sum(1 for rows in P.values() for s, e, *_ in rows if e > t0 and s < t1)


def plot(clip, t0, t1, P, labs, out):
    h5 = next((SPEC / clip).glob("mic_0_*.h5"))
    img = read_h5_window(h5, t0, t1)
    with h5py.File(h5) as f:
        nyq = f["spec"].attrs["sr"] / 2000
    khz = lambda y: (H - 1 - np.asarray(y, float)) / (H - 1) * nyq
    n = 1 + len(PRED)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.35 * n), facecolor="#ffffff", sharex=True)
    for i, ax in enumerate(axes):
        ax.imshow(img, cmap="magma", aspect="auto", extent=[t0, t1, 0, nyq])
        for s, e, lab in labs:
            ax.axvspan(s, e, ymin=0.93, ymax=1, color="#ffffff" if lab else "#9a9a9a", lw=0)
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, nyq)
        ax.set_yticks([0, 30, 60])
        ax.set_ylabel("spectrogram" if i == 0 else PRED[i - 1][0], rotation=0, ha="right", va="center", fontsize=10)
    for ax, (name, _) in zip(axes[1:], PRED):
        for s, e, a, win in P[name]:
            if e < t0 or s > t1:
                continue
            if a is None:  # das: time span only
                ax.add_patch(plt.Rectangle((s, 0.5), e - s, nyq - 1, fill=False, ec=MASK, lw=1.3))
                continue
            w0, pps = win
            for poly in a.get("segmentation") or []:
                p = np.asarray(poly, float).reshape(-1, 2)
                ax.add_patch(Polygon(np.c_[w0 + p[:, 0] / pps, khz(p[:, 1])], fc=MASK, ec=MASK, alpha=0.55, lw=0.8))
            if not a.get("segmentation"):
                bx, by, bw, bh = a["bbox"]
                ax.add_patch(plt.Rectangle((s, khz(by + bh)), e - s, bh / (H - 1) * nyq, fill=False, ec=MASK, lw=1.3))
    npos = sum(l for *_, l in labs)
    axes[0].set_title(f"{clip}, mic 0: white bars = labeled call ({npos}), grey = labeled not a call ({len(labs) - npos})",
                      fontsize=10, loc="left")
    axes[-1].set_xlabel("s")
    fig.tight_layout(h_pad=0.3)
    name = f"{clip.replace('/', '_')}_t{t0:.2f}-{t1:.2f}_mic0.png"
    fig.savefig(out / name, dpi=110, facecolor="#ffffff")
    plt.close(fig)
    print(name, npos, "calls", len(labs) - npos, "non-calls", flush=True)
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--n-neg", type=int, default=2)
    ap.add_argument("--span", type=float, default=3.0)
    ap.add_argument("--dataset", default="dryad_gerbil_full", choices=list(PRESETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global DATASET, SPEC, CAMPAIGNS, PRED
    DATASET, SPEC = a.dataset, ROOT / "outputs/spectrograms_h5" / a.dataset
    CAMPAIGNS, PRED = PRESETS[a.dataset]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    L = labeled()
    by = {}
    for r in L:
        by.setdefault(r[0], []).append(r[1:])
    cand = []
    for clip, rs in by.items():
        P = preds(clip)
        for s, e, lab in rs:
            t0 = max(0.0, (s + e) / 2 - a.span / 2)
            cand.append((busy(P, t0, t0 + a.span), lab, clip, t0))
    cand.sort(reverse=True)
    rows, used = [], set()
    for want, k in ((1, a.n_pos), (0, a.n_neg)):
        for b, lab, clip, t0 in cand:
            if k == 0:
                break
            if lab != want or clip in used:
                continue
            used.add(clip)
            k -= 1
            t1 = t0 + a.span
            labs = [(s, e, l) for s, e, l in by[clip] if e > t0 and s < t1]
            name = plot(clip, t0, t1, preds(clip), labs, out)
            rows.append({"file": name, "clip": clip, "t0": round(t0, 2), "centre_label": lab, "detections": b,
                         "labeled_calls": sum(l for *_, l in labs), "labeled_noncalls": sum(1 - l for *_, l in labs)})
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
