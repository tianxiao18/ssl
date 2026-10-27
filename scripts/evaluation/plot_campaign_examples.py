"""Labeled examples of one detector in a vox_label campaign: TP / FP / FN rows, each a stretch of spectrogram
centred on the labeled candidate with the detector's masks overlaid -> <out>/{tp,fp,fn}.png + examples.csv.

TP = labeled call the detector fired on, FP = labeled non-call it fired on, FN = labeled call it missed.
Rows are seeded-random, at most one per recording, titled with the real Colony_Vox WAV path.

    python scripts/evaluation/plot_campaign_examples.py --out outputs/eval/mongolia_k4_hp15k_examples
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vox_tracer.spec import read_h5_window  # noqa: E402

SRC = "/mnt/ceph/users/neurostatslab/saneslab_data/mongolia/Audiomoth_Files/Colony_Vox/"
NYQ, H, MASK = 96.0, 257, "#22d3ee"


def load(campaign):
    d = ROOT / "outputs/label_campaigns" / campaign
    lab = {}
    for line in open(d / "labels.jsonl"):
        r = json.loads(line)
        if r.get("retracted"):
            lab.pop(r["cand_id"], None)
        else:
            lab[r["cand_id"]] = int(r["label"])
    c = pd.read_csv(d / "candidates.csv")
    c = c[c.cand_id.isin(lab)].copy()
    c["y"] = c.cand_id.map(lab)
    return c, json.loads((d / "campaign.json").read_text())["dataset"]


def mic(row, det):
    anns = json.loads(row.source_anns)
    return next((a["ch"] for a in anns if a["detector"] == det), anns[0]["ch"])


def masks(det, ds, rec, ch, t0, t1):
    j = json.load(open(ROOT / "outputs" / det / ds / rec / f"coco_ch_{ch}.json"))
    ims = {im["id"]: im for im in j["images"] if im["window_end_sec"] > t0 and im["window_start_sec"] < t1}
    out = []
    for a in j["annotations"]:
        im = ims.get(a["image_id"])
        if im is None:
            continue
        sec = (im["window_end_sec"] - im["window_start_sec"]) / im["width"]
        for poly in a.get("segmentation") or []:
            p = np.asarray(poly, float).reshape(-1, 2)
            out.append(np.c_[im["window_start_sec"] + p[:, 0] * sec, (H - 1 - p[:, 1]) / (H - 1) * NYQ])
    return out


def figure(rows, det, ds, all_labs, span, fhi, title_kind, path):
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.9 * len(rows)), facecolor="#ffffff", squeeze=False)
    for ax, r in zip(axes[:, 0], rows.itertuples()):
        ch = mic(r, det)
        mid = (r.t_start + r.t_end) / 2
        t0 = max(0.0, mid - span / 2)
        t1 = t0 + span
        h5 = next((ROOT / "outputs/spectrograms_h5" / ds / r.clip).glob(f"mic_{ch}_*.h5"))
        ax.imshow(read_h5_window(h5, t0, t1), cmap="magma", aspect="auto", extent=[t0, t1, 0, NYQ])
        for p in masks(det, ds, r.clip, ch, t0, t1):
            ax.add_patch(Polygon(p, fc=MASK, ec=MASK, alpha=0.45, lw=0.8))
        for s, e, y in all_labs.get(r.clip, []):
            if e > t0 and s < t1:
                ax.axvspan(s, e, ymin=0.95, ymax=1, color="#ffffff" if y else "#9a9a9a", lw=0)
        ax.axvspan(r.t_start, r.t_end, ymin=0, ymax=0.04, color="#facc15", lw=0)
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, fhi)
        ax.set_ylabel("kHz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="x", useOffset=False)
        wav = ROOT / "data" / ds / r.clip / f"mic_0_{r.clip.split('/')[-1]}.wav"
        src = os.readlink(wav).removeprefix(SRC) if wav.is_symlink() else r.clip
        ax.set_title(f"{src}  mic {ch}  {r.t_start:.3f}-{r.t_end:.3f} s  (cand {r.cand_id}, "
                     f"{r.n_detectors} det)", fontsize=8, loc="left")
    axes[-1, 0].set_xlabel("s", fontsize=8)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path, dpi=110, facecolor="#ffffff")
    plt.close(fig)
    print(title_kind, len(rows), "->", path, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default="mongolia_audiomoth_k4_hp15k")
    ap.add_argument("--detector", default="sam3_ft_traces_gold_v2_hp15k")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--span", type=float, default=1.5)
    ap.add_argument("--fhi", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    c, ds = load(a.campaign)
    all_labs = {k: list(zip(g.t_start, g.t_end, g.y)) for k, g in c.groupby("clip")}
    z = c[f"z_{a.detector}"] == 1
    picked = []
    for kind, sel in (("tp", z & (c.y == 1)), ("fp", z & (c.y == 0)), ("fn", ~z & (c.y == 1))):
        s = c[sel].sample(frac=1, random_state=a.seed)
        one = s.drop_duplicates("clip")  # spread over recordings, then top up
        rows = pd.concat([one, s.drop(one.index)]).head(a.n).sort_values(["clip", "t_start"])
        figure(rows, a.detector, ds, all_labs, a.span, a.fhi, kind, out / f"{kind}.png")
        picked.append(rows.assign(kind=kind, total=int(sel.sum())))
    pd.concat(picked)[["kind", "total", "cand_id", "clip", "t_start", "t_end", "n_detectors", "y"]].to_csv(
        out / "examples.csv", index=False)


if __name__ == "__main__":
    main()
