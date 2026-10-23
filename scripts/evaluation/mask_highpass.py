"""Effect of a 15 kHz high-pass on detector masks (Mongolia AudioMoth, the k6 pool's recordings).

Each mask polygon is rasterized on its bbox; pixels below the cut are dropped. Per detector: how many masks
vanish, are truncated, or are untouched, and the fraction of mask pixels and of time span kept. Also which
labeled mongolia_audiomoth_k6 candidates keep at least one mask -> outputs/eval/mongolia_highpass/.

    python scripts/evaluation/mask_highpass.py --cut-khz 15
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vox_label.candidates import find_clips  # noqa: E402

DS = "mongolia_audiomoth"
CAMPAIGN = ROOT / "outputs/label_campaigns/mongolia_audiomoth_k6"
DETS = [("squeakout", "squeakout"), ("ridge", "ridge"), ("sam3 zero-shot (box)", "sam3_best"),
        ("sam3 zero-shot (text)", "sam3_traces_t037"), ("sam3 gold ft", "sam3_ft_traces_gold_v2"),
        ("sam3 das_yolo ft", "sam3_ft_traces_dasyolo_labeled_v2")]


def nyquist(clip):
    h5 = next((ROOT / "outputs/spectrograms_h5" / DS / clip).glob("mic_0_*.h5"))
    with h5py.File(h5) as f:
        return f["spec"].attrs["sr"] / 2


def mask_split(ann, cut_row):
    """(pixels, pixels kept, cols, cols kept) of one mask; kept = rows <= cut_row (frequency >= cut)."""
    x, y, w, h = ann["bbox"]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    W, H = int(np.ceil(x + w)) - x0 + 1, int(np.ceil(y + h)) - y0 + 1
    m = np.zeros((H, W), np.uint8)
    for poly in ann.get("segmentation") or []:
        p = np.round(np.asarray(poly, float).reshape(-1, 2) - [x0, y0]).astype(np.int32)
        cv2.fillPoly(m, [p], 1)
    if not m.any():  # degenerate polygon: fall back to the bbox
        m[:] = 1
    rows = np.arange(H)[:, None] + y0
    keep = m * (rows <= cut_row)
    return int(m.sum()), int(keep.sum()), int(m.any(0).sum()), int(keep.any(0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut-khz", type=float, default=15.0)
    ap.add_argument("--out", default=ROOT / "outputs/eval/mongolia_highpass", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clips = [l.strip() for l in open(ROOT / "outputs/mongolia_audiomoth/campaign_sample.txt") if l.strip()]
    assert not set(clips) - set(find_clips(ROOT / "outputs", DS, [d for _, d in DETS])), "sample not fully covered"
    nyq = {c: nyquist(c) for c in clips}
    assert len(set(nyq.values())) == 1, set(nyq.values())

    # labeled k6 candidates: which annotations they hold
    lab = {}
    for line in open(CAMPAIGN / "labels.jsonl"):
        r = json.loads(line)
        if r.get("retracted"):
            lab.pop(r["cand_id"], None)
        elif "label" in r:
            lab[r["cand_id"]] = int(r["label"])
    csv.field_size_limit(1 << 30)
    cand_anns, need = {}, set()
    with open(CAMPAIGN / "candidates.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            cid = int(r["cand_id"])
            if cid in lab:
                keys = [(s["detector"], r["clip"], s["ch"], s["ann_id"]) for s in json.loads(r["source_anns"])]
                cand_anns[cid] = keys
                need.update(keys)
    ann_kept = {}

    rows = []
    for name, det in DETS:
        st = dict(model=name, masks=0, removed=0, truncated=0, unchanged=0, px=0, px_kept=0, cols=0, cols_kept=0)
        for clip in clips:
            cut_row = (257 - 1) * (1 - a.cut_khz * 1000 / nyq[clip])  # flipud: row 0 = Nyquist
            for p in sorted((ROOT / "outputs" / det / DS / clip).glob("coco_ch_*.json")):
                ch = int(p.stem.split("_")[-1])
                for ann in json.load(open(p))["annotations"]:
                    px, pk, cl, ck = mask_split(ann, cut_row)
                    st["masks"] += 1
                    st["removed" if pk == 0 else "unchanged" if pk == px else "truncated"] += 1
                    st["px"] += px; st["px_kept"] += pk; st["cols"] += cl; st["cols_kept"] += ck
                    if (det, clip, ch, ann["id"]) in need:
                        ann_kept[(det, clip, ch, ann["id"])] = pk > 0
        st.update(recs=len(clips), masks_per_rec_before=round(st["masks"] / len(clips)),
                  masks_per_rec_after=round((st["masks"] - st["removed"]) / len(clips)),
                  frac_removed=st["removed"] / st["masks"], frac_truncated=st["truncated"] / st["masks"],
                  frac_unchanged=st["unchanged"] / st["masks"], frac_px_kept=st["px_kept"] / st["px"],
                  frac_time_kept=st["cols_kept"] / st["cols"])
        rows.append(st)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items() if k != "model"}, flush=True)
    with open(a.out / "mask_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lrows = [{"cand_id": cid, "label": lab[cid], "survives": int(any(ann_kept.get(k, False) for k in ks)),
              "detectors": "+".join(sorted({k[0] for k in ks}))} for cid, ks in cand_anns.items()]
    with open(a.out / "labeled_candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lrows[0]))
        w.writeheader()
        w.writerows(lrows)
    for l in (1, 0):
        sub = [r for r in lrows if r["label"] == l]
        print(f"labeled {'call' if l else 'non-call'}: {sum(r['survives'] for r in sub)}/{len(sub)} keep a mask")

    # stacked bars: share of masks removed / truncated / unchanged
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor="#ffffff")
    names = [r["model"] for r in rows][::-1]
    left = np.zeros(len(rows))
    for key, lbl, col in (("frac_removed", "removed", "#1c3f7a"), ("frac_truncated", "truncated", "#4f86d0"),
                          ("frac_unchanged", "unchanged", "#b9d2f0")):
        v = np.array([r[key] for r in rows][::-1])
        ax.barh(names, v, left=left, height=0.62, color=col, edgecolor="#ffffff", linewidth=2, label=lbl)
        left += v
    for i, r in enumerate(rows[::-1]):
        ax.text(1.01, i, f"{r['frac_removed']:.0%} removed", va="center", fontsize=8, color="#1a1a19")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"share of masks, high-pass at {a.cut_khz:g} kHz")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "mask_highpass.png", dpi=150, facecolor="#ffffff")
    print("wrote", a.out / "mask_highpass.png")


if __name__ == "__main__":
    main()
