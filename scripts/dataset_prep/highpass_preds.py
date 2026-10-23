"""Copy prediction trees keeping only mask pixels at or above a cutoff frequency.

Masks wholly above the cut are copied unchanged, masks wholly below are dropped, and masks that straddle it are
clipped and re-traced (new polygons and bbox). Same pixel -> Hz mapping as mask_highpass.py.

    python scripts/dataset_prep/highpass_preds.py --dataset mongolia_audiomoth --cut-khz 15 \
        --detectors sam3_best,sam3_traces_t037 --clips outputs/mongolia_audiomoth/campaign_sample.txt
    -> outputs/<detector>_hp15k/mongolia_audiomoth/...
"""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def clip_mask(ann, cut_row):
    """The annotation with pixels below `cut_row` removed; None if nothing is left, `ann` if untouched."""
    x, y, w, h = ann["bbox"]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    W, H = int(np.ceil(x + w)) - x0 + 1, int(np.ceil(y + h)) - y0 + 1
    m = np.zeros((H, W), np.uint8)
    for poly in ann.get("segmentation") or []:
        p = np.round(np.asarray(poly, float).reshape(-1, 2) - [x0, y0]).astype(np.int32)
        cv2.fillPoly(m, [p], 1)
    if not m.any():
        m[:] = 1
    keep_rows = (np.arange(H) + y0) <= cut_row
    if keep_rows.all() or not m[~keep_rows].any():
        return ann
    m[~keep_rows] = 0
    if not m.any():
        return None
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg = [(c.reshape(-1, 2) + [x0, y0]).astype(float).ravel().tolist() for c in cs if len(c) >= 3]
    ys, xs = np.nonzero(m)
    out = dict(ann)
    out["segmentation"] = seg or [[float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1),
                                   float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)]]
    out["bbox"] = [float(x0 + xs.min()), float(y0 + ys.min()), float(xs.max() - xs.min() + 1),
                   float(ys.max() - ys.min() + 1)]
    out["area"] = float(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--detectors", required=True)
    ap.add_argument("--clips", required=True, help="text file of clip paths, one per line")
    ap.add_argument("--cut-khz", type=float, required=True)
    a = ap.parse_args()
    clips = [l.strip() for l in open(a.clips) if l.strip()]
    suffix = f"_hp{a.cut_khz:g}k"
    for det in a.detectors.split(","):
        dst_root = ROOT / "outputs" / f"{det}{suffix}" / a.dataset
        if dst_root.exists():
            raise SystemExit(f"{dst_root} exists; refusing to overwrite")
        n = kept = clipped = 0
        for clip in clips:
            h5 = next((ROOT / "outputs/spectrograms_h5" / a.dataset / clip).glob("mic_0_*.h5"))
            with h5py.File(h5) as f:
                nyq = f["spec"].attrs["sr"] / 2
            for p in sorted((ROOT / "outputs" / det / a.dataset / clip).glob("coco_ch_*.json")):
                d = json.load(open(p))
                H = {im["id"]: im["height"] for im in d["images"]}
                anns = []
                for ann in d["annotations"]:
                    n += 1
                    cut_row = (H[ann["image_id"]] - 1) * (1 - a.cut_khz * 1000 / nyq)  # flipud: row 0 = Nyquist
                    new = clip_mask(ann, cut_row)
                    if new is not None:
                        anns.append(new)
                        clipped += new is not ann
                d["annotations"] = anns
                kept += len(anns)
                d.setdefault("info", {})["highpass_khz"] = a.cut_khz
                out = dst_root / clip / p.name
                out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(d, open(out, "w"))
        print(f"{det}: kept {kept}/{n} masks ({clipped} clipped) -> {dst_root}", flush=True)


if __name__ == "__main__":
    main()
