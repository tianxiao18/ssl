"""Convert DAS output (*annotations_gt.csv 'vox' rows) into coco_ch_<ch>.json predictions.

Lets DAS enter a candidate pool as its own detector. Clips are taken from a reference
detector's outputs so the clip sets line up. Each clip gets one image spanning the whole
recording, so an event crossing a 1 s boundary stays one detection.

Usage
-----
    python scripts/das_to_coco.py --dataset dryad_gerbil_full --like sam3_h5
    python scripts/build_candidate_pool.py --dataset dryad_gerbil_full \
        --detectors sam3_h5,ridge,squeakout,das --out ...
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_label.candidates import find_clips
from vox_tracer.coco import image_entry, make_coco

PX_PER_SEC = 1000  # virtual image width scale; only the ratio matters to bbox_to_time


def read_das(csv_path):
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["name"] != "vox":
                continue
            try:
                s, e = float(r["start_seconds"]), float(r["stop_seconds"])
            except (TypeError, ValueError):
                continue
            if not (math.isnan(s) or math.isnan(e)) and e > s:
                out.append((s, e))
    return out


def clip_duration(ref_dir):
    """End of the last window any reference coco file covers, or None."""
    ends = []
    for p in ref_dir.glob("coco_ch_*.json"):
        with open(p) as fh:
            ends += [im["window_end_sec"] for im in json.load(fh)["images"]]
    return max(ends) if ends else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--like", required=True, help="reference detector defining the clip set")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--pred-root", default="outputs")
    ap.add_argument("--name", default="das", help="detector name to write under pred-root")
    ap.add_argument("--channel", type=int, default=0, help="channel the DAS events are attributed to")
    args = ap.parse_args()

    clips = find_clips(args.pred_root, args.dataset, [args.like])
    if not clips:
        sys.exit(f"no {args.like} predictions for {args.dataset}")
    n_ok, n_ev, missing = 0, 0, []
    for rel in clips:
        csvs = sorted((Path(args.data_root) / args.dataset / rel).glob("*annotations_gt.csv"))
        if not csvs:
            missing.append(rel)
            continue
        events = read_das(csvs[0])
        dur = clip_duration(Path(args.pred_root) / args.like / args.dataset / rel)
        dur = max([dur or 0.0] + [e for _, e in events])
        width = math.ceil(dur * PX_PER_SEC)

        coco = make_coco("DAS vox events", "vox", extra_info={"source_csv": str(csvs[0])})
        coco["images"].append(image_entry(0, csvs[0].name, width, 1,
                                          window_start_sec=0.0,
                                          window_end_sec=width / PX_PER_SEC))
        for i, (s, e) in enumerate(events):
            x, w = s * PX_PER_SEC, (e - s) * PX_PER_SEC
            coco["annotations"].append({"id": i, "image_id": 0, "category_id": 1,
                                        "bbox": [x, 0, w, 1], "area": w, "iscrowd": 0})
        out = Path(args.pred_root) / args.name / args.dataset / rel / f"coco_ch_{args.channel}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(coco, fh)
        n_ok += 1
        n_ev += len(events)

    print(f"{n_ok}/{len(clips)} clips written, {n_ev} DAS events "
          f"-> {args.pred_root}/{args.name}/{args.dataset}")
    if missing:
        print(f"{len(missing)} clips had no *annotations_gt.csv, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
