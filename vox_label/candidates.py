"""Build and freeze the candidate-detection pool that a labeling campaign annotates.

A **candidate detection** is a place in the corpus where at least one detector fired.
For each one we know the **detection pattern** z -- which of the K detectors flagged it --
and a human annotator supplies the binary label x, "is this a genuine vocalization?".

Candidate formation is the one upstream choice that changes what precision and recall
even mean: whether two overlapping firings are one candidate or two decides what the
annotator is asked and what the resulting rates estimate. So it is done once, here,
and frozen by hashing the output. The campaign spec records that hash, and rebuilding
the pool with different settings invalidates the campaign rather than silently
redefining its estimands mid-flight.

The rule used (see `group_detections`): pool every detection from every detector and
every channel onto one time axis, single-linkage merge anything that overlaps in time,
and take each connected component as one candidate. Pooling channels matches what
vox_tracer.scoring already does when it scores predictions, so the resulting numbers
stay comparable to the existing metrics.csv -- and it means a call picked up on both
mics is one question to the annotator, not two correlated ones.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from vox_tracer.scoring import bbox_to_time


def load_detections(pred_root, dataset, clip_rel, detector, channels=None):
    """Every detection one detector made on one clip, as (t0, t1, channel, ann_id).

    Reads outputs/<detector>/<dataset>/<clip_rel>/coco_ch_<ch>.json. Times come from
    `vox_tracer.scoring.bbox_to_time`, which converts a bbox's x-extent from its own
    window's pixels back to seconds -- the COCO files store pixel coordinates relative
    to a one-second window, never absolute time.
    """
    out = []
    clip_dir = Path(pred_root) / detector / dataset / clip_rel
    for coco_path in sorted(clip_dir.glob("coco_ch_*.json")):
        ch = int(coco_path.stem.split("_")[-1])
        if channels is not None and ch not in channels:
            continue
        with open(coco_path) as fh:
            coco = json.load(fh)
        windows = {
            im["id"]: (im["window_start_sec"], im["window_end_sec"], im["width"])
            for im in coco["images"]
        }
        for ann in coco["annotations"]:
            win = windows.get(ann["image_id"])
            if win is None:
                continue
            t0, t1 = bbox_to_time(ann["bbox"], *win)
            out.append({"t0": t0, "t1": t1, "ch": ch, "ann_id": ann["id"],
                        "detector": detector})
    return out


def group_detections(detections, gap=0.0):
    """Single-linkage merge of detections that overlap in time; returns list of groups.

    `merge_intervals` in vox_tracer.scoring does the same union but discards which
    inputs landed in which output, and we need that provenance to recover the pattern
    z, so the sweep is repeated here.

    `gap` widens the linkage: two detections separated by less than `gap` seconds still
    merge. At the default 0.0 only genuine positive-length overlap links, so detections
    that merely touch stay separate -- matching the strict `iou_1d(...) > 0` criterion
    the repo's scorer uses to call a prediction a match.
    """
    if not detections:
        return []
    order = sorted(detections, key=lambda d: (d["t0"], d["t1"]))
    groups = [[order[0]]]
    running_end = order[0]["t1"]
    for det in order[1:]:
        if det["t0"] < running_end + gap:
            groups[-1].append(det)
            running_end = max(running_end, det["t1"])
        else:
            groups.append([det])
            running_end = det["t1"]
    return groups


def clip_candidates(pred_root, dataset, clip_rel, detectors, channels=None, gap=0.0):
    """The candidates for one clip: merge across all detectors and channels at once."""
    dets = []
    for det in detectors:
        dets.extend(load_detections(pred_root, dataset, clip_rel, det, channels))
    rows = []
    for group in group_detections(dets, gap=gap):
        fired = {d["detector"] for d in group}
        rows.append({
            "t_start": min(d["t0"] for d in group),
            "t_end": max(d["t1"] for d in group),
            "z": {k: int(k in fired) for k in detectors},
            "channels": sorted({d["ch"] for d in group}),
            "source_anns": [
                {"detector": d["detector"], "ch": d["ch"], "ann_id": d["ann_id"]}
                for d in group
            ],
        })
    return rows


def available_detectors(pred_root, dataset):
    """Prediction variants that exist for this dataset, with their clip counts.

    Which detectors go into the pool is the first real choice a campaign makes -- it
    fixes what z means and therefore which rules can ever be reported -- so the options
    are enumerable rather than something to be guessed from directory names.
    """
    root = Path(pred_root)
    out = {}
    for d in sorted(p.name for p in root.iterdir() if p.is_dir()):
        ds = root / d / dataset
        if not ds.is_dir():
            continue
        clips = {p.parent for p in ds.glob("*/*/coco_ch_*.json")}
        clips |= {p.parent for p in ds.glob("*/coco_ch_*.json")}
        if clips:
            out[d] = len(clips)
    return out


def find_clips(pred_root, dataset, detectors):
    """Clip paths (relative to outputs/<detector>/<dataset>) that EVERY detector covers.

    Restricting to the intersection matters: a clip one detector never ran on would
    otherwise contribute candidates whose z falsely records that detector as silent,
    biasing its recall downward for a reason that has nothing to do with the detector.
    """
    per_detector = []
    for det in detectors:
        root = Path(pred_root) / det / dataset
        clips = {
            p.parent.relative_to(root).as_posix()
            for p in root.glob("*/*/coco_ch_*.json")
        }
        if not clips:  # flat-layout dataset (one level below <dataset>)
            clips = {
                p.parent.relative_to(root).as_posix()
                for p in root.glob("*/coco_ch_*.json")
            }
        per_detector.append(clips)
    return sorted(set.intersection(*per_detector))


CSV_FIELDS = ["cand_id", "clip", "t_start", "t_end", "duration", "n_detectors",
              "channels", "source_anns"]


def build_pool(pred_root, dataset, detectors, out_csv, channels=None, gap=0.0,
               progress=None):
    """Build the whole pool and write it to `out_csv`. Returns (n_candidates, sha256)."""
    fields = CSV_FIELDS[:5] + [f"z_{d}" for d in detectors] + CSV_FIELDS[5:]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clips = find_clips(pred_root, dataset, detectors)
    cand_id = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, clip in enumerate(clips):
            for row in clip_candidates(pred_root, dataset, clip, detectors, channels, gap):
                rec = {
                    "cand_id": cand_id,
                    "clip": clip,
                    "t_start": f"{row['t_start']:.6f}",
                    "t_end": f"{row['t_end']:.6f}",
                    "duration": f"{row['t_end'] - row['t_start']:.6f}",
                    "n_detectors": sum(row["z"].values()),
                    "channels": ",".join(str(c) for c in row["channels"]),
                    "source_anns": json.dumps(row["source_anns"], separators=(",", ":")),
                }
                rec.update({f"z_{d}": row["z"][d] for d in detectors})
                writer.writerow(rec)
                cand_id += 1
            if progress and (i + 1) % progress == 0:
                print(f"  {i + 1}/{len(clips)} clips, {cand_id} candidates", flush=True)
    return cand_id, sha256_file(out_csv)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_pool(csv_path, detectors):
    """Read a frozen pool back. Returns (rows, detectors) with z as a tuple of ints."""
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "cand_id": int(r["cand_id"]),
                "clip": r["clip"],
                "t_start": float(r["t_start"]),
                "t_end": float(r["t_end"]),
                "channels": [int(c) for c in r["channels"].split(",") if c],
                "z": tuple(int(r[f"z_{d}"]) for d in detectors),
            })
    return rows


def pattern_histogram(rows):
    """Counter over detection patterns z -- the pool's shape, and the thing every
    rule's f_V is computed from."""
    return Counter(r["z"] for r in rows)
