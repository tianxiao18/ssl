"""AVA spectrograms for dryad_gerbil_full segments from one label source: DAS or SAM3 (sam3 gold ft).

Both arms use the exact paper preprocessing (dryad_to_ava_specs.PREPROCESS_PARAMS, ava get_spec on the raw audio),
so the two VAE datasets differ only in which segments the labels define:
  das   data/dryad_gerbil_full/<rec>/*_annotations_gt.csv (the release's DAS output)
  sam3  outputs/sam3_ft_traces_gold_v2_dryad/dryad_gerbil_full/<rec>/coco_ch_0.json (boxes -> time)
Per recording the intervals are merged where they overlap (DAS rows repeat; SAM3 boxes overlap), segments longer
than max_dur (0.3 s) are dropped as AVA does. Every segment overlapping a human-labeled candidate of --campaign is kept
(they are a uniform sample, so this adds no bias and gives compare_label_sources.py enough labeled points); the rest of
--max-segments is sampled at random (seed) over all recordings.
Writes <out>/syllables_XXXX.hdf5 (128 specs each: specs, onset, offset, recording) and <out>/segments.csv.

    python scripts/vae/labels_to_ava_specs.py --source das --out /mnt/ceph/users/the10/ava/dryad_das_vs_sam3/das
"""
import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from glob import glob
from pathlib import Path

import h5py
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/vae"))
from dryad_to_ava_specs import PREPROCESS_PARAMS, get_spec  # noqa: E402  (also installs the interp2d shim)
from vox_tracer.scoring import bbox_to_time, merge_intervals  # noqa: E402

SPF = 128


def das_intervals(data_root, rec):
    f = glob(str(data_root / rec / "*_annotations_gt.csv"))
    if not f:
        return None
    rows = csv.DictReader(open(f[0]))
    return [(float(r["start_seconds"]), float(r["stop_seconds"])) for r in rows if r["start_seconds"] and r["stop_seconds"]]


def sam3_intervals(sam3_dir, rec):
    f = sam3_dir / rec / "coco_ch_0.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    im = {i["id"]: i for i in d["images"]}
    return [bbox_to_time(a["bbox"], im[a["image_id"]]["window_start_sec"], im[a["image_id"]]["window_end_sec"],
                         im[a["image_id"]]["width"]) for a in d["annotations"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("das", "sam3"), required=True)
    ap.add_argument("--recordings-file", default="outputs/dryad_full_eval/recordings.txt",
                    help="default: the 100-recording subset that has sub100_k7 campaign labels")
    ap.add_argument("--data-root", default="data/dryad_gerbil_full")
    ap.add_argument("--sam3-dir", default="outputs/sam3_ft_traces_gold_v2_dryad/dryad_gerbil_full")
    ap.add_argument("--max-segments", type=int, default=50_000)
    ap.add_argument("--campaign", default="outputs/label_campaigns/dryad_gerbil_full_sub100_k7",
                    help="segments overlapping its labeled candidates are always kept ('' to disable)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.glob("syllables_*.hdf5")):
        raise SystemExit(f"{a.out} already has syllables_*.hdf5; refusing to overwrite")
    a.out.mkdir(parents=True, exist_ok=True)
    data_root, sam3_dir = ROOT / a.data_root, ROOT / a.sam3_dir
    recs = [l.strip() for l in open(ROOT / a.recordings_file) if l.strip()]

    segs, stats = [], Counter()
    for rec in recs:
        iv = das_intervals(data_root, rec) if a.source == "das" else sam3_intervals(sam3_dir, rec)
        if iv is None:
            stats["recordings_missing"] += 1
            continue
        stats["raw_intervals"] += len(iv)
        for s, e in merge_intervals(iv):
            if e - s > PREPROCESS_PARAMS["max_dur"]:
                stats["too_long"] += 1
                continue
            segs.append((rec, s, e))
    stats["segments"] = len(segs)
    forced = set()
    if a.campaign:
        labeled = set()
        for line in open(ROOT / a.campaign / "labels.jsonl"):
            r = json.loads(line)
            (labeled.discard if r.get("retracted") else labeled.add)(r["cand_id"])
        lab_iv = {}
        with open(ROOT / a.campaign / "candidates.csv") as f:
            for r in csv.DictReader(f):
                if int(r["cand_id"]) in labeled:
                    lab_iv.setdefault(r["clip"], []).append((float(r["t_start"]), float(r["t_end"])))
        forced = {i for i, (rec, s, e) in enumerate(segs) if any(ls < e and le > s for ls, le in lab_iv.get(rec, ()))}
    stats["forced_labeled_overlap"] = len(forced)
    rng = np.random.default_rng(a.seed)
    rest = np.array(sorted(set(range(len(segs))) - forced), dtype=int)
    n_rest = max(0, min(a.max_segments - len(forced), len(rest)))
    # labeled-overlap segments first, so the < 128 remainder AVA drops never contains them
    keep = sorted(forced) + sorted(rng.choice(rest, n_rest, replace=False).tolist())
    segs = [segs[i] for i in keep]
    print(dict(stats), f"-> sampled {len(segs)}", flush=True)

    target_freqs = np.linspace(PREPROCESS_PARAMS["min_freq"], PREPROCESS_PARAMS["max_freq"], PREPROCESS_PARAMS["num_freq_bins"])
    buf, written, n_file, invalid = [], [], 0, 0
    cur_rec, audio, fs = None, None, None

    def flush(final=False):
        nonlocal buf, n_file
        while len(buf) >= SPF:
            chunk, buf = buf[:SPF], buf[SPF:]
            with h5py.File(a.out / f"syllables_{n_file:04d}.hdf5", "w") as f:
                f.create_dataset("specs", data=np.stack([c[3] for c in chunk]))
                f.create_dataset("onset", data=np.array([c[1] for c in chunk], dtype=np.float32))
                f.create_dataset("offset", data=np.array([c[2] for c in chunk], dtype=np.float32))
                f.create_dataset("recording", data=np.array([c[0] for c in chunk], dtype="S96"))
            written.extend(c[:3] for c in chunk)
            n_file += 1
        if final and buf:
            print(f"dropped the last {len(buf)} specs (AVA needs {SPF} per file)")

    for rec, s, e in segs:
        if rec != cur_rec:
            wav = glob(str(data_root / rec / "*.wav"))[0]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=WavFileWarning)
                fs, audio = wavfile.read(wav, mmap=True)
            cur_rec = rec
        spec, valid = get_spec(s, e, audio, PREPROCESS_PARAMS, fs, target_freqs=target_freqs)
        if not valid or not np.any(spec):
            invalid += 1
            continue
        buf.append((rec, s, e, np.clip(spec, 0.0, 1.0).astype(np.float32)))
        if len(buf) >= SPF:
            flush()
        if (len(written) + len(buf)) % 10000 == 0:
            print(f"  {len(written) + len(buf)} specs", flush=True)
    flush(final=True)
    with open(a.out / "segments.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "recording", "onset", "offset"])
        w.writerows((i, r, round(s, 5), round(e, 5)) for i, (r, s, e) in enumerate(written))
    stats.update(invalid=invalid, written=len(written))
    (a.out / "stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(dict(stats), f"-> {a.out}")


if __name__ == "__main__":
    main()
