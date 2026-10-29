"""Progress of a Mongolia detector run (sample_mongolia_all.py picks in --dir) -> --out (default outputs/eval/mongolia_all/).

per_rec.csv  detector x recording: output present, detections
summary.csv  per detector: recordings done of the sample, detections, median per recording, zero-detection recordings
tasks.csv    per array task (from sacct.psv, a saved `sacct -P` snapshot): detector, audio h, elapsed, GPU-h / audio h;
             running tasks use the recordings finished so far
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DETS = {"sam3_best": "sam3_best", "squeakout": "squeakout", "ridge": "ridge", "zs_traces": "sam3_traces_mongolia_am",
        "gold_ft": "sam3_ft_traces_gold_v2", "dasyolo_ft": "sam3_ft_traces_dasyolo_labeled_v2"}
RATE = {"sam3_best": 0.159, "zs_traces": 0.073, "gold_ft": 0.077, "dasyolo_ft": 0.088, "squeakout": 0.010}


def lines(p):
    return [l.split("\t")[0].strip() for l in open(p) if l.strip()]


def per_rec(recs, dets):
    rows = []
    for det, d in dets.items():
        for rec in recs:
            f = ROOT / "outputs" / d / "mongolia_audiomoth" / rec / "coco_ch_0.json"
            rows.append({"detector": det, "recording": rec, "colony": rec.split("/")[0], "done": f.exists(),
                         "n_det": len(json.load(open(f))["annotations"]) if f.exists() else None})
    return pd.DataFrame(rows)


def tasks(dur, ALL, OUT):
    s = pd.read_csv(OUT / "sacct.psv", sep="|")
    s = s[s.JobID.str.contains("_")].copy()
    s["task"] = s.JobID.str.split("_").str[1].str.strip("[]").apply(  # pending ranges like 7309524_[82-87]
        lambda t: list(range(int(t.split("-")[0]), int(t.split("-")[-1]) + 1)))
    s = s.explode("task")
    gpu = [l.split("\t") for l in open(ALL / "gpu_tasks.tsv") if l.strip()]
    rows = []
    for _, r in s.iterrows():
        name, t = r.JobName, r.task
        if name == "mongolia-all-detectors":
            det, lst = gpu[t][0], ROOT / gpu[t][1].strip()
        else:
            det, lst = ("specs", ALL / f"specs_{t}.txt") if name.startswith("specs") else ("ridge", ALL / f"ridge_{t}.txt")
        recs = lines(lst)
        h = sum(dur[x] for x in recs) / 3600
        if r.State == "RUNNING" and det in DETS:  # partial: finished recordings (a task processes its list in order)
            done = [x for x in recs if (ROOT / "outputs" / DETS[det] / "mongolia_audiomoth" / x / "coco_ch_0.json").exists()]
            dh = sum(dur[x] for x in done) / 3600
        else:
            done, dh = (recs, h) if r.State == "COMPLETED" else ([], 0)
        el = r.ElapsedRaw / 3600 if r.State in ("COMPLETED", "RUNNING") else None
        rows.append({"job": f"{r.JobID.split('_')[0]}_{t}", "detector": det, "state": r.State, "recordings": len(recs), "audio_h": round(h, 2),
                     "done_recordings": len(done), "done_audio_h": round(dh, 2), "elapsed_h": round(el, 3) if el else None,
                     "gpu_h_per_audio_h": round(el / dh, 4) if el and dh and det in RATE else None,
                     "planned_rate": RATE.get(det)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/mongolia_audiomoth/all")
    ap.add_argument("--out", default="outputs/eval/mongolia_all")
    ap.add_argument("--detectors", nargs="+", default=list(DETS))
    a = ap.parse_args()
    ALL, OUT = ROOT / a.dir, ROOT / a.out
    recs = lines(ALL / "sample.txt")
    dur = {k: float(v) for k, v in (l.split("\t") for l in open(ROOT / "outputs/mongolia_audiomoth/durations.txt"))}
    df = per_rec(recs, {d: DETS[d] for d in a.detectors})
    df.to_csv(OUT / "per_rec.csv", index=False)
    g = df[df.done].groupby("detector").n_det
    s = pd.DataFrame({"sample": len(recs), "done": df.groupby("detector").done.sum(), "detections": g.sum(),
                      "median_per_rec": g.median(), "zero_det_recs": g.apply(lambda x: (x == 0).sum())}
                     ).reindex(a.detectors).reset_index()
    s.to_csv(OUT / "summary.csv", index=False)
    t = tasks(dur, ALL, OUT)
    t.to_csv(OUT / "tasks.csv", index=False)
    print(s.to_string(index=False))
    print(t.groupby(["detector", "state"]).agg(n=("job", "size"), audio_h=("audio_h", "sum"), elapsed_h=("elapsed_h", "sum"),
                                                rate=("gpu_h_per_audio_h", "median"), planned=("planned_rate", "first")))


if __name__ == "__main__":
    main()
