"""Random dryad_gerbil_full evaluation subset + per-recording GT from the paper's vocalization_df.csv.

Candidates: recordings with an HDF5 spectrogram, a SAM3 box-prompt run (outputs/sam3_h5) and >= 1 row in
vocalization_df.csv (recordings with none are left out: no way to tell silence from "not annotated").
Draws --n of them, cohort-stratified, and writes
  <out>/recordings.txt                     one <cohort>/<recording> per line
  <out>/shard_<k>.txt                      --shards round-robin splits for array jobs
  <out>/gt/<rel>/annotations_gt.csv        name,start_seconds,stop_seconds ('vox' rows)

    python scripts/dataset_prep/dryad_full_eval_set.py --n 100 --out outputs/dryad_full_eval
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocalization-df", default="/mnt/home/the10/ceph/dataset/dryad_gerbil/vocalization_df.csv")
    ap.add_argument("--spec-root", default="outputs/spectrograms_h5/dryad_gerbil_full")
    ap.add_argument("--sam3-root", default="outputs/sam3_h5/dryad_gerbil_full")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/dryad_full_eval")
    a = ap.parse_args()

    v = pd.read_csv(a.vocalization_df, usecols=["audio_filename", "onset", "offset"])
    by_wav = dict(tuple(v.groupby("audio_filename")))
    cands = sorted(str(p.relative_to(a.spec_root)) for p in Path(a.spec_root).glob("*/*") if p.is_dir()
                   and (Path(a.sam3_root) / p.relative_to(a.spec_root) / "coco_ch_0.json").exists()
                   and f"{p.name}.wav" in by_wav)
    cohorts = sorted({c.split("/")[0] for c in cands})
    rng = np.random.default_rng(a.seed)
    pick = []
    for co in cohorts:
        pool = [c for c in cands if c.startswith(co + "/")]
        k = round(a.n * len(pool) / len(cands))
        pick += sorted(rng.choice(pool, k, replace=False).tolist())
        print(f"{co}: {k} of {len(pool)} candidates")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recordings.txt").write_text("\n".join(pick) + "\n")
    for k in range(a.shards):
        (out / f"shard_{k}.txt").write_text("\n".join(pick[k::a.shards]) + "\n")
    n_ev = 0
    for rel in pick:
        g = by_wav[f"{Path(rel).name}.wav"].sort_values("onset")
        (out / "gt" / rel).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"name": "vox", "start_seconds": g.onset, "stop_seconds": g.offset}).to_csv(
            out / "gt" / rel / "annotations_gt.csv", index=False)
        n_ev += len(g)
    print(f"{len(pick)} recordings, {n_ev} GT calls -> {out}")


if __name__ == "__main__":
    main()
