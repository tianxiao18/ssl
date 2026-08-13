"""Materialize *annotations_gt.csv for gerbil_family from its calls_*.csv files, so
scripts/evaluate.py's default GT loader (_load_gt_from_csv, which expects
name/start_seconds/stop_seconds rows) can be reused unchanged.

Each experiment_* dir already holds one calls_exp<exp>_file<file>_ch<ch>.csv with
start_time_file_sec/stop_time_file_sec per annotated call -- unlike dryad_gerbil's
single paper-wide vocalization_df.csv, no cross-file join is needed. All rows are
written with name='vox' (gerbil_family has no "combined"/simultaneous-call concept).

    python scripts/dataset_prep/gerbil_family_gt_to_csv.py --recording-dir data/gerbil_family
"""
import argparse
import csv
from glob import glob
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--recording-dir", default="data/gerbil_family",
                    help="dir containing one experiment_* subdir per recording, each with its own calls_*.csv")
args = parser.parse_args()

recording_dir = Path(args.recording_dir)
exp_dirs = sorted(p for p in recording_dir.iterdir() if p.is_dir() and p.name.startswith("experiment_"))
print(f"Found {len(exp_dirs)} recordings under {recording_dir}: {[p.name for p in exp_dirs]}")

for exp_dir in exp_dirs:
    matches = sorted(glob(str(exp_dir / "calls_*.csv")))
    if not matches:
        print(f"  {exp_dir.name}: no calls_*.csv, skipping")
        continue
    df = pd.read_csv(matches[0], usecols=["start_time_file_sec", "stop_time_file_sec"])
    out_path = exp_dir / f"{exp_dir.name}_annotations_gt.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "start_seconds", "stop_seconds"])
        for s, e in sorted(zip(df["start_time_file_sec"], df["stop_time_file_sec"])):
            writer.writerow(["vox", float(s), float(e)])
    print(f"  {exp_dir.name}: {len(df)} vox intervals -> {out_path}")
