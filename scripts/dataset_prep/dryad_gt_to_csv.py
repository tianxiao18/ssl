"""Materialize per-recording *annotations_gt.csv files for dryad_gerbil from the
paper's vocalization_df.csv, so scripts/evaluate.py's default GT loader
(_load_gt_from_csv, which expects name/start_seconds/stop_seconds rows) can be
reused unchanged.

vocalization_df.csv has onset/offset per audio_filename for every one of the
paper's 583,237 vocalizations, keyed by the *unprefixed* wav name (no "mic_0_").
All rows are written with name='vox' (dryad has no "combined"/simultaneous-call
annotation concept, so combined_intervals in evaluate.py will just be empty for
this dataset).

    python scripts/dataset_prep/dryad_gt_to_csv.py \
        --vocalization-df /mnt/home/the10/ceph/dataset/dryad_gerbil/vocalization_df.csv \
        --recording-dir data/dryad_gerbil
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--vocalization-df", required=True)
parser.add_argument("--recording-dir", default="data/dryad_gerbil",
                    help="dir containing one subdir per recording (session name = subdir name)")
parser.add_argument("--chunksize", type=int, default=500_000)
args = parser.parse_args()

recording_dir = Path(args.recording_dir)
sessions = sorted(p.name for p in recording_dir.iterdir() if p.is_dir())
wav_to_session = {f"{s}.wav": s for s in sessions}
print(f"Found {len(sessions)} recordings under {recording_dir}: {sessions}")

rows_by_session = {s: [] for s in sessions}
for chunk in pd.read_csv(args.vocalization_df, usecols=["audio_filename", "onset", "offset"],
                         chunksize=args.chunksize):
    sub = chunk[chunk["audio_filename"].isin(wav_to_session)]
    for _, row in sub.iterrows():
        session = wav_to_session[row["audio_filename"]]
        rows_by_session[session].append((float(row["onset"]), float(row["offset"])))

for session, intervals in rows_by_session.items():
    out_path = recording_dir / session / f"{session}_annotations_gt.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "start_seconds", "stop_seconds"])
        for s, e in sorted(intervals):
            writer.writerow(["vox", s, e])
    print(f"  {session}: {len(intervals)} vox intervals -> {out_path}")
