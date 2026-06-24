"""
Generate spectrogram PNGs from raw audio recordings (step 1).

Run this before any of the model scripts. Spectrograms are written to spec_dir
and reused by all downstream runners.

Single recording:
    python scripts/gen_spectrograms.py <recording_dir> <spec_dir> [--channels 118,35]

All recordings under a data root (parallel):
    python scripts/gen_spectrograms.py <data_root> <out_base> --all [--workers 8]

Example:
    python scripts/gen_spectrograms.py \\
        /mnt/home/the10/ssl/data \\
        /mnt/home/the10/ssl/outputs/spectrograms \\
        --all --workers 8
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_tracer.spec import load_channel_audio, write_chunk_spectrograms

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("recording_dir", help="single recording dir, or data root when --all is set")
parser.add_argument("spec_dir",      help="output spectrogram dir, or output base when --all is set")
parser.add_argument("--channels",    default="118,35")
parser.add_argument("--chunk-sec",   type=float, default=1.0)
parser.add_argument("--all",         action="store_true", help="process all experiment_*/idx_* under recording_dir")
parser.add_argument("--workers",     type=int, default=4, help="parallel workers when --all is set (default: 4)")
args = parser.parse_args()

channels = [int(c) for c in args.channels.split(",")]


def _process_one(recording_dir, spec_dir, channels, chunk_sec):
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for ch in channels:
        result = load_channel_audio(recording_dir, ch)
        if result is None:
            print(f"  {Path(recording_dir).name} ch{ch}: no headmic_{ch}_*.wav, skipping")
            continue
        sr, audio, base_name = result
        paths, _ = write_chunk_spectrograms(audio, sr, spec_dir, base_name, chunk_sec)
        total += len(paths)
    print(f"  {Path(recording_dir).name} → {spec_dir} ({total} chunks)")
    return total


if not args.all:
    _process_one(args.recording_dir, args.spec_dir, channels, args.chunk_sec)
else:
    data_root = Path(args.recording_dir)
    out_base  = Path(args.spec_dir)

    def _out_dir(idx_dir):
        return out_base / idx_dir.parent.name / idx_dir.name

    recordings = [
        (idx_dir, _out_dir(idx_dir))
        for exp_dir in sorted(data_root.glob("experiment_*"))
        for idx_dir in sorted(exp_dir.glob("idx_*"))
    ]
    print(f"Found {len(recordings)} recordings, running with {args.workers} workers …")

    done, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, str(rec), str(out), channels, args.chunk_sec): rec
            for rec, out in recordings
        }
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as e:
                print(f"ERROR {futures[fut]}: {e}")
                failed += 1

    print(f"\nDone: {done} succeeded, {failed} failed.")
