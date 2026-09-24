"""Generate ONE continuous-spectrogram HDF5 file per recording (step 1, HDF5 variant).

Replaces the per-second PNG output of scripts/gen_spectrograms.py, which produces
one file per 1-sec chunk (~3.88M files / ~487GB for dryad_gerbil_full's ~1900
recordings -- an inode-quota and small-file-IO problem on shared GPFS/Ceph
regardless of which mount it lands on). This script instead runs ONE STFT over
the whole recording and writes it to a single HDF5 file, internally chunked
(--chunk-sec) so a later 1-sec window read only touches the on-disk block(s) it
needs -- see vox_tracer.spec.write_recording_spectrogram_h5/read_h5_window.

NOT wired into the rest of the pipeline yet: ridge.py, sam3_runner.py,
squeakout_runner.py, evaluate.py, and friends still read per-chunk PNGs from
scripts/gen_spectrograms.py's output and haven't been touched. This script only
produces the HDF5s; validate a recording's output (--recording, then inspect the
.h5 and pull a PNG back out with vox_tracer.spec.read_h5_window) before running
--all across a full corpus.

recording_dir is always data/<dataset>/<relative recording path> and h5_dir is
always outputs/spectrograms_h5/<dataset>/<relative recording path>.h5.
--recording accepts any relative path under data/<dataset>, including a nested
one (e.g. "cohort2_formatted/2020_07_19_..._merged" for dryad_gerbil_full).

Single recording:
    python scripts/gen_spectrograms_h5.py --dataset dryad_gerbil_full \
        --recording cohort2_formatted/2020_07_19_16_32_55_857727_merged \
        --channels 0 --prefix mic --lo -95 --hi -57

All recordings in the dataset (parallel):
    python scripts/gen_spectrograms_h5.py --dataset dryad_gerbil_full --all --workers 8 \
        --channels 0 --prefix mic --lo -95 --hi -57
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_tracer.spec import calibrate_db_range, load_channel_audio, write_recording_spectrogram_h5, SPEC_LO, SPEC_HI

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset",   required=True)
parser.add_argument("--recording", default=None,
                     help="single recording, as a path relative to data/<dataset> "
                          "(omit and pass --all instead to process every recording)")
parser.add_argument("--channels",  default="0")
parser.add_argument("--prefix",    default="mic")
parser.add_argument("--chunk-sec", type=float, default=1.0,
                     help="HDF5 internal storage chunk size, in seconds of columns "
                          "(matches the model's 1-sec input granularity by default)")
parser.add_argument("--calibrate", action="store_true",
                     help="derive (lo, hi) from this recording's own amplitude distribution "
                          "instead of --lo/--hi/module default (see gen_spectrograms.py's "
                          "docstring for why this is usually the wrong call for a new dataset)")
parser.add_argument("--lo", type=float, default=None)
parser.add_argument("--hi", type=float, default=None)
parser.add_argument("--all",       action="store_true")
parser.add_argument("--workers",   type=int, default=4)
args = parser.parse_args()

channels = [int(c) for c in args.channels.split(",")]
data_base = Path("data") / args.dataset
h5_base   = Path("outputs/spectrograms_h5") / args.dataset


def _process_one(rel_recording_dir):
    recording_dir = data_base / rel_recording_dir
    results = []
    for ch in channels:
        wav_matches = sorted(glob(str(recording_dir / f"{args.prefix}_{ch}_*.wav")))
        if not wav_matches:
            print(f"  {rel_recording_dir} ch {ch}: no {args.prefix}_{ch}_*.wav found, skipping")
            continue
        expected_h5 = h5_base / rel_recording_dir / f"{Path(wav_matches[0]).stem}.h5"
        if expected_h5.exists():
            print(f"  {rel_recording_dir} ch {ch}: already done ({expected_h5}), skipping")
            results.append(str(expected_h5))
            continue

        loaded = load_channel_audio(recording_dir, ch, prefix=args.prefix)
        sr, audio, base_name = loaded
        del loaded
        audio = audio.astype("float32", copy=False)  # drop the float64 copy before the big STFT
        if args.calibrate:
            lo, hi = calibrate_db_range(audio, sr)
        else:
            lo = args.lo if args.lo is not None else SPEC_LO
            hi = args.hi if args.hi is not None else SPEC_HI
        h5_path = h5_base / rel_recording_dir / f"{base_name}.h5"
        t0 = time.time()
        h5_path, shape = write_recording_spectrogram_h5(
            audio, sr, h5_path, lo=lo, hi=hi, chunk_sec=args.chunk_sec)
        dt = time.time() - t0
        size_mb = h5_path.stat().st_size / 1e6
        print(f"  {rel_recording_dir} ch {ch}: shape={shape} lo/hi=({lo:.1f},{hi:.1f}) "
              f"-> {h5_path} ({size_mb:.1f} MB, {dt:.1f}s)")
        results.append(str(h5_path))
    return results


def _discover_recordings():
    """Every directory under data_base containing a {prefix}_{ch}_*.wav for any requested channel."""
    seen = set()
    for ch in channels:
        for wav in sorted(data_base.glob(f"**/{args.prefix}_{ch}_*.wav")):
            seen.add(wav.parent.relative_to(data_base))
    return sorted(seen)


if not args.all:
    if args.recording is None:
        parser.error("pass --recording <rel path> or --all")
    _process_one(Path(args.recording))
else:
    recordings = _discover_recordings()
    print(f"Discovered {len(recordings)} recordings under {data_base}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_one, rel): rel for rel in recordings}
        for fut in as_completed(futs):
            fut.result()
