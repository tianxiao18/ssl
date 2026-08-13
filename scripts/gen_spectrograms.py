"""
Generate spectrogram PNGs from raw audio recordings (step 1).

Run this before any of the model scripts. recording_dir is always
data/<dataset>[/<recording>] and spec_dir is always
outputs/spectrograms/<dataset>[/<recording>] -- --dataset is required.

Single recording:
    python scripts/gen_spectrograms.py --dataset <dataset> --recording <exp>/<idx> [--channels 118,35]

All recordings in the dataset (parallel):
    python scripts/gen_spectrograms.py --dataset <dataset> --all [--workers 8]

Example:
    python scripts/gen_spectrograms.py --dataset gerbil_ssl --all --workers 8

Per-dataset dB calibration
---------------------------
The dB clipping window (lo/hi) is NOT universal -- it depends on the recording
rig's amplitude scale, and doesn't transfer across datasets. Known-good values:

  gerbil_ssl (headmic rig): module default, SPEC_LO/SPEC_HI = -70, 0
      (don't pass --lo/--hi/--calibrate; this is what ridge/SAM3/SqueakOut were tuned on)

  dryad_gerbil: --lo -95 --hi -57 (fixed, NOT --calibrate)
      Derived empirically, not from a formula: background noise floor p90 and
      real-call peak dB (from vocalization_df.csv onset/offset) were measured
      across all 5 available wavs via scipy.signal.spectrogram --
        background: p50 ~ -100, p90 ~ -94.2 to -95.3 dB (consistent across files)
        call peaks: max ~ -58.1 to -62.4 dB (consistent across files)
      lo sits at the background p90 (background clips ~black); hi sits a couple
      dB above the loudest observed call peak (calls reach near-white without
      saturating). A per-recording --calibrate pass was tried first and rejected:
      its 6x20s sparse sample window underestimated true call peaks in
      call-sparse recordings, producing an inconsistent per-file range (~24dB
      swing in hi between two recordings) that a fixed, corpus-wide window
      doesn't have.

  gerbil_family: --lo -100 --hi -50 (fixed, NOT --calibrate)
      Files: channel_30_file_148.wav (experiment_497), channel_30_file_055.wav
      (experiment_552); float32, sr=125000. Use --prefix channel --channels 30
      (filenames are channel_30_*.wav, not headmic_*). Derived from the
      calls_*.csv onset/offset columns via scipy.signal.spectrogram across
      both available files:
        background (call windows +/-50ms masked out): p50 ~ -105.7 to -106.0,
          p90 ~ -100.1 to -100.7 dB (consistent across both files)
        per-call peak dB (max bin within each annotated call window):
          p90 ~ -52.0 to -53.4 dB (consistent); raw max per-call peak ranged
          -42.1 to -29.7 (inconsistent, an outlier-driven ~12dB swing --
          matches the reason dryad_gerbil rejected per-recording --calibrate,
          so p90-of-call-peaks was used instead of the max)
      lo sits at the background p90 (background clips ~black); hi sits a
      couple dB above the p90 call peak (typical calls reach near-white
      without saturating; the loudest outlier calls saturate, which is fine).

  Any new dataset: re-derive the same way (don't reuse any of the above pairs)
  -- sample background-only and known-call windows, check whether the
  resulting (lo, hi) is consistent across that dataset's recordings before
  treating it as fixed.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_tracer.spec import calibrate_db_range, load_channel_audio, write_chunk_spectrograms

DATASETS = ("gerbil_ssl", "dryad_gerbil", "gerbil_family")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", required=True, choices=DATASETS)
parser.add_argument("--recording", default=None,
                    help="single experiment_X/idx_Y to process (omit and pass --all instead "
                         "to process every recording in the dataset)")
parser.add_argument("--channels",    default="118,35")
parser.add_argument("--prefix",      default="headmic",
                    help="recording-stream filename prefix, matching '{prefix}_{ch}_*.wav' audio "
                         "(default: headmic, the gerbil_ssl multi-mic rig). A single-stream dataset can "
                         "pass its own label with --channels 0.")
parser.add_argument("--chunk-sec",   type=float, default=1.0)
parser.add_argument("--calibrate",   action="store_true",
                    help="derive the dB clipping window (lo/hi) from each recording's own amplitude "
                         "distribution instead of the module default SPEC_LO/SPEC_HI=(-70, 0), which "
                         "was tuned for the gerbil_ssl headmic rig and can clip a differently-scaled "
                         "recording (e.g. a different rig, float-scaled audio) to near-black.")
parser.add_argument("--lo", type=float, default=None, help="manual dB clip floor (overrides --calibrate)")
parser.add_argument("--hi", type=float, default=None, help="manual dB clip ceiling (overrides --calibrate)")
parser.add_argument("--source-wav", type=Path, default=None,
                    help="raw audio file living outside the data/ layout (e.g. on ceph, arbitrarily "
                         "named). Single-recording mode only. When set, symlinks it into recording_dir "
                         "as '{prefix}_{channel}_{stem}.wav' (creating recording_dir if needed) before "
                         "generating -- so a dataset that isn't already arranged as {prefix}_{ch}_*.wav "
                         "per recording doesn't need a separate prep step. Requires exactly one "
                         "--channels value. recording_dir is unaffected/untouched when this is omitted.")
parser.add_argument("--all",         action="store_true", help="process all experiment_*/idx_* under recording_dir")
parser.add_argument("--workers",     type=int, default=4, help="parallel workers when --all is set (default: 4)")
args = parser.parse_args()

channels = [int(c) for c in args.channels.split(",")]

recording_base = Path("data") / args.dataset
spec_base      = Path("outputs") / "spectrograms" / args.dataset

if args.source_wav is not None:
    if args.all:
        parser.error("--source-wav is only valid without --all (one recording at a time)")
    if not args.recording:
        parser.error("--source-wav requires --recording <exp>/<idx>")
    if len(channels) != 1:
        parser.error("--source-wav requires exactly one --channels value")
    rec_dir = recording_base / args.recording
    rec_dir.mkdir(parents=True, exist_ok=True)
    link = rec_dir / f"{args.prefix}_{channels[0]}_{args.source_wav.stem}.wav"
    if not link.exists():
        link.symlink_to(args.source_wav.resolve())
        print(f"linked {link} -> {args.source_wav}")


def _process_one(recording_dir, spec_dir, channels, chunk_sec, prefix, calibrate, lo, hi):
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for ch in channels:
        result = load_channel_audio(recording_dir, ch, prefix=prefix)
        if result is None:
            print(f"  {Path(recording_dir).name} ch{ch}: no {prefix}_{ch}_*.wav, skipping")
            continue
        sr, audio, base_name = result
        ch_lo, ch_hi = lo, hi
        if calibrate and ch_lo is None and ch_hi is None:
            ch_lo, ch_hi = calibrate_db_range(audio, sr)
        kw = {}
        if ch_lo is not None:
            kw["lo"] = ch_lo
        if ch_hi is not None:
            kw["hi"] = ch_hi
        paths, _ = write_chunk_spectrograms(audio, sr, spec_dir, base_name, chunk_sec, **kw)
        total += len(paths)
    print(f"  {Path(recording_dir).name} → {spec_dir} ({total} chunks)")
    return total


if args.recording:
    _process_one(recording_base / args.recording, spec_base / args.recording, channels,
                 args.chunk_sec, args.prefix, args.calibrate, args.lo, args.hi)
elif args.all:
    def _out_dir(idx_dir):
        return spec_base / idx_dir.parent.name / idx_dir.name

    recordings = [
        (idx_dir, _out_dir(idx_dir))
        for exp_dir in sorted(recording_base.glob("experiment_*"))
        for idx_dir in sorted(exp_dir.glob("idx_*"))
    ]
    print(f"Found {len(recordings)} recordings, running with {args.workers} workers …")

    done, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, str(rec), str(out), channels, args.chunk_sec,
                       args.prefix, args.calibrate, args.lo, args.hi): rec
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
else:
    parser.error("either --recording <exp>/<idx> or --all is required")
