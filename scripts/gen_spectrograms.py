"""
Generate spectrogram PNGs from raw audio recordings (step 1).

Any folder of WAVs:
    python scripts/gen_spectrograms.py --input DIR [--h5] [--lo L --hi H]
Each WAV gets its own output folder mirroring DIR. Then: scripts/run.py <model> --input <output>.

Run this before any of the model scripts. recording_dir is always
data/<dataset>[/<recording>] and spec_dir is always
outputs/spectrograms/<dataset>[/<recording>] -- --dataset is required.

Single recording:
    python scripts/gen_spectrograms.py --dataset <dataset> --recording <exp>/<idx> [--channels 118,35]

All recordings in the dataset (parallel):
    python scripts/gen_spectrograms.py --dataset <dataset> [--workers 8]

Example:
    python scripts/gen_spectrograms.py --dataset gerbil_ssl --workers 8

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

  mongolia_wild_data: --lo -91 --hi -50 (fixed, NOT --calibrate)
      24-bit PCM, sr=192000; load_channel_audio rescales to [-1, 1] and these
      are on that scale. No annotations, so call windows came from unsupervised
      detection (15-45kHz band, 12dB over band median, >=3 frames; 2399 events).
        background: p90 ~ -91.0 dB (spread 1.0 dB across the 12 files)
        event peaks: p90 ~ -52.6 dB corpus-wide, but per-file p90 swings 15.6 dB
          -- hence fixed, same reason dryad_gerbil rejected --calibrate.

  Any new dataset: re-derive the same way (don't reuse any of the above pairs)
  -- sample background-only and known-call windows, check whether the
  resulting (lo, hi) is consistent across that dataset's recordings before
  treating it as fixed.
"""
import argparse
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_tracer.paths import discover_channels, discover_recordings
import numpy as np
from scipy.signal import spectrogram
from vox_tracer.spec import (NOVERLAP, NPERSEG, SPEC_HI, SPEC_LO, calibrate_db_range, load_channel_audio,
                             write_chunk_spectrograms, write_recording_spectrogram_h5)

# Force fork on non-Windows so Mac workers don't re-import this script.
if sys.platform != "win32":
    multiprocessing.set_start_method("fork", force=True)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", type=Path, default=None,
                    help="any folder (or single .wav): every .wav under it becomes its own recording")
parser.add_argument("--out", type=Path, default=None,
                    help="--input mode output folder (default: outputs/spectrograms/<input folder name>)")
parser.add_argument("--h5", action="store_true",
                    help="--input mode: write one HDF5 per WAV instead of 1 s PNGs (use for long/many recordings)")
parser.add_argument("--dataset", default=None, help="any dir name under data/ (alternative to --input)")
parser.add_argument("--recording", default=None,
                    help="single recording, relative to data/<dataset> (omit to process every recording)")
parser.add_argument("--channels",    default=None,
                    help="comma-separated channel ids (default: every {prefix}_<ch>_*.wav found)")
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
parser.add_argument("--all",         action="store_true",
                    help="process every dir under data/<dataset> (any depth) holding {prefix}_*.wav "
                         "(the default when --recording is not given)")
parser.add_argument("--workers",     type=int, default=4, help="parallel workers when --all is set (default: 4)")
args = parser.parse_args()
if (args.input is None) == (args.dataset is None):
    parser.error("pass exactly one of --input or --dataset")
args.all = args.all or (not args.recording and args.source_wav is None)
if sys.platform == "win32" and args.workers > 1:
    print("WARNING: Windows does not support fork; clamping --workers to 1.")
    args.workers = 1

if args.dataset:
    recording_base = Path("data") / args.dataset
    if not recording_base.is_dir() and args.source_wav is None:
        parser.error(f"{recording_base} does not exist")
    if args.channels:
        channels = [int(c) for c in args.channels.split(",")]
    elif args.source_wav is not None:
        parser.error("--source-wav requires --channels")
    else:
        _scope = ([recording_base / args.recording] if args.recording else
                  [d for _, d in discover_recordings(recording_base, f"{args.prefix}_*.wav")])
        channels = discover_channels(_scope, args.prefix)
        print(f"channels (auto): {channels}")
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


def _warn_clipping(audio, sr, lo, hi, name, sec=10.0):
    """Warn if the dB window turns the first few seconds almost all black or white."""
    _, _, pxx = spectrogram(audio[: int(sec * sr)], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    db = 10 * np.log10(pxx + 1e-12)
    black, white = (db <= lo).mean(), (db >= hi).mean()
    if black > 0.98 or white > 0.5:
        print(f"  WARNING {name}: {black:.0%} of pixels black, {white:.0%} white with lo/hi={lo}/{hi}; "
              f"dB p90={np.percentile(db, 90):.0f}, max={db.max():.0f}. Set --lo/--hi for this rig.")


def _run_pool(jobs, workers):
    """Run _process_one over (recording_dir, spec_dir, channels, prefix) jobs."""
    print(f"Found {len(jobs)} recordings, running with {workers} workers …")
    done, failed = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, str(rec), str(out), chs, args.chunk_sec, prefix,
                               args.calibrate, args.lo, args.hi, args.h5): rec
                   for rec, out, chs, prefix in jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as e:
                print(f"ERROR {futures[fut]}: {e}")
                failed += 1
    print(f"\nDone: {done} succeeded, {failed} failed.")


def _input_jobs(src, out_root):
    """One job per .wav under src: spec dir out_root/<rel path>/<stem>/ with an audio_0_<stem>.wav link."""
    src = src.resolve()
    wavs = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() == ".wav")
    jobs = []
    for wav in wavs:
        rel = Path(wav.stem) if src.is_file() else wav.relative_to(src).with_suffix("")
        spec_dir = out_root / rel
        spec_dir.mkdir(parents=True, exist_ok=True)
        link = spec_dir / f"audio_0_{wav.stem}.wav"
        if not link.exists():
            link.symlink_to(wav)
        jobs.append((spec_dir, spec_dir, [0], "audio"))
    return jobs


def _process_one(recording_dir, spec_dir, channels, chunk_sec, prefix, calibrate, lo, hi, h5=False):
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for ch in channels:
        if h5 and list(spec_dir.glob(f"{prefix}_{ch}_*.h5")):
            continue
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
        _warn_clipping(audio, sr, kw.get("lo", SPEC_LO), kw.get("hi", SPEC_HI), base_name)
        if h5:
            write_recording_spectrogram_h5(audio, sr, spec_dir / f"{base_name}.h5", chunk_sec=chunk_sec, **kw)
            total += 1
            continue
        paths, _ = write_chunk_spectrograms(audio, sr, spec_dir, base_name, chunk_sec, **kw)
        total += len(paths)
    print(f"  {Path(recording_dir).name} → {spec_dir} ({total} {'h5' if h5 else 'chunks'})")
    return total


if args.input:
    out_root = args.out or Path("outputs") / "spectrograms" / args.input.resolve().stem
    jobs = _input_jobs(args.input, out_root)
    if not jobs:
        parser.error(f"no .wav files under {args.input}")
    _run_pool(jobs, args.workers)
    print(f"Next: python scripts/run.py <model> --input {out_root}")
elif args.recording:
    _process_one(recording_base / args.recording, spec_base / args.recording, channels,
                 args.chunk_sec, args.prefix, args.calibrate, args.lo, args.hi)
elif args.all:
    _run_pool([(rec_dir, spec_base / rel, channels, args.prefix)
               for rel, rec_dir in discover_recordings(recording_base, f"{args.prefix}_*.wav")], args.workers)
else:
    parser.error("either --recording <exp>/<idx> or --all is required")
