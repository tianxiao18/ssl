"""
Run any vox_tracer model on pre-generated spectrogram PNGs.

Single recording:
    python scripts/run.py ridge     <spec_dir> <out_dir> [options]
    python scripts/run.py squeakout <spec_dir> <out_dir> [options]
    python scripts/run.py sam3      <spec_dir> <out_dir> [options]
    python scripts/run.py das_yolo  <recording_dir> <spec_dir> <out_dir> [options]

All recordings under a root (--all mirrors experiment_*/idx_* structure):
    python scripts/run.py ridge    <spec_base> <out_base> --all --workers 8
    python scripts/run.py das_yolo <data_root> <spec_base> <out_base> --all --workers 1

Output: <out_dir>/coco_ch_<ch>.json per channel.

Examples
--------
# single recording
python scripts/run.py sam3 \
    outputs/spectrograms/experiment_439/idx_000 \
    outputs/sam3/experiment_439/idx_000 \
    --sam3-checkpoint sam3/sam3.pt

# all recordings in parallel
python scripts/run.py sam3 \
    outputs/spectrograms \
    outputs/sam3 \
    --sam3-checkpoint sam3/sam3.pt
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _discover(base):
    """Yield (rel_path, abs_idx_dir) for every experiment_*/idx_* under base."""
    for exp_dir in sorted(Path(base).glob("experiment_*")):
        for idx_dir in sorted(exp_dir.glob("idx_*")):
            yield Path(exp_dir.name) / idx_dir.name, idx_dir


def _run_one(model, spec_dir, out_dir, kw):
    """Dispatch to the appropriate run_* function. Top-level so workers can pickle it."""
    kw = dict(kw)
    if model == "ridge":
        from vox_tracer.ridge import run_ridge
        run_ridge(spec_dir, out_dir, **kw)
    elif model == "squeakout":
        from vox_tracer.squeakout_runner import run_squeakout
        run_squeakout(spec_dir, out_dir, **kw)
    elif model == "sam3":
        from vox_tracer.sam3_runner import run_sam3
        run_sam3(spec_dir, out_dir, **kw)
    elif model == "das_yolo":
        from vox_tracer.das_yolo import run_yolo
        run_yolo(kw.pop("recording_dir"), spec_dir, out_dir, **kw)


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
sub = parser.add_subparsers(dest="model", required=True)


def _add_spec_args(p):
    p.add_argument("spec_dir")
    p.add_argument("out_dir")
    p.add_argument("--channels", default="118,35")
    p.add_argument("--all",     action="store_true",
                   help="process all experiment_*/idx_* under spec_dir; mirror structure into out_dir")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel workers for --all (keep 1 for GPU models, default: 1)")


# --- ridge ---
p_ridge = sub.add_parser("ridge", help="ridge filter segmentation")
_add_spec_args(p_ridge)
p_ridge.add_argument("--filter",        default="sato",  help="sato|meijering|frangi|hessian")
p_ridge.add_argument("--sigmas",        default="2,3,4")
p_ridge.add_argument("--threshold-pct", type=float, default=99.0)
p_ridge.add_argument("--sample-rate",   type=int,   default=125000)
p_ridge.add_argument("--freq-min",      type=float, default=20000.0)

# --- squeakout ---
p_sq = sub.add_parser("squeakout", help="SqueakOut neural segmentation")
_add_spec_args(p_sq)
p_sq.add_argument("--checkpoint", default=None)
p_sq.add_argument("--batch-size", type=int, default=16)

# --- sam3 ---
p_sam3 = sub.add_parser("sam3", help="SAM3 segmentation via sato ridge exemplar")
_add_spec_args(p_sam3)
p_sam3.add_argument("--sam3-checkpoint", default=None)
p_sam3.add_argument("--sigmas",          default="2,3,4")
p_sam3.add_argument("--threshold-pct",   type=float, default=99.0)
p_sam3.add_argument("--score-threshold", type=float, default=0.5)
p_sam3.add_argument("--sample-rate",     type=int,   default=125000)
p_sam3.add_argument("--freq-min",        type=float, default=20000.0)
p_sam3.add_argument("--overwrite",       action="store_true",
                    help="reprocess channels that already have output")

# --- das_yolo ---
p_yolo = sub.add_parser("das_yolo", help="DAS-guided YOLO detection")
p_yolo.add_argument("recording_dir",
                    help="data root (--all) or single recording dir with .wav + DAS CSV")
p_yolo.add_argument("spec_dir",
                    help="spec base (--all) or single pre-generated spec dir")
p_yolo.add_argument("out_dir")
p_yolo.add_argument("--channels",  default="118,35")
p_yolo.add_argument("--chunk-sec", type=float, default=1.0)
p_yolo.add_argument("--all",       action="store_true",
                    help="process all experiment_*/idx_* under recording_dir/spec_dir")
p_yolo.add_argument("--workers",   type=int, default=1)

args = parser.parse_args()
channels = [int(c) for c in args.channels.split(",")]

# Build model kwargs
if args.model == "ridge":
    kw = dict(channels=channels, filter_name=args.filter,
              sigmas=[float(s) for s in args.sigmas.split(",")],
              threshold_pct=args.threshold_pct,
              sample_rate=args.sample_rate, freq_min=args.freq_min)
elif args.model == "squeakout":
    kw = dict(channels=channels, checkpoint=args.checkpoint, batch_size=args.batch_size)
elif args.model == "sam3":
    kw = dict(channels=channels, checkpoint=args.sam3_checkpoint,
              sigmas=[float(s) for s in args.sigmas.split(",")],
              threshold_pct=args.threshold_pct, score_threshold=args.score_threshold,
              sample_rate=args.sample_rate, freq_min=args.freq_min,
              overwrite=args.overwrite)
elif args.model == "das_yolo":
    kw = dict(channels=channels, chunk_sec=args.chunk_sec)

if not args.all:
    if args.model == "das_yolo":
        kw["recording_dir"] = args.recording_dir
    _run_one(args.model, args.spec_dir, args.out_dir, kw)
else:
    spec_base = Path(args.spec_dir)
    out_base  = Path(args.out_dir)

    if args.model == "das_yolo":
        data_base = Path(args.recording_dir)
        tasks = [
            (str(idx_dir), str(out_base / rel), {**kw, "recording_dir": str(data_base / rel)})
            for rel, idx_dir in _discover(spec_base)
        ]
    else:
        tasks = [
            (str(idx_dir), str(out_base / rel), kw)
            for rel, idx_dir in _discover(spec_base)
        ]

    print(f"Found {len(tasks)} recordings, {args.workers} worker(s) …")
    done, failed = 0, 0

    if args.workers == 1:
        for spec_dir, out_dir, task_kw in tasks:
            try:
                _run_one(args.model, spec_dir, out_dir, task_kw)
                done += 1
            except Exception as e:
                print(f"ERROR {spec_dir}: {e}")
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, args.model, s, o, kw_): s
                for s, o, kw_ in tasks
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                    done += 1
                except Exception as e:
                    print(f"ERROR {futures[fut]}: {e}")
                    failed += 1

    print(f"\nDone: {done} succeeded, {failed} failed.")
