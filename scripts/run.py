"""
Run any vox_tracer model on pre-generated spectrogram PNGs.

spec_dir is always outputs/spectrograms/<dataset>[/<recording>] and out_dir is
always <out_dir>/<dataset>[/<recording>] -- --dataset is required and out_dir is
just the stage/variant name (e.g. "outputs/ridge_flatness", "outputs/sam3_best").

Single recording:
    python scripts/run.py ridge     <out_dir> --dataset <dataset> --recording <exp>/<idx> [options]
    python scripts/run.py das_yolo  <out_dir> --dataset <dataset> --recording <exp>/<idx> [options]

All recordings in the dataset:
    python scripts/run.py ridge    <out_dir> --dataset <dataset> --all --workers 8
    python scripts/run.py das_yolo <out_dir> --dataset <dataset> --all --workers 1

Output: <out_dir>/<dataset>/coco_ch_<ch>.json per channel.

Examples
--------
# single recording
python scripts/run.py sam3 outputs/sam3 --dataset gerbil_ssl \
    --recording experiment_445/idx_011 --sam3-checkpoint sam3/sam3.pt

# all recordings in parallel
python scripts/run.py sam3 outputs/sam3 --dataset gerbil_ssl \
    --all --sam3-checkpoint sam3/sam3.pt
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASETS = ("gerbil_ssl", "dryad_gerbil", "gerbil_family")


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
    p.add_argument("out_dir", help="stage/variant name, e.g. outputs/ridge_flatness "
                                    "(dataset is appended automatically)")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--recording", default=None,
                   help="single experiment_X/idx_Y to process (omit and pass --all instead "
                        "to process every recording in the dataset)")
    p.add_argument("--channels", default="118,35")
    p.add_argument("--prefix",  default="headmic",
                   help="recording-stream filename prefix, matching '{prefix}_{ch}_..._t{t0}-{t1}.png' "
                        "spectrograms and '{prefix}_{ch}_*.wav' audio (default: headmic, the gerbil_ssl "
                        "multi-mic rig). A single-stream dataset can pass its own label with --channels 0.")
    p.add_argument("--all",     action="store_true",
                   help="process all experiment_*/idx_* in the dataset; mirror structure into out_dir")
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
p_ridge.add_argument("--min-area",      type=int,   default=50,
                     help="stage-1 component-area cut (cross-validated best)")
p_ridge.add_argument("--recording-dir", default=None,
                     help="dir with the headmic wavs for the centroid gate "
                          "(default: data/<dataset>/<experiment>/<idx> inferred from spec_dir)")
# stage-2 detection filters (cross-validated: duration + spectral-centroid gates)
p_ridge.add_argument("--max-mask-area-frac",  type=float, default=0.15)
p_ridge.add_argument("--min-freq-sweep-frac", type=float, default=0.0,
                     help="frequency-sweep gate (0 = disabled; dropped by the filter search)")
p_ridge.add_argument("--min-mask-cols",       type=int,   default=9,
                     help="reject detections spanning fewer than this many time columns")
p_ridge.add_argument("--min-centroid-hz",     type=float, default=25000.0,
                     help="reject detections whose band-limited spectral centroid is below this "
                          "(pass 0 to disable)")
p_ridge.add_argument("--max-flatness",        type=float, default=None,
                     help="reject detections whose band-limited spectral flatness exceeds this "
                          "(disabled unless set; flatness-only best is ~0.21). To run flatness "
                          "as the sole stage-2 filter also pass --min-mask-cols 0 --min-centroid-hz 0")
p_ridge.add_argument("--reject-out-dir",      default=None,
                     help="also write every stage-2-rejected stage-1 candidate here, tagged with "
                          "extra.reject_reason (diagnostic only, not scored). With --all this is "
                          "mirrored per-recording the same way out_dir is.")

# --- squeakout ---
p_sq = sub.add_parser("squeakout", help="SqueakOut neural segmentation")
_add_spec_args(p_sq)
p_sq.add_argument("--checkpoint", default=None)
p_sq.add_argument("--batch-size", type=int, default=16)
p_sq.add_argument("--mask-threshold", type=float, default=None,
                  help="sigmoid cut to binarize masks (default: model's ~0.51). Lower it "
                       "(e.g. 0.1) for a permissive run so pr_curves.py has a high-recall arm.")
p_sq.add_argument("--overwrite",  action="store_true",
                  help="reprocess channels that already have output")
# stage-2 detection filters (shared with ridge/sam3 so the comparison is fair)
p_sq.add_argument("--recording-dir", default=None,
                  help="dir with the headmic wavs for the centroid gate "
                       "(default: data/<dataset>/<experiment>/<idx> inferred from spec_dir)")
p_sq.add_argument("--max-mask-area-frac",  type=float, default=0.15)
p_sq.add_argument("--min-freq-sweep-frac", type=float, default=0.0,
                  help="frequency-sweep gate (0 = disabled; dropped by the filter search)")
p_sq.add_argument("--min-mask-cols",       type=int,   default=9,
                  help="reject components spanning fewer than this many time columns")
p_sq.add_argument("--min-centroid-hz",     type=float, default=25000.0,
                  help="reject components whose band-limited spectral centroid is below this "
                       "(pass 0 to disable)")
p_sq.add_argument("--max-flatness",        type=float, default=None,
                  help="reject components whose band-limited spectral flatness exceeds this "
                       "(disabled unless set)")

# --- sam3 ---
p_sam3 = sub.add_parser("sam3", help="SAM3 segmentation via sato ridge exemplar")
_add_spec_args(p_sam3)
p_sam3.add_argument("--sam3-checkpoint", default=None)
p_sam3.add_argument("--sigmas",          default="2,3,4")
p_sam3.add_argument("--threshold-pct",   type=float, default=99.0)
p_sam3.add_argument("--score-threshold", type=float, default=0.5)
p_sam3.add_argument("--sample-rate",          type=int,   default=125000)
p_sam3.add_argument("--freq-min",             type=float, default=20000.0)
# stage 1: ridge-filter candidate component filters
p_sam3.add_argument("--min-area",             type=int,   default=30)
p_sam3.add_argument("--vert-aspect",          type=float, default=5.0)
p_sam3.add_argument("--horiz-aspect",         type=float, default=0.2)
p_sam3.add_argument("--close-kernel",         default="7,3", help="morph-close kernel 'w,h'")
# stage 2: SAM3 post-hoc mask filters
p_sam3.add_argument("--recording-dir",        default=None,
                    help="dir with the headmic wavs for the centroid gate "
                         "(default: data/<dataset>/<experiment>/<idx> inferred from spec_dir)")
p_sam3.add_argument("--max-mask-area-frac",   type=float, default=0.15)
p_sam3.add_argument("--min-freq-sweep-frac",  type=float, default=0.0,
                    help="frequency-sweep gate (0 = disabled; dropped by the filter search)")
p_sam3.add_argument("--min-mask-cols",        type=int,   default=9,
                    help="reject SAM3 masks spanning fewer than this many time columns")
p_sam3.add_argument("--min-centroid-hz",      type=float, default=25000.0,
                    help="reject SAM3 masks whose band-limited spectral centroid is below this "
                         "(pass 0 to disable)")
p_sam3.add_argument("--max-flatness",         type=float, default=None,
                    help="reject SAM3 masks whose band-limited spectral flatness exceeds this "
                         "(disabled unless set)")
p_sam3.add_argument("--overwrite",            action="store_true",
                    help="reprocess channels that already have output")

# --- das_yolo ---
p_yolo = sub.add_parser("das_yolo", help="DAS-guided YOLO detection")
p_yolo.add_argument("out_dir", help="stage/variant name, e.g. outputs/das_yolo "
                                     "(dataset is appended automatically)")
p_yolo.add_argument("--dataset", required=True, choices=DATASETS)
p_yolo.add_argument("--recording", default=None,
                    help="single experiment_X/idx_Y to process (omit and pass --all instead "
                         "to process every recording in the dataset)")
p_yolo.add_argument("--channels",  default="118,35")
p_yolo.add_argument("--chunk-sec", type=float, default=1.0)
p_yolo.add_argument("--all",       action="store_true",
                    help="process all experiment_*/idx_* in the dataset")
p_yolo.add_argument("--workers",   type=int, default=1)

args = parser.parse_args()
channels = [int(c) for c in args.channels.split(",")]

# Build model kwargs
if args.model == "ridge":
    kw = dict(channels=channels, filter_name=args.filter,
              sigmas=[float(s) for s in args.sigmas.split(",")],
              threshold_pct=args.threshold_pct,
              sample_rate=args.sample_rate, freq_min=args.freq_min,
              min_area=args.min_area, recording_dir=args.recording_dir,
              max_mask_area_frac=args.max_mask_area_frac,
              min_freq_sweep_frac=args.min_freq_sweep_frac,
              min_mask_cols=args.min_mask_cols,
              min_centroid_hz=args.min_centroid_hz,
              max_flatness=args.max_flatness,
              prefix=args.prefix)
elif args.model == "squeakout":
    kw = dict(channels=channels, checkpoint=args.checkpoint, batch_size=args.batch_size,
              mask_threshold=args.mask_threshold, overwrite=args.overwrite,
              recording_dir=args.recording_dir,
              max_mask_area_frac=args.max_mask_area_frac,
              min_freq_sweep_frac=args.min_freq_sweep_frac,
              min_mask_cols=args.min_mask_cols,
              min_centroid_hz=args.min_centroid_hz,
              max_flatness=args.max_flatness,
              prefix=args.prefix)
elif args.model == "sam3":
    kw = dict(channels=channels, checkpoint=args.sam3_checkpoint,
              sigmas=[float(s) for s in args.sigmas.split(",")],
              threshold_pct=args.threshold_pct, score_threshold=args.score_threshold,
              sample_rate=args.sample_rate, freq_min=args.freq_min,
              min_area=args.min_area, vert_aspect=args.vert_aspect,
              horiz_aspect=args.horiz_aspect,
              close_kernel=tuple(int(s) for s in args.close_kernel.split(",")),
              recording_dir=args.recording_dir,
              max_mask_area_frac=args.max_mask_area_frac,
              min_freq_sweep_frac=args.min_freq_sweep_frac,
              min_mask_cols=args.min_mask_cols,
              min_centroid_hz=args.min_centroid_hz,
              max_flatness=args.max_flatness,
              overwrite=args.overwrite,
              prefix=args.prefix)
elif args.model == "das_yolo":
    kw = dict(channels=channels, chunk_sec=args.chunk_sec)

spec_base = Path("outputs") / "spectrograms" / args.dataset
out_base  = Path(args.out_dir) / args.dataset
reject_base = (Path(args.reject_out_dir) / args.dataset
               if args.model == "ridge" and args.reject_out_dir else None)
data_base = Path("data") / args.dataset if args.model == "das_yolo" else None

if args.recording:
    spec_dir = spec_base / args.recording
    out_dir  = out_base / args.recording
    if args.model == "das_yolo":
        kw["recording_dir"] = str(data_base / args.recording)
    elif args.model == "ridge" and reject_base:
        kw["reject_out_dir"] = str(reject_base / args.recording)
    _run_one(args.model, str(spec_dir), str(out_dir), kw)
elif args.all:
    if args.model == "das_yolo":
        tasks = [
            (str(idx_dir), str(out_base / rel), {**kw, "recording_dir": str(data_base / rel)})
            for rel, idx_dir in _discover(spec_base)
        ]
    elif args.model == "ridge" and reject_base:
        tasks = [
            (str(idx_dir), str(out_base / rel),
             {**kw, "reject_out_dir": str(reject_base / rel)})
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
else:
    parser.error("either --recording <exp>/<idx> or --all is required")
