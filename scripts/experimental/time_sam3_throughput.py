"""DRAFT / diagnostic -- not part of the pipeline.

Times SAM3 throughput on a real recording's HDF5 spectrogram, separating:
  - import + model load (one-time cost, paid once per sbatch task)
  - per-window inference, reported in rolling batches so warmup/JIT effects
    (if any) are visible instead of averaged away

Usage:
    python scripts/experimental/time_sam3_throughput.py <spec_dir> --channels 0 --prefix mic \
        --sam3-checkpoint sam3/sam3.pt --freq-min 500 --min-centroid-hz 0 --batch 100
"""
import argparse
import time

t_start = time.time()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vox_tracer.sam3_runner import build_processor, iter_sam3_windows, _h5_windows
from vox_tracer.spec import group_specs_by_channel
from glob import glob

t_imports_done = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("spec_dir")
ap.add_argument("--channels", default="0")
ap.add_argument("--prefix", default="mic")
ap.add_argument("--sam3-checkpoint", default=None)
ap.add_argument("--freq-min", type=float, default=500.0)
ap.add_argument("--min-centroid-hz", type=float, default=0.0)  # unused here, kept for CLI parity
ap.add_argument("--batch", type=int, default=100, help="print a rate every N windows")
args = ap.parse_args()

spec_dir = Path(args.spec_dir)
channels = [int(c) for c in args.channels.split(",")]
ch = channels[0]

by_ch = group_specs_by_channel(spec_dir, channels, prefix=args.prefix)
entries = by_ch.get(ch)
if not entries:
    h5_matches = sorted(glob(str(spec_dir / f"{args.prefix}_{ch}_*.h5")))
    entries = _h5_windows(Path(h5_matches[0]), chunk_sec=1.0)
print(f"[{time.time()-t_start:.1f}s] {len(entries)} windows discovered")

t_model0 = time.time()
processor = build_processor(args.sam3_checkpoint, 0.5)
t_model1 = time.time()
print(f"imports: {t_imports_done - t_start:.1f}s | model load: {t_model1 - t_model0:.1f}s")

t_loop0 = time.time()
t_batch0 = t_loop0
n_done = 0
n_with_box = 0
for win in iter_sam3_windows(processor, entries, cache_sato=False, chunk_sec=1.0,
                              sigmas=[2,3,4], threshold_pct=99.0, sample_rate=125000,
                              freq_min=args.freq_min, min_area=30, vert_aspect=5.0,
                              horiz_aspect=0.2, close_kernel=(7,3)):
    n_done += 1
    if win["best_box"] is not None:
        n_with_box += 1
    if n_done % args.batch == 0:
        now = time.time()
        rate = args.batch / (now - t_batch0)
        print(f"  windows {n_done-args.batch+1}-{n_done}: {now-t_batch0:.1f}s "
              f"({rate:.2f} windows/s, {1/rate:.3f} s/window)")
        t_batch0 = now

t_loop1 = time.time()
print(f"\nTOTAL: {n_done} windows in {t_loop1-t_loop0:.1f}s "
      f"({n_done/(t_loop1-t_loop0):.2f} windows/s steady avg, "
      f"{(t_loop1-t_loop0)/n_done:.3f} s/window)")
print(f"{n_with_box}/{n_done} windows had a stage-1 candidate")
print(f"grand total incl. imports+model load: {time.time()-t_start:.1f}s")
