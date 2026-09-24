"""One-off: partition dryad_gerbil_full's recordings into N balanced shards for a
Slurm job array (see scripts/sbatch/run_sam3_dryad_full.sbatch).

Balances by actual window count (read from each HDF5's 't' dataset shape --
cheap, just a header read, not the array itself) via longest-processing-time-
first greedy bin-packing, since cohort2 (~1200 windows/recording) and cohort4
(~3600 windows/recording) have very different per-recording cost.

Writes outputs/sam3_shards/shard_NN.txt, one recording's relative path per line.
"""
import argparse
from pathlib import Path

import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--spec-base", default="outputs/spectrograms_h5/dryad_gerbil_full")
ap.add_argument("--prefix", default="mic")
ap.add_argument("--channel", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=16)
ap.add_argument("--chunk-sec", type=float, default=1.0)
ap.add_argument("--out-dir", default="outputs/sam3_shards")
args = ap.parse_args()

base = Path(args.spec_base)
h5_paths = sorted(base.glob(f"**/{args.prefix}_{args.channel}_*.h5"))
print(f"discovered {len(h5_paths)} recordings under {base}")

weighted = []
for h5_path in h5_paths:
    rel = h5_path.parent.relative_to(base)
    with h5py.File(h5_path, "r") as f:
        duration = float(f["t"][-1])
    n_windows = int(duration // args.chunk_sec)  # 1-sec windows, not raw STFT time-bins
    weighted.append((n_windows, str(rel)))

weighted.sort(reverse=True)  # LPT: heaviest first

shards = [[] for _ in range(args.n_shards)]
shard_weights = [0] * args.n_shards
for n_windows, rel in weighted:
    i = min(range(args.n_shards), key=lambda i: shard_weights[i])
    shards[i].append(rel)
    shard_weights[i] += n_windows

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
for i, recs in enumerate(shards):
    path = out_dir / f"shard_{i:02d}.txt"
    path.write_text("\n".join(recs) + "\n")
    print(f"  shard {i:02d}: {len(recs):4d} recordings, {shard_weights[i]:8d} windows -> {path}")

print(f"\ntotal: {sum(shard_weights):,} windows across {sum(len(s) for s in shards)} recordings, "
      f"{args.n_shards} shards")
print(f"max/min shard weight: {max(shard_weights):,} / {min(shard_weights):,} "
      f"(imbalance: {(max(shard_weights)/min(shard_weights)-1)*100:.1f}%)")
