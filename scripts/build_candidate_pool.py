"""Build and freeze the candidate pool for a labeling campaign.

Pools every detection from every detector and channel onto one time axis, merges
overlapping ones into candidates, and writes a CSV whose sha256 the campaign spec
pins. See vox_label/candidates.py for why this is frozen rather than recomputed.

Usage
-----
    python scripts/build_candidate_pool.py \
        --dataset gerbil_ssl \
        --detectors sam3_best,ridge,squeakout,das_yolo \
        --out outputs/label_campaigns/gerbil_ssl_k4/candidates.csv
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_label.candidates import (available_detectors, build_pool, find_clips,
                                  load_pool, pattern_histogram)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", default="outputs")
    ap.add_argument("--dataset", default="gerbil_ssl")
    ap.add_argument("--detectors", default="sam3_best,ridge,squeakout,das_yolo",
                    help="comma-separated, order is fixed into the pattern z")
    ap.add_argument("--channels", default=None,
                    help="comma-separated channel filter; default all channels found")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="link detections separated by less than this many seconds "
                         "(0 = merge only on genuine overlap)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true",
                    help="list the prediction variants available for this dataset and exit")
    args = ap.parse_args()

    avail = available_detectors(args.pred_root, args.dataset)
    if args.list:
        print(f"prediction variants for dataset {args.dataset!r} under {args.pred_root}/:")
        for d, n in sorted(avail.items(), key=lambda kv: -kv[1]):
            print(f"  {d:<32}{n:5d} clips")
        print("\nPass a comma-separated subset to --detectors. Their order is fixed into "
              "the pattern z and into every rule name, so keep it stable.")
        return
    if not args.out:
        ap.error("--out is required unless --list is given")

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    channels = None
    if args.channels:
        channels = {int(c) for c in args.channels.split(",")}

    missing = [d for d in detectors if d not in avail]
    if missing:
        ap.error(f"no predictions for {missing} on dataset {args.dataset!r}. "
                 f"Available: {sorted(avail)}")
    print(f"detectors (fixed order): {detectors}")
    print("per-detector clip coverage:")
    for d in detectors:
        print(f"  {d:<20}{avail[d]:5d} clips")
    shared = find_clips(args.pred_root, args.dataset, detectors)
    print(f"  {'intersection':<20}{len(shared):5d} clips used")
    dropped = {d: avail[d] - len(shared) for d in detectors if avail[d] > len(shared)}
    if dropped:
        # A clip one detector never ran on would contribute candidates whose z falsely
        # records that detector as silent, biasing its recall for a non-detector reason.
        print(f"  dropped (not covered by every detector): {dropped}")

    n, digest = build_pool(args.pred_root, args.dataset, detectors, args.out,
                           channels=channels, gap=args.gap, progress=100)
    print(f"\n{n} candidates -> {args.out}")
    print(f"sha256 {digest}")

    rows = load_pool(args.out, detectors)
    hist = pattern_histogram(rows)
    print(f"\npattern histogram ({len(hist)} of {2 ** len(detectors) - 1} possible):")
    for z, c in sorted(hist.items(), key=lambda kv: -kv[1]):
        fired = "+".join(d for d, b in zip(detectors, z) if b) or "(none)"
        print(f"  {''.join(map(str, z))}  {c:7d}  {c / n:6.1%}  {fired}")

    by_k = Counter(sum(r["z"]) for r in rows)
    print("\ncandidates by number of detectors firing:")
    for k in sorted(by_k):
        print(f"  {k}: {by_k[k]:7d}  {by_k[k] / n:6.1%}")

    print("\nf_V for single-detector rules (fraction of the pool each rule accepts):")
    for i, d in enumerate(detectors):
        f = sum(1 for r in rows if r["z"][i]) / n
        print(f"  {d:14s} f_V = {f:.4f}")

    durs = sorted(r["t_end"] - r["t_start"] for r in rows)
    def q(p):
        return durs[min(len(durs) - 1, int(p * len(durs)))]
    print(f"\nduration (s): p05={q(.05):.4f} median={q(.5):.4f} p95={q(.95):.4f} "
          f"max={durs[-1]:.4f}")
    print(f"clips: {len({r['clip'] for r in rows})}")


if __name__ == "__main__":
    main()
