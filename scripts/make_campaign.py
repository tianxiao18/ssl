"""Freeze a labeling campaign: choose the checkpoint grid, then commit to it.

This is the step where the annotation budget and the checkpoint grid are *chosen*, and
it is deliberately separate from labeling because the coverage guarantee depends on both
being fixed before any label exists. J enters the calibration, so checkpoints cannot be
appended later -- a campaign that reaches n_max without converging reports what it has.

Run with --plan-only first. That prints, for the rates you think are plausible, the
half-width each budget buys and the budget each tolerance costs (eq. 15), so the grid is
picked with numbers rather than by feel. Then re-run without it to write campaign.json.

    # what does a budget buy?
    python scripts/make_campaign.py --pool outputs/label_campaigns/x/candidates.csv \
        --detectors sam3_best,ridge,squeakout,das_yolo --plan-only

    # commit
    python scripts/make_campaign.py --pool outputs/label_campaigns/x/candidates.csv \
        --detectors sam3_best,ridge,squeakout,das_yolo \
        --out outputs/label_campaigns/x --n-max 6000 --checkpoints 12

    # label it
    python scripts/label_gui.py --annotator <your_name>

Three facts from section 5 that should shape the choice:

  Recall binds, not precision.   A precision near 0.9 has variance 0.09; a relative
      recall near 1/2 has 0.25. The binding target is whichever rate lies closest to
      1/2, not whichever detector performs worst.
  Corpus size is irrelevant.     N appears nowhere in eq. 15. A bigger corpus does not
      make benchmarking cheaper.
  Tolerance dominates.           The budget scales as 1/h^2, so loosening h is the only
      large lever. Dropping a detector saves almost nothing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_label import discover
from vox_label.setup import freeze
from vox_label.candidates import load_pool
from vox_label.exact_grid import c_of_j, even_grid, half_width, plan_n
from vox_label.rules import accept_fraction, default_rules, resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="candidates.csv from build_candidate_pool.py")
    ap.add_argument("--detectors", required=True, help="comma-separated, in pool order")
    ap.add_argument("--out", help="campaign directory to create (omit with --plan-only)")
    ap.add_argument("--dataset", default=None,
                    help="spectrogram dataset the pool's clips come from; detected from "
                         "the pool when only one matches")
    ap.add_argument("--spectrograms", default=None,
                    help="folder holding <dataset>/ spectrograms (PNG or HDF5); default "
                         "outputs/spectrograms[_h5] under wherever the GUI is started")
    ap.add_argument("--audio", default=None,
                    help="folder holding <dataset>/ wav files, for playback; default data/")
    ap.add_argument("--rules", default=None,
                    help="comma-separated rule names; default is each detector alone")
    ap.add_argument("--n-max", type=int, default=6000, help="annotation budget")
    ap.add_argument("--checkpoints", "-J", type=int, default=12,
                    help="number of checkpoints; more looks cost very little "
                         "(12 to 24 is +9%% annotations) and cannot be added later")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha-per-rule", action="store_true",
                    help="divide alpha across rules for a simultaneous claim (section 6)")
    ap.add_argument("--seed", type=int, default=20260911,
                    help="seeds the uniform annotation order; recorded in the spec")
    ap.add_argument("--recall-stream-max", type=int, default=None,
                    help="committed length of the recall-stream grids. Default n_max, "
                         "which is generous because P(x=1) is unknown before labeling")
    ap.add_argument("--p-real", type=float, default=None,
                    help="assumed P(x=1) for planning only; never written to the spec")
    ap.add_argument("--omega", type=float, default=0.5,
                    help="ranking dial, 0 recall to 1 precision; frozen with the spec")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="ranking resolution; frozen with the spec")
    ap.add_argument("--plan-only", action="store_true",
                    help="print what the grid buys and exit without writing anything")
    ap.add_argument("--tolerances", default="0.02,0.03,0.04,0.05",
                    help="half-widths to size in the planning table")
    args = ap.parse_args()

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    dataset = args.dataset
    if dataset is None and not args.plan_only:
        if args.spectrograms:
            root = args.spectrograms
            names = [p.name for p in Path(root).iterdir() if p.is_dir()]
            matches = discover.pool_datasets(args.pool, names, root, root)
        else:
            root = discover.PNG_ROOT
            matches = discover.pool_datasets(args.pool,
                                             [d["name"] for d in discover.datasets()])
        if len(matches) != 1:
            ap.error(f"pass --dataset: the pool's clips match {matches or 'no dataset'} "
                     f"under {Path(root).resolve()}")
        dataset = matches[0]
    rows = load_pool(args.pool, detectors)
    rules = ([r.strip() for r in args.rules.split(",")] if args.rules
             else default_rules(detectors))
    J, n_max, alpha = args.checkpoints, args.n_max, args.alpha
    eff_alpha = alpha / len(rules) if args.alpha_per_rule else alpha

    print(f"pool {args.pool}")
    print(f"  {len(rows)} candidates, detectors {detectors}")
    print(f"grid: n_max={n_max}, J={J}, alpha={alpha:g}"
          f"{f' -> {eff_alpha:.4g} per rule' if args.alpha_per_rule else ''}")
    print(f"  checkpoints: {even_grid(n_max, J)}")
    print(f"  boundary constant c(J)={c_of_j(J):.3f}  (Bonferroni would need more)")

    print(f"\nprecision streams -- f_V is exact, known before any label:")
    print(f"  {'rule':<14}{'f_V':>7}{'stream at n_max':>17}{'+/-pts (worst case)':>21}")
    for r in rules:
        f = accept_fraction(resolve(r, detectors), rows)
        print(f"  {r:<14}{f:>7.3f}{int(f * n_max):>17}"
              f"{100 * half_width(0.25, f, n_max, J):>21.2f}")

    print(f"\nrecall streams -- length is n_max * P(x=1), unknown until labels exist:")
    print(f"  {'P(x=1)':>8}{'stream at n_max':>17}{'+/-pts':>10}")
    for px in ([args.p_real] if args.p_real else [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]):
        print(f"  {px:>8.2f}{int(px * n_max):>17}{100 * half_width(0.25, px, n_max, J):>10.2f}")

    print(f"\nbudget needed per tolerance (eq. 15, worst-case variance 0.25):")
    tols = [float(t) for t in args.tolerances.split(",")]
    header = "".join(f"{f'+/-{100 * t:g}pts':>13}" for t in tols)
    print(f"  {'stream f':<10}{header}")
    for f in sorted({round(accept_fraction(resolve(r, detectors), rows), 3) for r in rules}
                    | {args.p_real} - {None}):
        cells = "".join(f"{plan_n(0.25, f, t, J):>13.0f}" for t in tols)
        print(f"  {f:<10.3f}{cells}")

    if args.plan_only:
        print("\n--plan-only: nothing written. Re-run with --out to freeze the campaign.")
        return
    if not args.out:
        ap.error("--out is required unless --plan-only is given")

    path, spec = freeze(args.out, args.pool, detectors, dataset, rules=rules, alpha=alpha,
                        n_max=n_max, J=J, seed=args.seed,
                        recall_stream_max=args.recall_stream_max,
                        alpha_per_rule=args.alpha_per_rule,
                        omega=args.omega, delta=args.delta,
                        spectrogram_root=args.spectrograms, audio_root=args.audio)
    print(f"\nfrozen -> {path}  (dataset {dataset})")
    print("This spec is now read-only. To change the grid, the rules, or the pool, "
          "build a new campaign directory.")


if __name__ == "__main__":
    main()
