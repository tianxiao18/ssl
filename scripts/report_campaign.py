"""Report a labeling campaign: exact intervals for every rule, at committed checkpoints.

Each interval is read at the last checkpoint its own stream actually crossed, which is
the only place the coverage guarantee applies. Read individually, every interval here
is valid at alpha as it stands; a simultaneous claim across rules wants the campaign
built with --alpha-per-rule (section 6).

    python scripts/report_campaign.py --campaign outputs/label_campaigns/gerbil_ssl_k4

Also writes a labels CSV in the schema of data/gerbil_ssl/sampled_contiguous_detections.csv
so the existing consumers (evaluate.py --gt-csv, pr_curves.py) can read the output.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_label.campaign import Campaign
from vox_label.render import make_source  # noqa: F401  (kept for symmetry of imports)
from vox_label.server import LabelService

# repo's validated categorical palette (references/palette.md), same order as
# scripts/evaluation/gt_budget_experiment.py so a method keeps its colour across figures.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TARGETS = ["precision", "recall", "f1"]
TITLES = {"precision": "precision", "recall": "relative recall", "f1": "F1"}


class _NoSource:
    def crop(self, *a):
        raise RuntimeError("reporting does not render crops")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", default=None, help="figure path (default <campaign>/report.png)")
    ap.add_argument("--csv", default=None, help="labels CSV (default <campaign>/labels.csv)")
    args = ap.parse_args()

    c = Campaign(args.campaign)
    c.verify_pool()
    svc = LabelService(c, _NoSource())
    rows = svc.compute_report()
    n = len(svc.labels)

    print(f"campaign {c.spec['name']}   {n} annotations   alpha={c.alpha:g}"
          f"{'  (per-rule)' if c.spec['alpha_per_rule'] else ''}")
    print(f"{'rule':<13}{'target':<16}{'est':>8}{'lo':>9}{'hi':>9}{'+/-pts':>9}"
          f"{'checkpoint':>14}{'stream':>8}")
    for r in rows:
        if not r["available"]:
            print(f"{r['rule']:<13}{TITLES[r['target']]:<16}{'--':>8}"
                  f"{'not yet: stream ' + str(r['stream_len']) + ' < first checkpoint ' + str(r['next_checkpoint']):>49}")
            continue
        print(f"{r['rule']:<13}{TITLES[r['target']]:<16}{r['estimate']:>8.3f}"
              f"{r['lo']:>9.3f}{r['hi']:>9.3f}{100 * r['half_width']:>9.2f}"
              f"{str(r['checkpoint_index']) + '/' + str(r['n_checkpoints']) + ' @ ' + str(r['checkpoint']):>14}"
              f"{r['stream_len']:>8}")

    _plot(c, rows, n, args.out or Path(args.campaign) / "report.png")
    _labels_csv(c, svc, args.csv or Path(args.campaign) / "labels.csv")


def _plot(c, rows, n, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(c.rules) + 2.6), sharey=True)
    ypos = list(range(len(c.rules)))[::-1]
    for ax, target in zip(axes, TARGETS):
        for y, rule in zip(ypos, c.rules):
            r = next(x for x in rows if x["rule"] == rule and x["target"] == target)
            color = PALETTE[c.rules.index(rule) % len(PALETTE)]
            if not r["available"]:
                ax.text(0.5, y, "no checkpoint reached", ha="center", va="center",
                        fontsize=8, color="#8a8a86")
                continue
            ax.plot([r["lo"], r["hi"]], [y, y], color=color, lw=3.5, alpha=.45,
                    solid_capstyle="butt")
            ax.plot([r["estimate"]], [y], "o", color=color, ms=6)
        ax.set_yticks(ypos, c.rules, fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#dddddd", lw=.6)
        ax.set_axisbelow(True)
        # Per the project's plotting preference: short per-panel titles, no suptitle.
        ax.set_title(TITLES[target], fontsize=11)
    axes[0].set_ylabel(f"{n} annotations, alpha={c.alpha:g}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nfigure -> {out}")


def _labels_csv(c, svc, out):
    """Emit labels in the sampled_contiguous_detections.csv schema."""
    dets = c.detectors
    fields = (["experiment", "idx", "clip", "start_seconds", "stop_seconds",
               "duration_seconds", "methods", "n_methods"]
              + [f"has_{d}" for d in dets] + ["is_vocalization", "combined", "annotator"])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for cid in c.order:
            rec = svc.labels.get(cid)
            if rec is None:
                continue
            row = c.pool[cid]
            parts = row["clip"].split("/")
            fired = [d for d, b in zip(dets, row["z"]) if b]
            w.writerow({
                "experiment": parts[0].replace("experiment_", "") if parts else "",
                "idx": parts[1].replace("idx_", "") if len(parts) > 1 else "",
                "clip": row["clip"],
                "start_seconds": row["t_start"], "stop_seconds": row["t_end"],
                "duration_seconds": row["t_end"] - row["t_start"],
                "methods": "+".join(fired), "n_methods": len(fired),
                **{f"has_{d}": bool(b) for d, b in zip(dets, row["z"])},
                "is_vocalization": "yes" if rec["label"] else "no",
                "combined": rec.get("combined", False),
                "annotator": rec.get("annotator", ""),
            })
    print(f"labels -> {out}")


if __name__ == "__main__":
    main()
