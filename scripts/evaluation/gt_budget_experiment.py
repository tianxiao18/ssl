"""
How much ground-truth (gt_csv) do you actually need?

Two questions, one bootstrap:
  1. Stability   — as a function of the number of labeled GT vocalizations N, how
                   wide is the uncertainty on precision/recall for a single method?
                   F1 is intentionally left out of this part: confidence-sequences-note.pdf
                   only gives an exact closed-form F1 CI (eq. 8-9) when precision and
                   recall share one candidate stream, which isn't true here (they come
                   from two different matched-event pools) — see vox_tracer.scoring.
  2. Comparison  — as a function of N, how reliably would you conclude "method A
                   beats method B" for each pair of methods (bootstrap CI on the
                   paired F1 difference excludes zero, with the correct sign)? F1
                   stays here since this is a real paired bootstrap, not a closed form.

Method
------
Recordings (sessions) are the actual annotation/resampling unit — an annotator
labels a whole recording, not individual calls, and recordings carry a variable
number of GT vocalizations each (0-5 here). Starting from the FULL per-recording
metrics.csv already produced by evaluate.py (one row per session with
tp/pred_tp/fp/fn/n_gt), we treat that full set of recordings as the population
and ask: "if I had only annotated enough recordings to cover N vocalizations,
how much would my estimate wobble?" This is answered by the standard
nonparametric bootstrap: for each target vocalization budget N, draw many
samples of recordings *with replacement* (sized to average out to N
vocalizations), recompute the micro-averaged metric on each draw, and look at
the spread of the resulting distribution. All methods share the exact same
per-trial resampling draw, so every pairwise difference is a paired comparison.

This does NOT tell you the metric's true value more precisely than the full
dataset already does — it tells you how much annotation budget you'd need to
pin that value down (or detect a method difference) if you were starting from
scratch.

Usage
-----
    python scripts/evaluation/gt_budget_experiment.py outputs/eval/gerbil_ssl \\
        --methods sam3_best,ridge,squeakout \\
        --metrics metrics_sampled_contiguous_detections.csv \\
        --out outputs/eval/gt_budget_experiment.png
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import beta

# repo's validated categorical palette (references/palette.md), fixed order — one
# color per method, and the same order reused for the pairwise-diff lines.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR_REF  = "#8a8a86"   # neutral gray — full-dataset reference line
COLOR_ZERO = "#52514e"   # text-secondary gray — zero line

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("eval_base", help="dir holding <method>/<metrics-csv>, e.g. outputs/eval/gerbil_ssl")
parser.add_argument("--methods", default="sam3_best,ridge,squeakout",
                    help="comma list of method dir names; stability is computed for each, "
                         "pairwise comparison for every combination of 2")
parser.add_argument("--metrics", default="metrics_sampled_contiguous_detections.csv",
                    help="metrics CSV filename written by evaluate.py inside each method dir")
parser.add_argument("--sample-sizes", default=None,
                    help="comma list of N (GT vocalizations) to test; default: 12 linearly "
                         "spaced points from 20 up to the full vocalization count")
parser.add_argument("--n-trials", type=int, default=3000,
                    help="bootstrap replicates drawn per sample size")
parser.add_argument("--ci", type=float, default=0.95, help="confidence level for the bootstrap interval")
parser.add_argument("--ci-method", choices=["bootstrap", "clopper-pearson", "confidence-sequence"],
                    default="bootstrap",
                    help="how to get the per-N precision/recall CI in the stability panels. "
                         "'bootstrap' (default): unchanged percentile CI from the recording resampling "
                         "trials. 'clopper-pearson': exact closed-form binomial CI (no resampling), "
                         "valid ONLY at one N fixed in advance — see confidence-sequences-note.pdf "
                         "sec. 3.1. 'confidence-sequence': exact anytime-valid interval (Krichevsky-"
                         "Trofimov forecaster, sec. 3.2) that stays valid simultaneously at every N, so "
                         "it's safe to peek and stop annotating early — costs roughly 4x the annotations "
                         "of clopper-pearson for the same width (sec. 3.3). The pairwise F1-diff "
                         "comparison always stays on the paired bootstrap — neither closed-form method "
                         "has an analog for a correlated difference of proportions, or for F1 itself "
                         "(no single shared candidate stream here to invert eq. 8-9 exactly).")
parser.add_argument("--pr-tol", type=float, default=0.05,
                    help="report/mark the smallest tested N whose precision AND recall CI "
                         "half-width both drop below this")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs/eval/gt_budget_experiment.png")
parser.add_argument("--out-csv", default=None,
                    help="per-N summary CSV; default: --out with .csv extension")
args = parser.parse_args()

methods   = [m.strip() for m in args.methods.split(",")]
eval_base = Path(args.eval_base)
if len(methods) < 1:
    raise SystemExit("--methods needs at least 1 name")
if len(methods) > len(PALETTE):
    raise SystemExit(f"--methods: at most {len(PALETTE)} methods supported")
pairs = list(combinations(methods, 2))


# ── load per-recording counts, aligned by session across methods ──────────────

def load_rows(method):
    path = eval_base / method / args.metrics
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["session"]] = {
                "tp": int(row["tp"]), "pred_tp": int(row["pred_tp"]),
                "fp": int(row["fp"]), "fn": int(row["fn"]), "n_gt": int(row["n_gt"]),
            }
    return rows


per_method_rows = {m: load_rows(m) for m in methods}
sessions = sorted(set.intersection(*(set(r) for r in per_method_rows.values())))
n_total = len(sessions)
if n_total < 10:
    raise SystemExit(f"only {n_total} recordings common to all methods — too few to bootstrap")
n_gt_total = sum(per_method_rows[methods[0]][s]["n_gt"] for s in sessions)
print(f"{n_total} recordings common to {methods}, {n_gt_total} total GT vocalizations "
      f"({n_gt_total / n_total:.1f}/recording)")

counts = {}  # method -> dict of (n_total,) arrays
for m in methods:
    rows = per_method_rows[m]
    counts[m] = {k: np.array([rows[s][k] for s in sessions], dtype=np.int64)
                 for k in ("tp", "pred_tp", "fp", "fn")}
ngt_arr = np.array([per_method_rows[methods[0]][s]["n_gt"] for s in sessions], dtype=np.int64)
vox_per_recording = n_gt_total / n_total  # mean, used only to size the recording draw


def micro_metrics(tp, pred_tp, fp, fn):
    """Vectorized micro precision/recall/F1 over already-summed counts."""
    rec  = tp / (tp + fn)
    prec = pred_tp / (pred_tp + fp)
    f1   = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def clopper_pearson(x, n, alpha):
    """Exact binomial CI for x successes out of n trials (Clopper & Pearson, 1934).
    Valid only at this one n, fixed in advance — sec. 3.1 of confidence-sequences-note.pdf.
    """
    lo = 0.0 if x <= 0 else beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x >= n else beta.ppf(1 - alpha / 2, x + 1, n - x)
    return lo, hi


def _kt_regret(x, t):
    """Exact regret of the Krichevsky-Trofimov forecaster after t rounds with x
    successes (eq. 13 & 16): hindsight log-likelihood minus the KT forecaster's."""
    if x <= 0 or x >= t:
        hindsight_ll = 0.0  # 0*log(0) := 0
    else:
        p_hat = x / t
        hindsight_ll = x * np.log(p_hat) + (t - x) * np.log(1 - p_hat)
    forecaster_ll = gammaln(x + 0.5) + gammaln(t - x + 0.5) - gammaln(t + 1) - np.log(np.pi)
    return hindsight_ll - forecaster_ll


def _kl_bernoulli(u, v):
    """KL(Bernoulli(u) || Bernoulli(v)), with the 0*log(0) := 0 convention."""
    v = min(max(v, 1e-12), 1 - 1e-12)
    t1 = 0.0 if u <= 0 else u * np.log(u / v)
    t2 = 0.0 if u >= 1 else (1 - u) * np.log((1 - u) / (1 - v))
    return t1 + t2


def confidence_sequence_interval(x, t, alpha):
    """Exact anytime-valid CI for a Bernoulli rate from x successes in t i.i.d. trials,
    via the KT-forecaster confidence sequence (eq. 12-19, sec. 3.2). Unlike
    clopper_pearson, this stays valid simultaneously at every t — safe to recompute as
    more annotations come in and stop as soon as it's tight enough.
    """
    if t <= 0:
        return 0.0, 1.0
    p_hat = x / t
    thresh = (np.log(1 / alpha) + _kt_regret(x, t)) / t

    def f(p):
        return _kl_bernoulli(p_hat, p) - thresh

    lo = 0.0 if x <= 0 else brentq(f, 1e-12, p_hat)
    hi = 1.0 if x >= t else brentq(f, p_hat, 1 - 1e-12)
    return lo, hi


# full-dataset (all N_total recordings, no resampling) point estimate — used as the
# plug-in rate for the closed-form --ci-methods, and reported as the reference line either way.
full_f1, full_prec, full_rec = {}, {}, {}
for m in methods:
    c = counts[m]
    prec, rec, f1 = micro_metrics(c["tp"].sum(), c["pred_tp"].sum(), c["fp"].sum(), c["fn"].sum())
    full_f1[m], full_prec[m], full_rec[m] = f1, prec, rec


def stream_metrics_for_N(m, N, interval_fn):
    """Closed-form CI for method m's precision/recall at GT budget N, using
    interval_fn(x, n, alpha) -> (lo, hi) for the underlying binomial CI (either
    clopper_pearson or confidence_sequence_interval — same stream construction either way).

    Recall's stream is exactly N (annotated GT vocalizations); precision's stream
    length is the full dataset's accepted-candidate rate scaled by N/n_gt_total, since
    the two streams here come from different matched-event pools (sec. 3.1/2.2 of
    confidence-sequences-note.pdf). F1 is left out — see module docstring.
    """
    c = counts[m]
    m_stream = max(1, round(int(c["pred_tp"].sum() + c["fp"].sum()) * N / n_gt_total))
    prec_lo, prec_hi = interval_fn(round(full_prec[m] * m_stream), m_stream, alpha)
    rec_lo, rec_hi = interval_fn(round(full_rec[m] * N), N, alpha)
    return {"prec_med": full_prec[m], "prec_lo": prec_lo, "prec_hi": prec_hi,
            "rec_med": full_rec[m], "rec_lo": rec_lo, "rec_hi": rec_hi}


def cs_sizing_N(sigma2, h, alpha, J=1, n_iters=8):
    """Confidence-sequence sizing formula (eq. 21, sec. 4.1): trials needed for a
    KT-forecaster confidence sequence's half-width to reach h, for a Bernoulli(p)
    stream with variance sigma2 = p(1-p). n appears on both sides (via the KT
    forecaster's log(t) regret term), so this is a fixed-point iteration — the note
    reports 3 iterations suffice; we use a few extra for margin. Unlike
    stream_metrics_for_N/first_stable_n, this needs no per-N sweep at all: one call
    gives the answer directly. Stripped of the note's `f` annotation-pool scaling
    factor since each stream's own trial count is already the quantity being sized
    here (see stream_metrics_for_N for how that maps back to GT-vocalization units).
    """
    n = max(2 * sigma2 / h ** 2, 10.0)  # seed: first guess ignoring the log(n) term
    for _ in range(n_iters):
        n = (2 * sigma2 / h ** 2) * (np.log(J / alpha) + 0.5 * np.log(max(n, 1.0)) + 0.226)
    return n


# ── sample sizes ────────────────────────────────────────────────────────────
# Targets are expressed in GT vocalizations (the unit the user cares about), then
# converted to a recording-draw size via the dataset's mean vocalizations/recording
# — recordings are what's actually resampled, since GT events within one recording
# aren't independent draws. The achieved vocalization count per draw is random
# (recordings carry 0-5 GT events each), so the true x-position plotted for each
# target is the median vocalization count actually realized across bootstrap trials.

if args.sample_sizes:
    vox_targets = sorted(int(x) for x in args.sample_sizes.split(","))
else:
    vox_targets = sorted(set(np.linspace(20, n_gt_total, 12).round().astype(int)))

rec_draw_sizes = sorted(set(
    int(np.clip(round(target / vox_per_recording), 1, n_total)) for target in vox_targets
))

alpha = 1 - args.ci
lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

rng = np.random.default_rng(args.seed)


# ── bootstrap ────────────────────────────────────────────────────────────────
# Same resampling draw shared across all methods at each (N, trial) so every
# pairwise diff is a true paired bootstrap, not independent draws.

summary = []  # one dict per sample size, for the CSV
metric_names = ("prec", "rec")  # F1 excluded from stability tracking — see module docstring
curves = {m: {f"{k}_{s}": [] for k in metric_names for s in ("med", "lo", "hi")} for m in methods}

for n_rec in rec_draw_sizes:
    idx = rng.integers(0, n_total, size=(args.n_trials, n_rec))
    N = int(np.median(ngt_arr[idx].sum(axis=1)))  # realized GT-vocalization count for this draw size

    f1_by_method = {}
    for m in methods:
        c = counts[m]
        tp_s      = c["tp"][idx].sum(axis=1)
        pred_tp_s = c["pred_tp"][idx].sum(axis=1)
        fp_s      = c["fp"][idx].sum(axis=1)
        fn_s      = c["fn"][idx].sum(axis=1)
        prec, rec, f1 = micro_metrics(tp_s, pred_tp_s, fp_s, fn_s)
        f1_by_method[m] = f1  # kept only for the pairwise comparison below (real paired bootstrap)
        if args.ci_method == "bootstrap":
            for k, vals in (("prec", prec), ("rec", rec)):
                med, lo, hi = np.nanmedian(vals), np.nanpercentile(vals, lo_pct), np.nanpercentile(vals, hi_pct)
                curves[m][f"{k}_med"].append(med)
                curves[m][f"{k}_lo"].append(lo)
                curves[m][f"{k}_hi"].append(hi)
        else:  # closed-form: no trial distribution needed
            interval_fn = clopper_pearson if args.ci_method == "clopper-pearson" else confidence_sequence_interval
            cf = stream_metrics_for_N(m, N, interval_fn)
            for k in metric_names:
                curves[m][f"{k}_med"].append(cf[f"{k}_med"])
                curves[m][f"{k}_lo"].append(cf[f"{k}_lo"])
                curves[m][f"{k}_hi"].append(cf[f"{k}_hi"])

    row = {"N": N, "n_recordings": n_rec}
    for m in methods:
        for k in metric_names:
            row[f"{m}_{k}_med"] = curves[m][f"{k}_med"][-1]
            row[f"{m}_{k}_lo"]  = curves[m][f"{k}_lo"][-1]
            row[f"{m}_{k}_hi"]  = curves[m][f"{k}_hi"][-1]

    for (ma, mb) in pairs:
        d = f1_by_method[ma] - f1_by_method[mb]
        dmed, dlo, dhi = np.nanmedian(d), np.nanpercentile(d, lo_pct), np.nanpercentile(d, hi_pct)
        key = f"diff_{ma}_vs_{mb}"
        row[f"{key}_med"] = dmed; row[f"{key}_lo"] = dlo; row[f"{key}_hi"] = dhi
        row[f"{key}_excludes_zero"] = bool(dlo > 0 or dhi < 0)

    summary.append(row)


# ── stability thresholds (shared by the text report and the plot's vertical lines) ──

def half_width(row, m, k):
    return (row[f"{m}_{k}_hi"] - row[f"{m}_{k}_lo"]) / 2


def first_stable_n(m, keys, tol):
    return next((r["N"] for r in summary if all(half_width(r, m, k) <= tol for k in keys)), None)


pr_stable_n = {m: first_stable_n(m, ("prec", "rec"), args.pr_tol) for m in methods}


# ── text report ──────────────────────────────────────────────────────────────

print(f"\nFull-dataset F1 ({n_total} recordings): " +
      ", ".join(f"{m}={full_f1[m]:.3f}" for m in methods))
print(f"CI method: {args.ci_method}"
      + (" (per-N stability panels; pairwise diff always uses the paired bootstrap)"
         if args.ci_method != "bootstrap" else ""))

print("\nStability (precision AND recall {:.0%} CI half-width <= {}):".format(args.ci, args.pr_tol))
for m in methods:
    msg = f"N={pr_stable_n[m]}" if pr_stable_n[m] else f"not reached within tested range (up to N={summary[-1]['N']})"
    print(f"  {m}: {msg}")

print(f"\nClosed-form confidence-sequence sizing (eq. 21, no sweep needed) for "
      f"precision AND recall {args.pr_tol:.0%} half-width:")
for m in methods:
    accepted_total = int(counts[m]["pred_tp"].sum() + counts[m]["fp"].sum())
    n_prec_trials = cs_sizing_N(full_prec[m] * (1 - full_prec[m]), args.pr_tol, alpha)
    N_prec = n_prec_trials * n_gt_total / accepted_total
    N_rec = cs_sizing_N(full_rec[m] * (1 - full_rec[m]), args.pr_tol, alpha)
    N_needed = max(N_prec, N_rec)
    line = f"  {m}: precision needs N~{N_prec:.0f}, recall needs N~{N_rec:.0f} -> both need N~{N_needed:.0f}"
    if args.ci_method == "confidence-sequence" and pr_stable_n[m]:
        line += f"  (swept curve found N={pr_stable_n[m]}, ratio={N_needed / pr_stable_n[m]:.2f})"
    print(line)

if pairs:
    print(f"\nPairwise comparison ({args.ci:.0%} bootstrap CI on F1 diff excludes zero, correct sign):")
    for (ma, mb) in pairs:
        true_diff = full_f1[ma] - full_f1[mb]
        true_sign = np.sign(true_diff)
        key = f"diff_{ma}_vs_{mb}"
        n_confident = next((r["N"] for r in summary
                            if r[f"{key}_excludes_zero"] and np.sign(r[f"{key}_med"]) == true_sign), None)
        msg = (f"N={n_confident}" if n_confident
               else f"not reached within tested range (up to N={summary[-1]['N']})")
        print(f"  {ma} vs {mb} (full-data diff {true_diff:+.3f}): {msg}")


# ── CSV ──────────────────────────────────────────────────────────────────────

out_csv = Path(args.out_csv) if args.out_csv else Path(args.out).with_suffix(".csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print(f"\nper-N summary -> {out_csv}")


# ── plot ─────────────────────────────────────────────────────────────────────
# One panel per metric (precision, recall — F1 excluded, see module docstring):
# median + CI band per method, a dotted horizontal line at the full-dataset value,
# and a dashed vertical line at the budget where that method's CI half-width first
# drops inside tolerance — the tolerance itself is named in the legend, not the title.

ns = [r["N"] for r in summary]
fig, (ax_prec, ax_rec) = plt.subplots(1, 2, figsize=(12, 4.5))


def plot_metric(ax, key, stable_n, title, tol):
    for m, color in zip(methods, PALETTE):
        ax.plot(ns, curves[m][f"{key}_med"], color=color, lw=2, marker="o", ms=4, label=m)
        ax.fill_between(ns, curves[m][f"{key}_lo"], curves[m][f"{key}_hi"], color=color, alpha=0.15, linewidth=0)
        ax.axhline({"prec": full_prec, "rec": full_rec}[key][m],
                  color=color, lw=1, ls=":", alpha=0.6)
        if stable_n[m]:
            ax.axvline(stable_n[m], color=color, lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel("GT vocalizations labeled (N)")
    ax.set_ylabel(title)
    ax.set_title(f"{title} stability vs. GT budget", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=COLOR_ZERO, lw=1.2, ls="--"))
    labels.append(f"stable within ±{tol:.0%}")
    ax.legend(handles, labels, fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plot_metric(ax_prec, "prec", pr_stable_n, "Precision", args.pr_tol)
plot_metric(ax_rec,  "rec",  pr_stable_n, "Recall",    args.pr_tol)

fig.tight_layout()
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"plot -> {out_path}")
