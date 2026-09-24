"""Monte-Carlo calibration check for the error bars the label GUI prints.

Does the promised 95% coverage actually hold? Four checks at alpha = 0.05, on the
frozen gerbil_ssl_k4 grids:

1. exact grid, simultaneous coverage vs the true rate p -- computed exactly with the
   survival recursion *and* by MC, using the same GammaCache the server uses.
2. per-checkpoint coverage P(p in C_j) across the grid, exact and MC.
3. anytime CS (the live panel): cumulative miscoverage P(exists s <= t: p not in CS_s).
3b. --gamma-scan N: coverage at N random p under the server's rounded gamma against
   two candidate fixes, which is where the one real defect shows up.
4. end-to-end through vox_label.streams + counts_at + exact_grid_interval on a
   simulated labeling run with 4 detectors and 7 rules -- precision, relative recall
   and F1, plus the family-wise rate over all 21 bars at once.

Usage
-----
    python scripts/experimental/ci_calibration_sim.py \
        --out outputs/eval/ci_calibration.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from vox_label.anytime import cs_interval
from vox_label.exact_grid import (GammaCache, calibrate_gamma, cp_acceptance,
                                  even_grid, exact_grid_interval, f1_from_jaccard,
                                  in_confidence_set, survival, survival_with_regions)
from vox_label.streams import STREAM_KINDS, counts_at

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
COLOR_REF = "#8a8a86"
ALPHA = 0.05
# The real sam3_best|precision grid from outputs/label_campaigns/gerbil_ssl_k4.
GRID = [189, 378, 567, 756, 945, 1134, 1324, 1513, 1702, 1891, 2080, 2269]


def wilson(k, n, z=1.96):
    """Interval on an MC coverage estimate, so a dip below 0.95 can be read as real."""
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ── 1-2. the committed grid ────────────────────────────────────────────────────

def paths(p, grid, m, rng):
    """m simulated cumulative-success paths observed at the grid checkpoints."""
    blocks = np.diff(np.concatenate([[0], grid]))
    return np.cumsum(rng.binomial(blocks, p, size=(m, len(grid))), axis=1)


def grid_coverage(p, grid, gamma, m, rng):
    """(exact, MC) coverage of p at each checkpoint prefix, given the level gamma."""
    regions = [cp_acceptance(n, p, gamma) for n in grid]
    exact = [survival_with_regions(p, grid[:j + 1], regions[:j + 1])
             for j in range(len(grid))]
    c = paths(p, grid, m, rng)
    inside = np.ones(m, dtype=bool)
    mc = []
    for j, (lo, hi) in enumerate(regions):
        inside &= (c[:, j] >= lo) & (c[:, j] <= hi)
        mc.append(inside.mean())
    return np.array(exact), np.array(mc)


def leg_grid(p_values, m, rng, verbose=True):
    """Coverage vs p, with the server's rounded gamma and with the ideal one."""
    cache = GammaCache(GRID, ALPHA)   # exactly what server.py:gamma_cache builds
    rows = []
    for p in p_values:
        g_srv = cache(p)
        g_ideal = calibrate_gamma(round(float(p), 6), GRID, ALPHA)
        ex_srv, mc_srv = grid_coverage(p, GRID, g_srv, m, rng)
        ex_ideal = grid_coverage(p, GRID, g_ideal, m, rng)[0]
        rows.append({"p": p, "gamma_server": g_srv, "gamma_ideal": g_ideal,
                     "exact_server": ex_srv, "exact_ideal": ex_ideal, "mc": mc_srv})
        if verbose:
            print(f"  p={p:.4f}  gamma={g_srv:.5f}  exact={ex_srv[-1]:.4f}  "
                  f"mc={mc_srv[-1]:.4f}", flush=True)
    return rows


def check_inversion(p, m, rng, cache):
    """The printed lo/hi must agree with the membership test the coverage uses."""
    c = paths(p, GRID, m, rng)
    n_cover, n_agree = 0, 0
    for row in c:
        counts = row.tolist()
        lo, hi = exact_grid_interval(counts, GRID, len(GRID) - 1, ALPHA, gamma_of_p=cache)
        covers = lo <= p <= hi
        n_cover += covers
        n_agree += covers == in_confidence_set(p, counts, GRID, len(GRID) - 1, cache)
    return n_cover / m, n_agree / m


def gamma_scan(n, rng):
    """Worst-case coverage of GammaCache's rounded gamma, against two fixes.

    Rounding p before calibrating can land on a *larger* gamma than p's own, which
    shrinks every acceptance region and costs coverage -- the cache's docstring only
    accounts for rounding the other way. Fix A takes the smaller gamma of the two
    lattice points bracketing p; fix B shrinks the cached gamma by 2%.
    """
    cache = GammaCache(GRID, ALPHA)
    step = 10.0 ** -cache.decimals
    out = []
    for p in rng.uniform(0.01, 0.99, n):
        g = cache(p)
        lo = np.floor(p / step) * step
        bracket = min(cache(lo), cache(lo + step))
        out.append((p, survival(p, GRID, g), survival(p, GRID, bracket),
                    survival(p, GRID, 0.98 * g)))
        print(f"  p={p:.6f} server={out[-1][1]:.5f} bracket={out[-1][2]:.5f} "
              f"margin={out[-1][3]:.5f}", flush=True)
    a = np.array(out)
    for i, name in ((1, "server"), (2, "bracket"), (3, "margin 2%")):
        print(f"  {name:10s} worst {a[:, i].min():.5f}  below nominal "
              f"{(a[:, i] < 1 - ALPHA).sum()}/{n}")
    return a


# ── 3. the live panel's confidence sequence ────────────────────────────────────

def cs_covers(p, x, t, alpha):
    """Vectorized 'p in CS_t', identical to anytime.cs_interval's boundary."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    ph = x / t
    with np.errstate(divide="ignore", invalid="ignore"):
        hind = np.where((x > 0) & (x < t),
                        x * np.log(np.clip(ph, 1e-300, None))
                        + (t - x) * np.log(np.clip(1 - ph, 1e-300, None)), 0.0)
        kl = np.where(ph > 0, ph * np.log(np.clip(ph, 1e-300, None) / p), 0.0) + \
             np.where(ph < 1, (1 - ph) * np.log(np.clip(1 - ph, 1e-300, None) / (1 - p)), 0.0)
    fore = gammaln(x + 0.5) + gammaln(t - x + 0.5) - gammaln(t + 1) - np.log(np.pi)
    return t * kl <= np.log(1 / alpha) + (hind - fore)


def leg_cs(p_values, t_max, m, rng, batch=500):
    """Cumulative miscoverage curve of the CS, per p."""
    t = np.arange(1, t_max + 1, dtype=float)
    out = {}
    for p in p_values:
        first_miss = np.full(m, t_max + 1, dtype=int)
        for s in range(0, m, batch):
            k = min(batch, m - s)
            x = np.cumsum(rng.binomial(1, p, size=(k, t_max)), axis=1)
            miss = ~cs_covers(p, x, t[None, :], ALPHA)
            any_miss = miss.any(axis=1)
            idx = np.argmax(miss, axis=1) + 1
            first_miss[s:s + k] = np.where(any_miss, idx, t_max + 1)
        out[p] = np.array([(first_miss <= tt).mean() for tt in t])
        print(f"  p={p:.2f}  miscoverage by t={t_max}: {out[p][-1]:.4f}", flush=True)
    return t, out


def check_connected(grid, trials, rng, mesh=399):
    """C_j must be an interval, or the outward bisection in exact_grid_interval would
    report only the component containing p_hat. Checked at small n, where lattice
    discreteness is worst."""
    cache = GammaCache(grid, ALPHA, decimals=3)
    ps = np.linspace(0.002, 0.998, mesh)
    j, bad = len(grid) - 1, 0
    for _ in range(trials):
        p = rng.uniform(0.05, 0.95)
        counts = np.cumsum(rng.binomial(np.diff(np.concatenate([[0], grid])), p)).tolist()
        m = np.array([in_confidence_set(float(q), counts, grid, j, cache) for q in ps])
        idx = np.flatnonzero(m)
        if len(idx) and (idx[-1] - idx[0] + 1) != len(idx):
            bad += 1
    return bad


def check_cs_boundary(rng, n=200):
    """cs_covers must agree with the interval the GUI actually prints."""
    bad = 0
    for _ in range(n):
        t = int(rng.integers(1, 400))
        x = int(rng.integers(0, t + 1))
        p = float(rng.uniform(0.01, 0.99))
        lo, hi = cs_interval(x, t, ALPHA)
        if (lo <= p <= hi) != bool(cs_covers(p, x, t, ALPHA)):
            bad += 1
    return bad


# ── 4. end to end, through the GUI's own stream code ───────────────────────────

DETECTORS = ["d0", "d1", "d2", "d3"]
SENS = [0.85, 0.70, 0.95, 0.72]     # P(fires | real)
FPR = [0.30, 0.25, 0.80, 0.28]      # P(fires | not real)
RULES = {"d0": lambda z: z[0], "d1": lambda z: z[1], "d2": lambda z: z[2],
         "d3": lambda z: z[3],
         "atleast_2": lambda z: sum(z) >= 2, "atleast_3": lambda z: sum(z) >= 3,
         "unanimous": lambda z: sum(z) == 4}
# The same rules over the whole pool at once, for the estimands.
RULES_VEC = {"d0": lambda z: z[:, 0] > 0, "d1": lambda z: z[:, 1] > 0,
             "d2": lambda z: z[:, 2] > 0, "d3": lambda z: z[:, 3] > 0,
             "atleast_2": lambda z: z.sum(1) >= 2, "atleast_3": lambda z: z.sum(1) >= 3,
             "unanimous": lambda z: z.sum(1) == 4}


def make_pool(n, rng, prevalence=0.35):
    """A candidate pool: 4 correlated detector bits plus the annotator's truth."""
    x = rng.binomial(1, prevalence, size=n)
    z = np.empty((n, len(DETECTORS)), dtype=np.int8)
    for k in range(len(DETECTORS)):
        z[:, k] = rng.binomial(1, np.where(x == 1, SENS[k], FPR[k]))
    # Only candidates at least one detector fired on exist at all.
    keep = z.sum(axis=1) > 0
    return z[keep], x[keep]


def true_rates(z, x):
    """Pool-level precision / relative recall / F1 per rule -- the estimands."""
    out = {}
    for name, fn in RULES_VEC.items():
        v = fn(z)
        real = x == 1
        prec = (v & real).sum() / v.sum()
        rec = (v & real).sum() / real.sum()
        jac = (v & real).sum() / (v | real).sum()
        out[name] = {"precision": prec, "recall": rec, "f1": jac}
    return out


def leg_end_to_end(z, x, truth, grids, n_annot, trials, rng, invert_trials):
    """Coverage of the bar the GUI would print, per rule and target."""
    caches = {k: GammaCache(g, ALPHA) for k, g in grids.items()}
    rule_names, target_names = list(RULES), list(STREAM_KINDS)
    # nan marks a bar the run never reached a checkpoint for, so it is not scored.
    ok = np.full((trials, len(rule_names), len(target_names)), np.nan)
    inv_hit, inv_n = 0, 0
    n_pool = len(x)
    zt = [tuple(int(v) for v in row) for row in z]   # rule_fn is called per annotation
    for it in range(trials):
        idx = rng.choice(n_pool, size=n_annot, replace=False)
        ann = [{"z": zt[i], "label": int(x[i])} for i in idx]
        for ri, (rule, fn) in enumerate(RULES.items()):
            for ki, (kind, extract) in enumerate(STREAM_KINDS.items()):
                grid = grids[kind]
                stream = extract(ann, fn)
                counts, crossed = counts_at(stream, grid)
                if crossed == 0:
                    continue
                j = crossed - 1
                p_true = truth[rule][kind]
                ok[it, ri, ki] = in_confidence_set(p_true, counts, grid, j, caches[kind])
                # Same trial, but through the inversion the GUI actually prints.
                if it < invert_trials and rule == "atleast_2":
                    lo, hi = exact_grid_interval(counts, grid, j, ALPHA,
                                                 gamma_of_p=caches[kind])
                    if kind == "f1":
                        lo, hi = f1_from_jaccard(lo), f1_from_jaccard(hi)
                        p_true = f1_from_jaccard(p_true)
                    inv_hit += lo <= p_true <= hi
                    inv_n += 1
        if (it + 1) % 100 == 0:
            print(f"  trial {it + 1}/{trials}", flush=True)
    return ok, rule_names, target_names, (inv_hit, inv_n)


def cluster_ci(mat, rng, b=3000):
    """(point, lo, hi) for the coverage of `mat` (trials x bars), resampling trials.

    Bars within a trial share one annotation run, so they are not independent; the
    bootstrap unit has to be the trial.
    """
    flat = mat.reshape(len(mat), -1)
    point = np.nanmean(flat)
    idx = rng.integers(0, len(flat), size=(b, len(flat)))
    draws = np.array([np.nanmean(flat[i]) for i in idx])
    return float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ── plot ───────────────────────────────────────────────────────────────────────

def plot(out_path, rows, per_cp, t, cs_curves, e2e):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax = axes[0, 0]
    p = [r["p"] for r in rows]
    ax.plot(p, [r["exact_ideal"][-1] for r in rows], "-", color=COLOR_REF, lw=2.4,
            label="exact, gamma(p)")
    ax.plot(p, [r["exact_server"][-1] for r in rows], "-", color=PALETTE[0], lw=1.4,
            label="exact, server GammaCache")
    ax.plot(p, [r["mc"][-1] for r in rows], "o", color=PALETTE[1], ms=3.5,
            label="Monte Carlo")
    ax.axhline(1 - ALPHA, color="#e34948", ls="--", lw=1.2,
               label="nominal 0.95")
    ax.set_xlabel("true rate p")
    ax.set_ylabel("simultaneous coverage")
    ax.set_title("Exact grid: coverage over all 12 checkpoints")
    ax.set_ylim(0.940, 0.965)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[0, 1]
    js = np.arange(1, len(GRID) + 1)
    for i, (pv, (ex, mc)) in enumerate(per_cp.items()):
        ax.plot(js, ex, "-", color=PALETTE[i], lw=1.8, label=f"p={pv:g} exact")
        ax.plot(js, mc, "o", color=PALETTE[i], ms=3.5, mfc="none")
    ax.axhline(1 - ALPHA, color=COLOR_REF, ls="--", lw=1.2)
    ax.set_xlabel("checkpoint j")
    ax.set_ylabel("coverage of C_j")
    ax.set_title("Coverage at each committed checkpoint")
    ax.set_ylim(0.93, 1.005)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 0]
    for i, (pv, curve) in enumerate(cs_curves.items()):
        ax.plot(t, curve, color=PALETTE[i], lw=1.6, label=f"p={pv:g}")
    ax.axhline(ALPHA, color=COLOR_REF, ls="--", lw=1.2, label="nominal 0.05")
    ax.set_xscale("log")
    ax.set_xlabel("annotations t (log)")
    ax.set_ylabel("P(missed by t)")
    ax.set_title("Live panel CS: cumulative miscoverage")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    labels, vals, los, his = [], [], [], []
    for name, (v, lo, hi) in e2e.items():
        labels.append(name)
        vals.append(v)
        los.append(v - lo)
        his.append(hi - v)
    y = np.arange(len(labels))
    ax.errorbar(vals, y, xerr=[los, his], fmt="o", ms=5, lw=1.4, color=PALETTE[0],
                ecolor=PALETTE[0], capsize=3)
    ax.axvline(1 - ALPHA, color=COLOR_REF, ls="--", lw=1.2, label="nominal 0.95")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("empirical coverage")
    ax.set_title("End-to-end through the GUI stream code")
    ax.legend(fontsize=8, loc="lower left")
    for a in axes.ravel():
        a.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/eval/ci_calibration.png")
    ap.add_argument("--mc", type=int, default=20000, help="paths per p for the grid legs")
    ap.add_argument("--cs-mc", type=int, default=4000)
    ap.add_argument("--cs-tmax", type=int, default=2000)
    ap.add_argument("--e2e-trials", type=int, default=600)
    ap.add_argument("--invert-trials", type=int, default=60)
    ap.add_argument("--invert-paths", type=int, default=25,
                    help="full inversions on the real grid (slow: a cold gamma "
                         "calibration is ~0.26 s)")
    ap.add_argument("--connect-trials", type=int, default=20)
    ap.add_argument("--gamma-scan", type=int, default=0,
                    help="random p to scan for the GammaCache rounding defect "
                         "(slow: ~1 s per p; 120 is enough to see it)")
    ap.add_argument("--n-p", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("[1] exact grid, coverage vs p")
    # Deliberately off-lattice so GammaCache's rounding to 4 decimals is exercised.
    p_values = np.linspace(0.0213, 0.9787, args.n_p)
    rows = leg_grid(p_values, args.mc, rng)

    print("[2] per-checkpoint coverage")
    cache = GammaCache(GRID, ALPHA)
    per_cp = {}
    for pv in (0.25, 0.6, 0.9):
        per_cp[pv] = grid_coverage(pv, GRID, cache(pv), args.mc, rng)
    cov, agree = check_inversion(0.6, args.invert_paths, rng, cache)
    print(f"  inversion vs membership agreement at p=0.6: {agree:.3f} (cover {cov:.3f})")

    bad_cc = check_connected(even_grid(200, 4), args.connect_trials, rng)
    print(f"  disconnected confidence sets: {bad_cc}/{args.connect_trials}")

    if args.gamma_scan:
        print("[3b] GammaCache rounding")
        gamma_scan(args.gamma_scan, rng)

    print("[3] anytime CS")
    bad = check_cs_boundary(rng)
    print(f"  cs_covers disagreements with cs_interval: {bad}/200")
    t, cs_curves = leg_cs([0.1, 0.5, 0.85], args.cs_tmax, args.cs_mc, rng)

    print("[4] end to end")
    z, x = make_pool(400000, rng)
    truth = true_rates(z, x)
    n_annot = 600
    grids = {"precision": even_grid(200, 4), "recall": even_grid(200, 4),
             "f1": even_grid(240, 4)}
    ok, rule_names, target_names, (inv_hit, inv_n) = leg_end_to_end(
        z, x, truth, grids, n_annot, args.e2e_trials, rng, args.invert_trials)

    e2e = {}
    for ki, kind in enumerate(target_names):
        v, lo, hi = cluster_ci(ok[:, :, ki], rng)
        e2e[f"{kind}, 7 rules"] = (v, lo, hi)
        per_rule = np.nanmean(ok[:, :, ki], axis=0)
        print(f"  {kind}: coverage {v:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"per-rule min {np.nanmin(per_rule):.3f} max {np.nanmax(per_rule):.3f}")
    fam = np.array([np.all(t[~np.isnan(t)] > 0) for t in ok.reshape(len(ok), -1)])
    lo, hi = wilson(fam.sum(), len(fam))
    e2e["all 21 bars jointly"] = (fam.mean(), lo, hi)
    print(f"  family-wise (all 21 bars at once): {fam.mean():.4f} [{lo:.4f}, {hi:.4f}]")
    if inv_n:
        lo, hi = wilson(inv_hit, inv_n)
        e2e["printed lo/hi (atleast_2)"] = (inv_hit / inv_n, lo, hi)
        print(f"  printed interval coverage: {inv_hit / inv_n:.4f} over {inv_n} bars")

    worst = min(rows, key=lambda r: r["exact_server"][-1])
    print(f"\nworst exact simultaneous coverage: {worst['exact_server'][-1]:.5f} "
          f"at p={worst['p']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(args.out, rows, per_cp, t, cs_curves, e2e)


if __name__ == "__main__":
    main()
