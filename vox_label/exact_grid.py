"""Exact confidence intervals for a Bernoulli rate on a committed grid of checkpoints.

Implements sections 4-5 of exact-grid-confidence-intervals.pdf (repo root).

The problem this solves
-----------------------
A researcher annotates candidate detections one block at a time and wants to stop
as soon as the error bars look tight enough. That is fatal for a textbook interval,
whose 1-alpha guarantee attaches to a single sample size fixed before any data are
seen. The fix used here is to commit in advance to a *grid* of stopping points

    n_1 < n_2 < ... < n_J = n_max

and consult the interval only at those points. We then need simultaneous coverage,

    P(p in C_j for every j <= J)  >=  1 - alpha,                           (eq. 9)

which makes stopping at whichever checkpoint the researcher likes safe, including a
checkpoint chosen after seeing all of them.

The construction is by inversion. For each candidate rate p we specify an acceptance
region A_j(p) at every checkpoint and take

    C_j = { p : the count at checkpoint i lies in A_i(p) for all i <= j }.  (eq. 10)

Note C_j is an *intersection* over i <= j: a rate ruled out early stays ruled out, so
the reported bars are monotone -- they shrink as annotation proceeds and never widen.

Why it is cheap
---------------
Calibrating eq. 9 needs the probability that a binomial path stays inside a sequence
of regions. The classical literature approximates this with Brownian motion and reads
off a boundary constant (Pocock, 1977). It does not have to. For fixed p the count at
checkpoint j+1 is the count at checkpoint j plus an independent Binomial(n_{j+1}-n_j, p),
so the surviving probability mass propagates forward exactly, one block at a time --
a convolution followed by a masking (eq. 13). That is the Armitage et al. (1969)
recursion carried out on the binomial lattice rather than in its normal limit.

Because the J statistics are strongly correlated (Corr(Z_i, Z_j) = sqrt(n_i/n_j),
eq. 12 -- adjacent looks at J=12 correlate at 0.96), the calibrated per-checkpoint
level is 2.5-3.2x the Bonferroni level, and the intervals come out 11-13% narrower
than Bonferroni-Clopper-Pearson while carrying an exact guarantee.

Everything in this module is a repeated application of `survival`.
"""
import math

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import binom

# Boundary constants c(J) solving P(max_{j<=J} |Z_j| < c) = 1 - alpha under the
# correlation structure of eq. 12, at alpha = 0.05. Table 2 of the report. Used only
# by `plan_n`; the exact construction below never needs them.
_C_OF_J = {1: 1.960, 2: 2.179, 4: 2.361, 6: 2.455, 12: 2.586, 24: 2.694, 100: 2.872}

# Above this output length the FFT beats the direct O(n*m) convolution comfortably.
_FFT_MIN = 512


def cp_acceptance(n, p, gamma):
    """Equal-tailed Clopper-Pearson acceptance region for Binomial(n, p) at level gamma.

    Returns the inclusive bounds (s_lo, s_hi) of

        A(p) = { s : P(S <= s) > gamma/2  and  P(S >= s) > gamma/2 },

    or (1, 0) -- an empty range -- if no count qualifies. Inverting this family over p
    at a single checkpoint is exactly the Clopper-Pearson interval, which is why the
    J=1 case of `exact_grid_interval` reproduces it.
    """
    if gamma <= 0.0:
        return 0, n
    half = gamma / 2.0
    s = np.arange(n + 1)
    # P(S <= s) > half  =>  s >= first index where the cdf clears half.
    lo_ok = binom.cdf(s, n, p) > half
    # P(S >= s) > half  =>  sf(s-1) > half, decreasing in s.
    hi_ok = binom.sf(s - 1, n, p) > half
    ok = lo_ok & hi_ok
    if not ok.any():
        return 1, 0
    idx = np.flatnonzero(ok)
    return int(idx[0]), int(idx[-1])


def _convolve(v, b):
    """v * b, using the FFT once the result is long enough for it to pay off.

    fftconvolve can emit tiny negative values from round-off; they are clipped away
    because `v` is a (sub-)probability vector and negative mass is meaningless.
    """
    if len(v) + len(b) - 1 >= _FFT_MIN:
        out = fftconvolve(v, b)
        np.clip(out, 0.0, None, out=out)
        return out
    return np.convolve(v, b)


def survival_with_regions(p, grid, regions):
    """Probability that a Binomial(n_max, p) path stays inside every region (eq. 13).

    `grid` is the committed checkpoints n_1 < ... < n_J (cumulative annotation counts);
    `regions` gives the inclusive (s_lo, s_hi) acceptance bounds at each one.

    The recursion is

        v_j = ( v_{j-1} * Bin(n_j - n_{j-1}, p) ) . 1{ . in A_j(p) },   v_0 = delta_0,

    i.e. advance the distribution of the running count by one block, then delete the
    mass that escaped. After the first deletion v_j is no longer a probability vector:
    it is the joint probability of reaching each count *and* having stayed inside every
    region so far, which is precisely what we want to total up.

    Deleting mass is legitimate because the candidates annotated in the next block are
    drawn independently of what happened earlier, so the same convolution applies to
    the surviving remnant as would apply to the whole. This is the one place the
    uniform-sampling assumption does real work.
    """
    v = np.array([1.0])
    prev = 0
    for n_j, (s_lo, s_hi) in zip(grid, regions):
        block = n_j - prev
        if block > 0:
            v = _convolve(v, binom.pmf(np.arange(block + 1), block, p))
        # Mask to the acceptance region, padding v out if it is shorter than s_hi.
        if s_hi < s_lo:
            return 0.0
        keep = np.zeros_like(v)
        hi = min(s_hi, len(v) - 1)
        if s_lo <= hi:
            keep[s_lo:hi + 1] = v[s_lo:hi + 1]
        v = keep
        prev = n_j
    return float(v.sum())


def survival(p, grid, gamma):
    """`survival_with_regions` with Clopper-Pearson regions at level gamma -- the
    left-hand side of eq. 11."""
    regions = [cp_acceptance(n_j, p, gamma) for n_j in grid]
    return survival_with_regions(p, grid, regions)


def calibrate_gamma(p, grid, alpha, tol=1e-7):
    """Largest per-checkpoint level gamma whose path-survival is still at least 1-alpha.

    Survival is monotone decreasing in gamma (a larger level shrinks every acceptance
    region, and the regions are nested), so a plain bisection is exact up to `tol`.
    The answer sits between the Bonferroni level alpha/J and alpha itself; at J=12,
    alpha=0.05 it lands near 0.0105, about 2.5x Bonferroni.
    """
    target = 1.0 - alpha
    lo, hi = 0.0, alpha
    if survival(p, grid, hi) >= target:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if survival(p, grid, mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo


class GammaCache:
    """Memoised gamma(p) on a rounded p-grid.

    Inverting eq. 10 evaluates gamma at many nearby p, and gamma varies smoothly and
    very slowly in p (0.0105 to 0.0131 across p = 0.5 to 0.99). Rounding p to `decimals`
    before calibrating cuts the work by an order of magnitude. Because a *smaller*
    gamma only widens the acceptance regions, the rounding cannot break the guarantee
    so long as the same cache is used for calibration and for membership -- which it is.
    """

    def __init__(self, grid, alpha, decimals=4):
        self.grid = tuple(grid)
        self.alpha = alpha
        self.decimals = decimals
        self._cache = {}

    def __call__(self, p):
        key = round(float(p), self.decimals)
        key = min(max(key, 10.0 ** -self.decimals), 1.0 - 10.0 ** -self.decimals)
        if key not in self._cache:
            self._cache[key] = calibrate_gamma(key, self.grid, self.alpha)
        return self._cache[key]


def in_confidence_set(p, counts, grid, j, gamma_of_p):
    """Is p in C_j (eq. 10)? True iff every observed count up to checkpoint j lies in
    that checkpoint's acceptance region at the calibrated level."""
    if p <= 0.0 or p >= 1.0:
        return False
    gamma = gamma_of_p(p)
    for i in range(j + 1):
        s_lo, s_hi = cp_acceptance(grid[i], p, gamma)
        if not (s_lo <= counts[i] <= s_hi):
            return False
    return True


def exact_grid_interval(counts, grid, j, alpha, gamma_of_p=None, tol=1e-6):
    """The exact interval C_j for the rate p, valid simultaneously across all checkpoints.

    `counts[i]` is the number of successes observed in the first `grid[i]` observations
    of the stream; `j` indexes the checkpoint being reported (0-based). Only counts
    0..j are consulted.

    Endpoints are found by bisecting p outward from the point estimate, as the report
    prescribes. The point estimate itself is in C_j except in degenerate cases, so it
    is a safe interior seed.
    """
    if j < 0 or j >= len(grid):
        raise IndexError(f"checkpoint {j} outside grid of length {len(grid)}")
    if gamma_of_p is None:
        gamma_of_p = GammaCache(grid, alpha)

    n, x = grid[j], counts[j]
    p_hat = x / n

    def inside(p):
        return in_confidence_set(p, counts, grid, j, gamma_of_p)

    # Seed the bisection at a point known to be inside. p_hat normally is; if the path
    # is odd enough that it is not, scan a coarse mesh for any interior point.
    seed = p_hat if inside(p_hat) else None
    if seed is None:
        for cand in np.linspace(1e-6, 1 - 1e-6, 401):
            if inside(cand):
                seed = float(cand)
                break
    if seed is None:
        return 0.0, 1.0

    if x <= 0:
        lo = 0.0
    else:
        a, b = 0.0, seed          # a outside, b inside
        while b - a > tol:
            mid = 0.5 * (a + b)
            if inside(mid):
                b = mid
            else:
                a = mid
        lo = b

    if x >= n:
        hi = 1.0
    else:
        a, b = seed, 1.0          # a inside, b outside
        while b - a > tol:
            mid = 0.5 * (a + b)
            if inside(mid):
                a = mid
            else:
                b = mid
        hi = a

    return lo, hi


def f1_from_jaccard(r):
    """F1 = 2r/(1+r) for the Jaccard rate r of eq. 7."""
    return 2.0 * r / (1.0 + r)


def f1_interval(counts, grid, j, alpha, gamma_of_p=None):
    """Interval for F1 by transforming the interval for the Jaccard rate r_V.

    F1 is a single Bernoulli rate in disguise: expanding the harmonic mean gives
    F1(V) = 2 r_V / (1 + r_V) with r_V = P(accepted and real | accepted or real),
    itself a conditional probability with a stream of its own. The map r -> 2r/(1+r)
    is strictly increasing, so the endpoints carry over with coverage preserved exactly
    -- no separate construction is needed.

    `counts`/`grid` here describe the *Jaccard* stream: candidates with V(z)=1 or x=1,
    recording whether both held.
    """
    lo, hi = exact_grid_interval(counts, grid, j, alpha, gamma_of_p=gamma_of_p)
    return f1_from_jaccard(lo), f1_from_jaccard(hi)


def c_of_j(J):
    """Boundary constant c(J) of Table 2, log-interpolated between tabulated J.

    c grows like sqrt(log log J), so interpolating in log J is both natural and
    accurate over the tabulated range.
    """
    ks = sorted(_C_OF_J)
    if J in _C_OF_J:
        return _C_OF_J[J]
    if J < ks[0] or J > ks[-1]:
        raise ValueError(f"J={J} outside tabulated range {ks[0]}..{ks[-1]}")
    return float(np.interp(np.log(J), np.log(ks), [_C_OF_J[k] for k in ks]))


def plan_n(sigma2, f, h, J=12):
    """Annotations needed for a half-width of h on one target (eq. 15).

        n  ~=  (2 sigma^2 / (f h^2)) * c(J)^2 / 2

    with (sigma^2, f) = (Prec(1-Prec), f_V) for a precision and
    (Rec(1-Rec), P(x=1)) for a relative recall. `f` is the fraction of annotations that
    reach the stream in question, which is what makes a selective rule expensive: it
    buys its precision estimate from only a slice of the budget.

    Three things worth knowing before spending money on this:

    * Recall usually binds, not precision. A precision near 0.9 has sigma^2 = 0.09; a
      relative recall near 1/2 has sigma^2 up to 0.25. The binding target is whichever
      rate is closest to 1/2, not whichever detector performs worst.
    * Corpus size is irrelevant -- N appears nowhere. A bigger corpus does not make
      benchmarking cheaper, it only widens the pool the same n is drawn from.
    * Tolerance dominates. The budget scales as 1/h^2, so loosening h is the only large
      lever available; dropping a detector is a rounding error by comparison.

    Returns a float; round up when committing a grid.
    """
    c = c_of_j(J)
    return (2.0 * sigma2 / (f * h * h)) * (c * c / 2.0)


def half_width(sigma2, f, n, J=12):
    """Eq. 15 the other way round: the half-width a budget of n buys for one target."""
    if f <= 0 or n <= 0:
        return float("nan")
    return math.sqrt(2 * sigma2 / (f * n) * (c_of_j(J) ** 2 / 2))


def even_grid(n_max, J):
    """J evenly spaced checkpoints ending at n_max, e.g. 500, 1000, ..., 6000.

    c(J) depends on the checkpoint times only through their ratios, so this grid is
    scale-free: doubling every checkpoint changes nothing, and starting the grid later
    (at the smallest n where stopping is even conceivable) tightens every subsequent
    interval for free.
    """
    if J < 1:
        raise ValueError("J must be >= 1")
    return [int(round(n_max * (i + 1) / J)) for i in range(J)]
