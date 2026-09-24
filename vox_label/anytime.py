"""Anytime-valid confidence sequence for a Bernoulli rate.

The companion construction to `exact_grid`, and the answer to "can I just watch the
score go by?". A confidence sequence is valid *simultaneously at every t*, so it may be
consulted after every single annotation with no grid, no commitment, and no penalty for
looking. The report prices this directly: it costs roughly a factor of two in
annotations against the committed grid (17,674 vs 9,722 at +/-2 points, table 3), which
is what buying unrestricted looking costs.

Both are used in this campaign, for different jobs:

* This CS drives the live panel during labeling. Always valid, always wider.
* `exact_grid` produces the headline numbers at committed checkpoints. Tighter.

Reporting an exact-grid interval after watching this panel is legitimate, and worth
being precise about why. Eq. 9 gives coverage at every checkpoint *simultaneously*, so
whichever checkpoint gets reported covers, and it does not matter how that checkpoint
was chosen -- including choosing it after seeing all of them. What is not allowed is
reporting a grid interval at a non-checkpoint length; that is the n = 4,500 failure of
section 6, and the server has no code path for it.

Implementation is the Krichevsky-Trofimov mixture, ported from
`scripts/evaluation/gt_budget_experiment.py:178` (eq. 12-19 of the confidence-sequences
note) so the campaign and the older budget experiment report the same bars.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln


def _kt_regret(x, t):
    """Exact regret of the KT forecaster after t rounds with x successes: the
    hindsight log-likelihood minus the forecaster's."""
    if x <= 0 or x >= t:
        hindsight = 0.0  # 0*log(0) := 0
    else:
        p_hat = x / t
        hindsight = x * np.log(p_hat) + (t - x) * np.log(1 - p_hat)
    forecaster = gammaln(x + 0.5) + gammaln(t - x + 0.5) - gammaln(t + 1) - np.log(np.pi)
    return hindsight - forecaster


def _kl_bernoulli(u, v):
    v = min(max(v, 1e-12), 1 - 1e-12)
    t1 = 0.0 if u <= 0 else u * np.log(u / v)
    t2 = 0.0 if u >= 1 else (1 - u) * np.log((1 - u) / (1 - v))
    return t1 + t2


def cs_interval(x, t, alpha):
    """Anytime-valid interval for p from x successes in t i.i.d. Bernoulli trials.

    Safe to recompute after every observation: the guarantee holds uniformly over t,
    so an annotator may watch it without invalidating anything.
    """
    if t <= 0:
        return 0.0, 1.0
    p_hat = x / t
    thresh = (np.log(1 / alpha) + _kt_regret(x, t)) / t

    def f(p):
        return _kl_bernoulli(p_hat, p) - thresh

    lo = 0.0 if x <= 0 else brentq(f, 1e-12, p_hat)
    hi = 1.0 if x >= t else brentq(f, p_hat, 1 - 1e-12)
    return float(lo), float(hi)
