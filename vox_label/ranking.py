"""Anytime-valid scores and pairwise ranking of decision rules by S_omega (main.pdf).

S_omega(R) = TP / (omega A_R + (1 - omega) A*): recall at omega=0, F1 at 0.5, precision
at 1. Everything here is betting (Waudby-Smith & Ramdas 2024), valid at every n, so it
may be recomputed after each annotation.

Pairs extend the paper's test of S_j = S_k to the family S_j = s + gap, S_k = s. At
gap = 0 this is exactly eq. 41; inverting over gap gives a confidence sequence for the
gap, from which every relation of eq. 9 is read.
"""
import math
from itertools import combinations

import numpy as np
from scipy.stats import norm

N0_SINGLE = 10          # section 2.4 default
N0_SUM = 10             # sum direction; see pair_grid_eval
# Eq. 51 assumes a gap near delta; for far-apart rules it gives thousands and a stake
# stuck near 0. Capping at 100 cost <=16% in simulation at either extreme.
N0_Y_CAP = 100
_CHUNK = 4_000_000      # T x G elements per block

RELATIONS = ("succ_delta", "approx", "succ", "unresolved")


# ── the betting kernel ─────────────────────────────────────────────────────────

def _excl_cumsum(a):
    out = np.zeros_like(a)
    np.cumsum(a[:-1], axis=0, out=out[1:])
    return out


def _log_wealth(C, P, sig0, n0, cap):
    """Log-wealth paths (T, G) of V_t = C_t . P_g under the stake of eq. 20-21.

    Because V is linear in the parameters, the running sums of V and V^2 at every
    grid point come from prefix sums of C and of its outer products.
    """
    T, K = C.shape
    Cx = _excl_cumsum(C)
    Qx = _excl_cumsum(C[:, :, None] * C[:, None, :])
    out = np.empty((T, len(P)))
    step = max(1, _CHUNK // max(T, 1))
    for g0 in range(0, len(P), step):
        p = P[g0:g0 + step]
        s1 = Cx @ p.T
        s2 = np.zeros_like(s1)
        for k in range(K):
            for l in range(K):
                s2 += Qx[:, k, l, None] * (p[:, k] * p[:, l])[None, :]
        den = n0[g0:g0 + step] * sig0[g0:g0 + step] + np.maximum(s2, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(den > 0, s1 / den, 0.0)
        c = cap[g0:g0 + step]
        lam = np.clip(lam, -c, c)
        out[:, g0:g0 + step] = np.cumsum(np.log1p(lam * (C @ p.T)), axis=0)
    return out


def _moments(C, p, sig0, n0):
    """mu-hat and sigma-hat^2 of eq. 21 after all T candidates, at one point."""
    v = C @ p
    n = len(v)
    return v.sum() / (n0 + n), (n0 * sig0 + (v * v).sum()) / (n0 + n)


def _first_crossing(logm, thr):
    """1-based index of the first candidate at which log-wealth reached thr, else inf."""
    hit = logm >= thr
    any_hit = hit.any(axis=0)
    tau = np.full(logm.shape[1], np.inf)
    tau[any_hit] = hit.argmax(axis=0)[any_hit] + 1
    return tau


# ── grid refinement ────────────────────────────────────────────────────────────

def _lattice(lo, hi, step):
    a, b = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    return np.arange(a, b + 1) * step


def _refine(evaluate, bounds, levels, feasible, n):
    """Evaluate on nested lattices, each confined to the box around the previous
    level's accepted points. Lattices are aligned, so later calls see a subset of
    earlier points and reported sets can only shrink.

    Returns (points (G, D), tau (G,), extra) concatenated over levels.
    """
    box = [list(b) for b in bounds]
    seen, pts_all, tau_all, extra_all = set(), [], [], []
    for steps in levels:
        axes = [_lattice(max(lo, b[0]), min(hi, b[1]), st)
                for (lo, hi), b, st in zip(box, bounds, steps)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))
        mesh = np.round(mesh, 10)
        mesh = mesh[feasible(mesh)]
        keep = [i for i, q in enumerate(map(tuple, mesh)) if q not in seen]
        mesh = mesh[keep]
        seen.update(map(tuple, mesh))
        if len(mesh):
            tau, extra = evaluate(mesh)
            pts_all.append(mesh), tau_all.append(tau), extra_all.append(extra)
        pts = np.concatenate(pts_all)
        tau = np.concatenate(tau_all)
        acc = pts[tau > n]
        if len(acc):
            box = [[acc[:, d].min() - st, acc[:, d].max() + st]
                   for d, st in enumerate(steps)]
        else:
            c = pts[np.argmax(tau)]
            box = [[c[d] - 2 * st, c[d] + 2 * st] for d, st in enumerate(steps)]
    return pts, tau, np.concatenate(extra_all)


# ── per-candidate terms ────────────────────────────────────────────────────────

def terms(x, r, omega):
    """a = x R and w = omega R + (1-omega) x, so Z(s) = a - s w (eq. 13)."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    return x * r, omega * r + (1 - omega) * x


def denominator(a_r, a_star, omega):
    return omega * a_r + (1 - omega) * a_star


def sigma0_single(s, a_r, a_star, omega):
    """Eq. 22: the null variance of Z(s), with A* a rough guess."""
    d = denominator(a_r, a_star, omega)
    return np.maximum(s * d * (1 - 2 * s) + s * s * (
        omega ** 2 * a_r + 2 * omega * (1 - omega) * s * d + (1 - omega) ** 2 * a_star), 0.0)


# ── one rule ───────────────────────────────────────────────────────────────────

def rule_cs(x, r, omega, alpha, a_r, a_star=0.5):
    """Confidence sequence for S_omega(R) (eq. 18), as its hull at the current n."""
    a, w = terms(x, r, omega)
    n = len(a)
    est = a.sum() / w.sum() if n and w.sum() > 0 else None
    if n == 0:
        return {"n": 0, "estimate": None, "lo": 0.0, "hi": 1.0}
    C = np.stack([a, -w], 1)
    thr = math.log(1 / alpha)

    def evaluate(pts):
        s = pts[:, 0]
        P = np.stack([np.ones_like(s), s], 1)
        g = len(s)
        logm = _log_wealth(C, P, sigma0_single(s, a_r, a_star, omega),
                           np.full(g, N0_SINGLE), np.full(g, 0.5))
        return _first_crossing(logm, thr), np.zeros(g)

    pts, tau, _ = _refine(evaluate, [(0.0, 1.0)], [(0.01,), (0.001,)],
                          lambda m: np.ones(len(m), bool), n)
    acc = pts[tau > n, 0]
    lo, hi = (float(acc.min()), float(acc.max())) if len(acc) else (None, None)
    return {"n": n, "estimate": est, "lo": lo, "hi": hi}


# ── a pair of rules ────────────────────────────────────────────────────────────

# (x, r_j, r_k) for all eight outcomes of one annotation.
_OUTCOMES = np.array([(x, j, k) for x in (0, 1) for j in (0, 1) for k in (0, 1)], float)


def _pair_coeffs(x, rj, rk, omega):
    """Rows (T, 3) for Y and Sigma as linear forms in (1, s, gap), eq. 34-35 with
    S_j = s + gap and S_k = s."""
    aj, wj = terms(x, rj, omega)
    ak, wk = terms(x, rk, omega)
    return (np.stack([aj - ak, -(wj - wk), -wj], 1),
            np.stack([aj + ak, -(wj + wk), -wj], 1))


def bounds(s, gap, omega):
    """B_Y and B_Sigma: the largest |Y|, |Sigma| over the eight outcomes."""
    oY, oS = _pair_coeffs(*_OUTCOMES.T, omega)
    P = np.stack([np.ones_like(s), s, gap], 1)
    return np.abs(oY @ P.T).max(0), np.abs(oS @ P.T).max(0)


def pair_grid_eval(x, rj, rk, omega, pat, d_min, delta, thr):
    """evaluate() for _refine over points (s, gap).

    Priors: Y takes the corpus bound of eq. 48, generalised to gap != 0 by weighting
    each firing pattern with its max over the unknown label; pseudo-count eq. 51, capped.
    Sigma takes B_Sigma^2 (eq. 48) and N0_SUM, the single-rule default: the design gap
    in eq. 51 would give it tens of thousands of pseudo-counts and a stake stuck at 0.
    """
    CY, CS = _pair_coeffs(x, rj, rk, omega)
    oY, _ = _pair_coeffs(*_OUTCOMES.T, omega)
    mu = max(delta, 1e-3) * d_min

    def evaluate(pts):
        s, gap = pts[:, 0], pts[:, 1]
        P = np.stack([np.ones_like(s), s, gap], 1)
        vY, (bY, bS) = oY @ P.T, bounds(s, gap, omega)
        sig0_y = np.zeros(len(s))
        for (j, k), p in pat.items():
            rows = (_OUTCOMES[:, 1] == j) & (_OUTCOMES[:, 2] == k)
            sig0_y += p * (vY[rows] ** 2).max(0)
        n0_y = np.maximum(sig0_y / (2 * mu * mu), N0_SINGLE)
        n0_y = np.minimum(n0_y, N0_Y_CAP)
        lY = _log_wealth(CY, P, sig0_y, n0_y, 1 / (2 * np.maximum(bY, 1e-12)))
        lS = _log_wealth(CS, P, bS ** 2, np.full(len(s), N0_SUM),
                         1 / (2 * np.maximum(bS, 1e-12)))
        logm = np.logaddexp(lY, lS) - math.log(2)
        extra = np.stack([logm[-1], sig0_y, n0_y, bS ** 2], 1)
        return _first_crossing(logm, thr), extra

    return evaluate, CY, CS


def _gap_step(delta):
    return delta / math.ceil(delta / 0.0025) if delta > 0 else 0.0025


def must_reject(gap, relation, delta, flip=False):
    """Mask of gaps that must all be rejected to declare `relation` (eq. 9)."""
    g = -gap if flip else gap
    if relation == "succ_delta":
        return g <= delta
    if relation == "succ":
        return g <= 0
    if relation == "approx":
        return np.abs(g) >= delta
    raise ValueError(relation)


def compare(x, rj, rk, omega, delta, alpha, pat, d_j, d_k, forecast_beta=0.2):
    """The strongest relation between rules j and k established so far.

    `pat` maps firing patterns (r_j, r_k) to their corpus frequency; `d_j`, `d_k` are
    the planning denominators. Returns the gap CS hull, the relation and the first n
    at which each relation holds on this call's grid (never retracted later).
    """
    n = len(x)
    thr = math.log(1 / alpha)
    d_min = min(d_j, d_k)
    evaluate, CY, CS = pair_grid_eval(x, rj, rk, omega, pat, d_min, delta, thr)
    h = _gap_step(delta)
    pts, tau, extra = _refine(
        evaluate, [(0.0, 1.0), (-1.0, 1.0)],
        [(0.08, 32 * h), (0.02, 8 * h), (0.005, h)],
        lambda m: (m[:, 0] + m[:, 1] >= -1e-9) & (m[:, 0] + m[:, 1] <= 1 + 1e-9), n)
    gap = pts[:, 1]
    acc = tau > n
    lo, hi = (float(gap[acc].min()), float(gap[acc].max())) if acc.any() else (None, None)

    # First declaration time of each relation: the last rejection it needed.
    declared = {}
    for rel, flip, key in (("succ_delta", False, "j_succ_delta"),
                           ("succ_delta", True, "k_succ_delta"),
                           ("approx", False, "approx"),
                           ("succ", False, "j_succ"), ("succ", True, "k_succ")):
        m = must_reject(gap, rel, delta, flip)
        t = float(tau[m].max()) if m.any() else 0.0
        declared[key] = int(t) if t <= n and acc.any() else None

    relation, winner = "unresolved", None
    for key, rel, win in (("j_succ_delta", "succ_delta", "j"), ("k_succ_delta", "succ_delta", "k"),
                          ("approx", "approx", None),
                          ("j_succ", "succ", "j"), ("k_succ", "succ", "k")):
        if declared[key] is not None:
            relation, winner = rel, win
            break

    out = {"relation": relation, "winner": winner, "gap_lo": lo, "gap_hi": hi,
           "declared": declared, "n": n, "forecast": None}
    if relation in ("unresolved", "succ") and n:
        out["forecast"] = _forecast(pts, tau, extra, CY, CS, n, thr, delta, forecast_beta)
    return out


def _forecast(pts, tau, extra, CY, CS, n, thr, delta, beta):
    """Eq. 60-62 at the least-rejected point, for the soonest reachable resolution."""
    best = None
    gap = pts[:, 1]
    for key, rel, flip in (("j_succ_delta", "succ_delta", False),
                           ("k_succ_delta", "succ_delta", True), ("approx", "approx", False)):
        live = must_reject(gap, rel, delta, flip) & (tau > n)
        if not live.any():
            continue
        i = np.flatnonzero(live)[np.argmin(extra[live, 0])]
        logm, sig0_y, n0_y, sig0_s = extra[i]
        p = np.array([1.0, pts[i, 0], pts[i, 1]])
        c = 0.0
        for C, s0, k0 in ((CY, sig0_y, n0_y), (CS, sig0_s, N0_SUM)):
            mu, var = _moments(C, p, s0, k0)
            if var > 0:
                c = max(c, mu * mu / (2 * var))
        if c <= 0:
            continue
        left = max(thr - logm, 0.0)
        z = norm.ppf(1 - beta)
        v = 2 * c
        root = (z * math.sqrt(v) + math.sqrt(z * z * v + 4 * c * left)) / (2 * c)
        cand = {"n_hat": n + left / c, "remaining": left / c,
                "remaining_hi": root * root, "towards": key,
                "reliable": n > 8 * n0_y, "n0_y": float(n0_y)}
        if best is None or cand["n_hat"] < best["n_hat"]:
            best = cand
    return best


# ── planning (eq. 56) ──────────────────────────────────────────────────────────

def plan_pair(p_dis, d_j, d_k, delta, omega, alpha, beta=0.2, s=0.5, grid=201):
    """Annotations for power 1-beta to detect a gap delta between two rules, before
    any label exists: eq. 56 with mu = delta min D, sigma^2 = B_Y^2 p_dis, v = 2c."""
    b_y = max(s * omega, 1 - s * omega)
    sig2 = b_y ** 2 * p_dis
    mu = delta * min(d_j, d_k)
    if sig2 <= 0 or mu <= 0:
        return None
    c = mu * mu / (2 * sig2)
    ell = math.log(1 / alpha) + math.log(2)
    v, b = 2 * c, math.log(3)
    return math.ceil(max(2 * ell / c, 4 * (2 * v + b * c / 3) / c ** 2
                         * math.log(grid / beta)))


# ── everything a campaign shows ────────────────────────────────────────────────

def corpus_stats(pool_rows, rules, rule_fns):
    """A_R per rule and firing-pattern frequencies per pair. Label-free, so computed
    once over the whole pool."""
    dec = {r: np.array([bool(rule_fns[r](row["z"])) for row in pool_rows], bool)
           for r in rules}
    a_r = {r: float(v.mean()) if len(v) else 0.0 for r, v in dec.items()}
    pats = {(j, k): {(a, b): float(np.mean((dec[j] == a) & (dec[k] == b)))
                     for a in (0, 1) for b in (0, 1)}
            for j, k in combinations(rules, 2)}
    return {"a_r": a_r, "patterns": pats}


def rank(annotations, rules, rule_fns, corpus, omega, delta, alpha, a_star=0.5):
    """Per-rule CSs at alpha and the partial order at alpha / #pairs (eq. 11).

    `annotations` must be in committed order, as Campaign.annotations returns them;
    `corpus` comes from corpus_stats.
    """
    x = np.array([a["label"] for a in annotations], float)
    dec = {r: np.array([bool(rule_fns[r](a["z"])) for a in annotations], float)
           for r in rules}
    a_r = corpus["a_r"]
    pairs = list(combinations(rules, 2))
    alpha_pair = alpha / max(len(pairs), 1)

    per_rule = [{"rule": r, "a_r": a_r[r],
                 **rule_cs(x, dec[r], omega, alpha, a_r[r], a_star)} for r in rules]

    out_pairs = []
    for j, k in pairs:
        pat = corpus["patterns"][(j, k)]
        d_j = denominator(a_r[j], a_star, omega)
        d_k = denominator(a_r[k], a_star, omega)
        res = compare(x, dec[j], dec[k], omega, delta, alpha_pair, pat, d_j, d_k)
        p_dis = pat[(1, 0)] + pat[(0, 1)]
        out_pairs.append({"a": j, "b": k, "p_dis": p_dis,
                          "plan_n": plan_pair(p_dis, d_j, d_k, delta, omega, alpha_pair),
                          **res})
    return _plain({"omega": omega, "delta": delta, "alpha": alpha,
                   "alpha_pair": alpha_pair, "a_star_guess": a_star, "n": len(x),
                   "rules": per_rule, "pairs": out_pairs})


def _plain(v):
    """numpy scalars to Python, so the result serialises as JSON."""
    if isinstance(v, dict):
        return {k: _plain(u) for k, u in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(u) for u in v]
    return v.item() if isinstance(v, np.generic) else v
