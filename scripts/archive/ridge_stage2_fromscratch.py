"""
From-scratch stage-2 filter search (combinations, not a few hand-picked gates).

Stage-2 is a post-hoc filter on the final detection table, so any gate combination
can be scored directly on detections.csv (no pipeline rerun). This does:
  1. greedy forward selection from the empty filter over a gate pool (each gate =
     feature + tuned threshold), event-F1 objective -> the best combination "from
     scratch" and the order features add value;
  2. grouped 4-fold CV (mean +/- std) of a curated set that INCLUDES flatness-
     anchored combos (flatness+n_cols, flatness+centroid, ...) and multi-feature
     combos, ranked, so "why not combine / build on flatness" is answered directly.

Reads the freshly-rebuilt detections.csv (8 sessions, best stage-1 min_area=50).
    python scripts/ridge_stage2_fromscratch.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ridge_filter_experiment import discover_sessions, event_metrics, ALL_FEATS  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

OUT = ROOT / "outputs/eval/ridge_stage2_fromscratch"; OUT.mkdir(parents=True, exist_ok=True)
det = pd.read_csv(ROOT / "outputs/eval/ridge_filter_experiment/detections.csv")
sessions = discover_sessions(8)
names = [s[0] for s in sessions]; sess_by = {s[0]: s for s in sessions}
folds = [names[i::4] for i in range(4)]

# gate pool: (feature, op, [thresholds])
POOL = [
    ("flatness", "<=", [0.15, 0.21, 0.30]),
    ("ridge_score", ">=", [0.03, 0.05, 0.08]),
    ("n_cols", ">=", [7, 9, 11]),
    ("centroid", ">=", [23000, 25000, 27000]),
    ("f_high_khz", ">=", [24, 28, 30]),
    ("solidity", "<=", [0.5, 0.7, 0.85]),
    ("sweep_frac", ">=", [0.02, 0.04, 0.06]),
    ("duration_ms", ">=", [10, 18, 25]),
    ("zcr", ">=", [0.35, 0.40]),
]


def keep_of(df, gates):
    k = pd.Series(True, index=df.index)
    for f, op, t in gates:
        k &= (df[f] >= t) if op == ">=" else (df[f] <= t)
    return k.fillna(False)


def f1_all(gates):
    """Event F1 pooled over all 8 sessions for a gate list."""
    return event_metrics(det, keep_of(det, gates), sessions)["f1"]


def cv(gates_or_fn):
    R, P, F = [], [], []
    for te in folds:
        sub = [sess_by[n] for n in te]
        if callable(gates_or_fn):
            keep = gates_or_fn(te)
        else:
            keep = keep_of(det, gates_or_fn)
        m = event_metrics(det, keep.reindex(det.index, fill_value=False), sub)
        R.append(m["recall"]); P.append(m["precision"]); F.append(m["f1"])
    return dict(recall=np.mean(R), recall_std=np.std(R), precision=np.mean(P),
                precision_std=np.std(P), f1=np.mean(F), f1_std=np.std(F))


def tree_keep(te_names):
    tr = det[~det.session.isin(te_names)]; ted = det[det.session.isin(te_names)]
    X = tr[ALL_FEATS].fillna(tr[ALL_FEATS].median()); y = (tr.label == "tp").astype(int)
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(X, y)
    return pd.Series(t.predict(ted[ALL_FEATS].fillna(tr[ALL_FEATS].median())) == 1, index=ted.index)


# ---- 1. forward selection from empty ----
print("=== forward selection (event F1, pooled 8 sessions) ===")
chosen, used, cur = [], set(), f1_all([])
print(f"  start (no filter): F1={cur:.3f}")
while True:
    best = None
    for f, op, ths in POOL:
        if f in used:
            continue
        for t in ths:
            v = f1_all(chosen + [(f, op, t)])
            if best is None or v > best[0]:
                best = (v, f, op, t)
    if best[0] <= cur + 0.002:
        break
    cur = best[0]; chosen.append((best[1], best[2], best[3])); used.add(best[1])
    print(f"  + {best[1]} {best[2]} {best[3]}   -> F1={cur:.3f}")
fwd = chosen
print(f"  forward-selected filter: {['%s%s%g'%(f,o,t) for f,o,t in fwd]}")

# ---- 2. curated CV ranking (incl. flatness-anchored + combos) ----
FL = ("flatness", "<=", 0.21); NC = ("n_cols", ">=", 9); CE = ("centroid", ">=", 25000)
RS = ("ridge_score", ">=", 0.05); FH = ("f_high_khz", ">=", 28); SO = ("solidity", "<=", 0.7)
STRAT = {
    "raw":                         [],
    "current geo filter":          None,  # special
    "flatness":                    [FL],
    "ridge_score":                 [RS],
    "n_cols":                      [NC],
    "centroid":                    [CE],
    "flatness + n_cols":           [FL, NC],
    "flatness + centroid":         [FL, CE],
    "flatness + ridge_score":      [FL, RS],
    "flatness + n_cols + centroid":[FL, NC, CE],
    "n_cols + centroid":           [NC, CE],
    "ridge_score + n_cols":        [RS, NC],
    "flatness + ridge + n_cols":   [FL, RS, NC],
    "forward-selected":            fwd,
    "learned tree (d=3)":          "tree",
}
def base(d): return (d.area_frac <= 0.15) & (d.n_cols >= 5) & (d.sweep_frac >= 0.04)

rows = []
for name, spec in STRAT.items():
    if spec == "tree":
        m = cv(tree_keep)
    elif name == "current geo filter":
        m = cv(lambda te, c=base: keep_of(det, []) & base(det))
    else:
        m = cv(spec)
    rows.append({"strategy": name, **m})
S = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
S.to_csv(OUT / "stage2_strategies.csv", index=False)
print("\n=== stage-2 filters, grouped 4-fold CV, ranked by F1 ===")
for r in S.itertuples():
    print(f"  {r.strategy:30s} F1={r.f1:.3f}±{r.f1_std:.3f}  R={r.recall:.3f}  P={r.precision:.3f}")

# ---- figure: ranked horizontal bars ----
best_name = S.iloc[0].strategy
fig, ax = plt.subplots(figsize=(11, 7.5), constrained_layout=True)
d = S.iloc[::-1]
def color(n):
    if n == best_name: return "#0ca30c"
    if n == "current geo filter": return "#e08a1e"
    if n == "forward-selected": return "#8e5fd3"
    if n.startswith("flatness"): return "#2a78d6"
    return "#8a8984"
ax.barh(range(len(d)), d.f1, xerr=d.f1_std, color=[color(n) for n in d.strategy], capsize=2,
        error_kw=dict(lw=1, ecolor="#333"))
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.strategy, fontsize=9)
for i, (v, s) in enumerate(zip(d.f1, d.f1_std)):
    ax.text(v + s + 0.004, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("event-level F1 (grouped 4-fold CV, mean ± std)")
ax.set_title("From-scratch stage-2 filters ranked  "
             "(green=best, blue=flatness-anchored, purple=forward-selected, orange=current)",
             fontweight="bold", fontsize=11)
ax.set_xlim(0.5, 1.0); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "stage2_fromscratch_rank.png", dpi=140, bbox_inches="tight", facecolor="white")
print(f"\nwrote {OUT}/stage2_fromscratch_rank.png, stage2_strategies.csv")
