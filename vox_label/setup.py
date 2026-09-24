"""Choosing a campaign before labeling starts: what exists, what a grid would buy,
and binding one to the server.

The GUI used to take a single frozen campaign on the command line. This adds the step
in front of it: pick the dataset, resume a campaign that is part way through, or freeze
a new one over an existing pool with the grid parameters entered by hand.

Creating is still `campaign.create`, which refuses to overwrite. Nothing here can edit
a frozen spec; a new campaign means a new directory.
"""
from pathlib import Path

from vox_label import discover
from vox_label.campaign import Campaign, create
from vox_label.candidates import load_pool
from vox_label.exact_grid import c_of_j, even_grid, half_width, plan_n
from vox_label.render import make_source
from vox_label.rules import accept_fraction, default_rules, resolve

# Display-only settings. Free to change at any point, including mid-campaign: they
# decide what the annotator sees, not what is estimated.
VIEW_DEFAULTS = {
    "pad": 0.35,
    "disp_w": 900,
    "jpeg_q": 72,
    "reveal_every": 500,
    "f_lo_khz": 5.0,
    "f_hi_khz": 60.0,
    "nyquist_khz": 62.5,
    "prefix": None,
    "backend": "auto",
}

# Grid parameters. Frozen into campaign.json at creation and never editable after.
GRID_DEFAULTS = {
    "n_max": 6000,
    "J": 12,
    "alpha": 0.05,
    "alpha_per_rule": False,
    "seed": 20260911,
    "recall_stream_max": None,
}

TOLERANCES = [0.02, 0.03, 0.04, 0.05]
P_REAL_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

_pool_cache = {}


def cached_pool(pool_csv, detectors):
    key = (str(pool_csv), tuple(detectors))
    if key not in _pool_cache:
        _pool_cache[key] = load_pool(pool_csv, detectors)
    return _pool_cache[key]


_VIEW_TYPES = {"pad": float, "disp_w": int, "jpeg_q": int, "reveal_every": int,
               "f_lo_khz": float, "f_hi_khz": float, "nyquist_khz": float}


def view_for(dataset_info, overrides=None):
    """View settings for a dataset: defaults, then what was read off the data."""
    view = dict(VIEW_DEFAULTS)
    if dataset_info:
        if dataset_info.get("nyquist_khz"):
            view["nyquist_khz"] = dataset_info["nyquist_khz"]
            view["f_hi_khz"] = min(view["f_hi_khz"], dataset_info["nyquist_khz"])
        view["prefix"] = dataset_info.get("prefix")
        view["backend"] = dataset_info.get("backend", "auto")
    for k, v in (overrides or {}).items():
        if v is None or k not in view:
            continue
        view[k] = _VIEW_TYPES[k](v) if k in _VIEW_TYPES else v
    # A band above Nyquist shows blank rows rather than failing, so clamp it here.
    view["f_hi_khz"] = min(view["f_hi_khz"], view["nyquist_khz"])
    view["f_lo_khz"] = min(view["f_lo_khz"], view["f_hi_khz"] - 0.1)
    return view


def options(view_overrides=None):
    """Everything the setup page needs to render its form."""
    ds = discover.datasets()
    known = [d["name"] for d in ds]
    return {
        "datasets": ds,
        "campaigns": discover.campaigns(known_datasets=known),
        "pools": discover.pools(),
        "grid_defaults": dict(GRID_DEFAULTS),
        "view_defaults": {d["name"]: view_for(d, view_overrides) for d in ds},
        "tolerances": TOLERANCES,
    }


def _validated_rules(rules, detectors):
    for r in rules:
        resolve(r, detectors)  # raises ValueError on an unknown rule name
    return rules


def plan(pool_csv, detectors, rules=None, n_max=6000, J=12, alpha=0.05,
         alpha_per_rule=False, recall_stream_max=None, p_real=None):
    """What a grid would buy, without writing anything.

    The same numbers `make_campaign.py --plan-only` prints: the committed checkpoints,
    each rule's f_V and the half-width n_max buys it, and the budget each tolerance
    would cost. J is not editable later, so this is the moment to look.
    """
    detectors = list(detectors)
    rows = cached_pool(pool_csv, detectors)
    rules = _validated_rules(list(rules) if rules else default_rules(detectors),
                             detectors)
    eff_alpha = alpha / len(rules) if alpha_per_rule else alpha
    recall_max = recall_stream_max or n_max

    precision = []
    for r in rules:
        f = accept_fraction(resolve(r, detectors), rows)
        precision.append({
            "rule": r,
            "f_v": round(f, 4),
            "grid": even_grid(max(J, int(f * n_max)), J),
            "stream_at_n_max": int(f * n_max),
            "half_width_pts": round(100 * half_width(0.25, f, n_max, J), 2),
        })

    recall = [{
        "p_real": px,
        "stream_at_n_max": int(px * n_max),
        "half_width_pts": round(100 * half_width(0.25, px, n_max, J), 2),
    } for px in ([p_real] if p_real else P_REAL_GRID)]

    fractions = sorted({round(p["f_v"], 3) for p in precision} | ({p_real} if p_real else set()))
    budget = [{"f": f, "n": [round(plan_n(0.25, f, t, J)) for t in TOLERANCES]}
              for f in fractions if f > 0]

    return {
        "n_candidates": len(rows),
        "detectors": detectors,
        "rules": rules,
        "n_max": n_max,
        "J": J,
        "alpha": alpha,
        "alpha_effective": eff_alpha,
        "c_of_j": round(c_of_j(J), 3),
        "recall_grid": even_grid(recall_max, J),
        "precision": precision,
        "recall": recall,
        "budget": budget,
        "tolerances": TOLERANCES,
    }


def freeze(out_dir, pool_csv, detectors, dataset, rules=None, **grid):
    """Write a new campaign.json. Refuses an existing spec, as `create` always has."""
    params = dict(GRID_DEFAULTS)
    params.update({k: v for k, v in grid.items() if k in GRID_DEFAULTS})
    detectors = list(detectors)
    rules = _validated_rules(list(rules) if rules else default_rules(detectors),
                             detectors)
    if not Path(pool_csv).exists():
        raise FileNotFoundError(f"pool {pool_csv} does not exist")
    pool_dets = discover.pool_detectors(pool_csv)
    unknown = [d for d in detectors if d not in pool_dets]
    if unknown:
        raise ValueError(f"pool {pool_csv} has no columns for {unknown}. "
                         f"It was built with {pool_dets}.")
    if params["n_max"] < params["J"]:
        raise ValueError(f"n_max ({params['n_max']}) must be at least J ({params['J']})")
    c_of_j(params["J"])  # raises outside the tabulated range
    path, spec = create(out_dir, pool_csv, detectors, rules=rules, dataset=dataset,
                        **params)
    return path, spec


def build_source(dataset, backend="auto", prefix=None):
    """Spectrogram backend for a dataset. `prefix` applies to chunk PNGs only.

    The root follows the resolved backend rather than being passed in: forcing png on
    a dataset stored as HDF5 would otherwise look for chunks under the h5 root and
    find none.
    """
    resolved = backend
    if backend == "auto":
        resolved = "h5" if Path(discover.H5_ROOT, dataset).exists() else "png"
    root = discover.H5_ROOT if resolved == "h5" else discover.PNG_ROOT
    kw = {"prefix": prefix} if resolved == "png" and prefix else {}
    return make_source(root, dataset, kind=resolved, **kw)


def open_campaign(campaign_dir, dataset=None, view=None, verify=True):
    """Load a campaign and its source, ready to hand to a LabelService.

    Labels are replayed from labels.jsonl by the Campaign itself, so a half-finished
    campaign resumes at the next unlabeled candidate with no extra step.
    """
    campaign = Campaign(campaign_dir)
    if verify:
        campaign.verify_pool()
    dataset = dataset or campaign.dataset
    if not dataset:
        raise ValueError(
            f"{campaign_dir} does not record which dataset it annotates (it was frozen "
            f"before that field existed). Pick one on the setup page.")
    info = next((d for d in discover.datasets() if d["name"] == dataset), None)
    settings = view_for(info, {**campaign.load_view(), **(view or {})})
    source = build_source(dataset, settings["backend"], settings.get("prefix"))
    return campaign, source, dataset, settings
