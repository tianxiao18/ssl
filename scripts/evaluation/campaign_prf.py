"""Precision / recall / F1 per detector from vox_label campaigns, computed exactly as the GUI does
(LabelService.live = anytime-valid CS at the current n; compute_report = exact-grid interval at the last
crossed checkpoint), plus one quantitative bar figure per dataset -> outputs/eval/campaign_prf/.

Only labeled rows of each pool are read, so the 2.4 GB dryad pool fits in the session's memory."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vox_label.campaign import Campaign
from vox_label.rules import resolve
from vox_label.server import LabelService

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = {"gerbil_ssl": "gerbil_ssl_test_k7", "dryad": "dryad_gerbil_full_k7_traces",
             "mongolia_audiomoth": "mongolia_audiomoth_k6", "mongolia_k4_hp15k": "mongolia_audiomoth_k4_hp15k"}
NAME = {"squeakout": "squeakout", "ridge": "ridge", "ridge_raw": "ridge", "das_yolo": "das_yolo", "das": "das",
        "sam3_best": "sam3 zero-shot (box)", "sam3_h5": "sam3 zero-shot (box)",
        "sam3_traces_t037": "sam3 zero-shot (text)", "sam3_traces_dryad_full": "sam3 zero-shot (text)",
        "sam3_ft_traces_gold_v2": "sam3 gold ft", "sam3_ft_traces_gold_v2_dryad": "sam3 gold ft",
        "sam3_ft_traces_dasyolo_labeled_v2": "sam3 das_yolo ft", "sam3_ft_traces_dasyolo_labeled_v2_dryad": "sam3 das_yolo ft"}
NAME.update({k + "_hp15k": v for k, v in NAME.items()})
ORDER = ["squeakout", "ridge", "das_yolo", "das", "sam3 zero-shot (box)", "sam3 zero-shot (text)", "sam3 gold ft", "sam3 das_yolo ft"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#8a8f98"]  # as compare_sam3_ft; das takes the das_yolo slot


def light_campaign(name):
    """A Campaign whose pool holds only the labeled candidates (all the estimates touch)."""
    c = Campaign.__new__(Campaign)
    c.dir = ROOT / "outputs/label_campaigns" / name
    c.spec = json.loads((c.dir / "campaign.json").read_text())
    c.detectors, c.rules, c.alpha = c.spec["detectors"], c.spec["rules"], c.spec["alpha_effective"]
    c.dataset, c.order = c.spec.get("dataset"), c.spec["order"]
    c.labels_path = c.dir / "labels.jsonl"
    labels = c.load_labels()
    c.pool = {}
    with open(ROOT / c.spec["pool_csv"], newline="") as fh:
        for r in csv.DictReader(fh):
            cid = int(r["cand_id"])
            if cid in labels:
                c.pool[cid] = {"cand_id": cid, "clip": r["clip"], "z": tuple(int(r[f"z_{d}"]) for d in c.detectors)}
    c._rule_fns = {n: resolve(n, c.detectors) for n in c.rules}
    return c, labels


def estimates(ds, name):
    c, labels = light_campaign(name)
    svc = SimpleNamespace(c=c, labels=labels, _gamma={})
    svc.gamma_cache = lambda grid: LabelService.gamma_cache(svc, grid)
    live = pd.DataFrame(LabelService.live(svc)).assign(kind="live")
    grid = pd.DataFrame(LabelService.compute_report(svc)).assign(kind="grid")
    out = pd.concat([live, grid], ignore_index=True)
    return out.assign(dataset=ds, campaign=name, n_labels=len(labels), model=out.rule.map(NAME))


def quantitative(df, ds, out_png):
    """Bars per model for precision / recall / F1 with the GUI's live 95% CS, styled as compare_sam3_ft.quantitative."""
    d = df[(df.kind == "live") & (df.dataset == ds)]
    models = sorted(d.model.unique(), key=lambda m: ORDER.index(m))
    fig, ax = plt.subplots(figsize=(6.2, 3.9), facecolor="#ffffff")
    x, w = np.arange(3), 0.8 / len(models)
    for k, m in enumerate(models):
        r = d[d.model == m].set_index("target").loc[["precision", "recall", "f1"]]
        v, lo, hi = r.estimate.to_numpy(float), r.lo.to_numpy(float), r.hi.to_numpy(float)
        xs = x + (k - (len(models) - 1) / 2) * w
        ax.bar(xs, v, w * 0.92, color=COLORS[ORDER.index(m)], label=m)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#1a1a19", lw=0.8, capsize=2)
    ax.set_xticks(x, ["precision", "recall", "F1"])
    ax.set(ylim=(0, 1.12), title=f"detection vs vox labeler labels (n = {d.n_labels.iloc[0]})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out_png, dpi=150, facecolor="#ffffff")
    plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=ROOT / "outputs/eval/campaign_prf", type=Path)
    ap.add_argument("--datasets", default="gerbil_ssl,dryad", help=f"comma-separated subset of {list(CAMPAIGNS)}")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dss = a.datasets.split(",")
    df = pd.concat([estimates(ds, CAMPAIGNS[ds]) for ds in dss], ignore_index=True)
    df.to_csv(a.out / "prf.csv", index=False)
    cols = ["dataset", "model", "target", "kind", "n", "x", "estimate", "lo", "hi", "checkpoint", "available"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    for ds in dss:
        quantitative(df, ds, a.out / f"quantitative_{ds}.png")


if __name__ == "__main__":
    main()
