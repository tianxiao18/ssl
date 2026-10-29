"""Compare VAEs trained on DAS vs SAM3 segments (labels_to_ava_specs.py + train_ava_ab.py --arms das,sam3).

Human ground truth: the dryad_gerbil_full_sub100_k7 campaign (same 100 recordings). Each segment takes the label of
the labeled candidate it overlaps most in time (1 = real call, 0 = not), else it stays unlabeled.
Per arm: segments, duration, share of labeled segments that are real calls, and how well the latent space separates
real from not (5-fold logistic regression AUC on the labeled latents). Writes <out>/label_compare.json,
label_compare.png (UMAP per arm coloured by the human label, plus duration histograms).

    python scripts/vae/compare_label_sources.py --data /mnt/ceph/users/the10/ava/dryad_das_vs_sam3 \
        --compare outputs/ava/dryad_das_vs_sam3
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COL = {1: "#2a78d6", 0: "#d6402a", -1: "#c9ccd1"}


def campaign_labels(cdir):
    lab = {}
    for line in open(cdir / "labels.jsonl"):
        r = json.loads(line)
        if r.get("retracted"):
            lab.pop(r["cand_id"], None)
        elif r.get("label") in (0, 1):
            lab[r["cand_id"]] = int(r["label"])
    c = pd.read_csv(cdir / "candidates.csv", usecols=["cand_id", "clip", "t_start", "t_end"])
    c = c[c.cand_id.isin(lab)].copy()
    c["label"] = c.cand_id.map(lab)
    by = defaultdict(list)
    for r in c.itertuples():
        by[r.clip].append((r.t_start, r.t_end, r.label))
    return by


def label_segments(seg, by):
    out = np.full(len(seg), -1)
    for i, r in enumerate(seg.itertuples()):
        best = 0.0
        for s, e, l in by.get(r.recording, ()):
            ov = min(e, r.offset) - max(s, r.onset)
            if ov > best:
                best, out[i] = ov, l
    return out


def cv_auc(z, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if min(np.bincount(y, minlength=2)) < 5:
        return None
    k = min(5, int(min(np.bincount(y))))
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), z, y,
                          cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dir with <arm>/segments.csv")
    ap.add_argument("--compare", type=Path, required=True, help="train_ava_ab.py --out (latent_<arm>.npy)")
    ap.add_argument("--arms", default="das,sam3")
    ap.add_argument("--campaign", default="outputs/label_campaigns/dryad_gerbil_full_sub100_k7")
    ap.add_argument("--umap-max", type=int, default=30000, help="unlabeled points shown per arm (all labeled ones are)")
    a = ap.parse_args()
    import umap
    arms = a.arms.split(",")
    by = campaign_labels(ROOT / a.campaign)
    res, emb = {}, {}
    for arm in arms:
        seg = pd.read_csv(a.data / arm / "segments.csv")
        z = np.load(a.compare / f"latent_{arm}.npy")
        n = min(len(seg), len(z))
        seg, z = seg.iloc[:n], z[:n]
        y = label_segments(seg, by)
        lab = y >= 0
        rng = np.random.default_rng(0)
        show = np.where(lab)[0].tolist() + rng.choice(np.where(~lab)[0], min(a.umap_max, (~lab).sum()), replace=False).tolist()
        emb[arm] = (umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(z[show]), y[show],
                    (seg.offset - seg.onset).values * 1000)
        res[arm] = {"segments": int(n), "median_dur_ms": round(float(np.median(seg.offset - seg.onset)) * 1000, 1),
                    "labeled": int(lab.sum()), "labeled_real": int((y == 1).sum()), "labeled_not": int((y == 0).sum()),
                    "share_real": round(float((y[lab] == 1).mean()), 3) if lab.any() else None,
                    "latent_auc_real_vs_not": cv_auc(z[lab], y[lab])}
        print(arm, res[arm], flush=True)
    (a.compare / "label_compare.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(1, len(arms) + 1, figsize=(5.2 * (len(arms) + 1), 4.6), facecolor="#ffffff")
    for i, arm in enumerate(arms):
        e, y, _ = emb[arm]
        for l, name, s in ((-1, "unlabeled", 2), (0, "not a call", 9), (1, "real call", 9)):
            m = y == l
            ax[i].scatter(e[m, 0], e[m, 1], s=s, color=COL[l], alpha=0.35 if l < 0 else 0.9, lw=0, label=f"{name} ({m.sum()})")
        r = res[arm]
        auc = r["latent_auc_real_vs_not"]
        ax[i].set(title=f"{arm}: {r['share_real']:.0%} of labeled are calls, latent AUC {auc:.2f}" if auc else arm,
                  xticks=[], yticks=[])
        ax[i].legend(frameon=False, fontsize=8, markerscale=2)
    for arm, c in zip(arms, ("#eb6834", "#2a78d6")):
        ax[-1].hist(np.clip(emb[arm][2], 0, 300), bins=np.linspace(0, 300, 31), density=True, histtype="stepfilled",
                    alpha=0.35, color=c, edgecolor=c, label=arm)
    ax[-1].set(xlabel="segment duration (ms)", ylabel="density", title="Segment durations")
    ax[-1].legend(frameon=False)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.compare / "label_compare.png", dpi=130, facecolor="#ffffff")
    print(f"-> {a.compare}")


if __name__ == "__main__":
    main()
