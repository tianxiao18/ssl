"""Pick zero-shot SAM3 text prompt + score threshold on the train recordings, report once on the test recordings.

Detection is scored as in compare_sam3_ft.py (evaluate.py's score_combined: both mics pooled, any time overlap,
predictions centred in the annotator's labeled span), each annotator separately. For each prompt the threshold
maximizing the mean train F1 of the two annotators is kept; the prompt with the best such train F1 is the pick.
Operating points whose predictions merge into fewer than MIN_EVENT_RATIO x the GT events are not eligible:
score_combined counts precision on merged events, so a few long boxes spanning everything (e.g. "line" boxing
the noise bands) would otherwise score near-perfect F1.
Inputs: <pred-root>/<prompt slug>/gerbil_ssl/<rel>/coco_ch_<ch>.json (sam3_prompt_select.sbatch, threshold 0.05).
--ref adds fixed models (name=<pred base>[@score]) to the test table and figure.

    python scripts/evaluation/select_prompt_threshold.py --prompts traces "white traces" trace squiggle \
        --ref sam3-text-boxft=outputs/sam3_ft_box2mask/gerbil_ssl --out outputs/eval/sam3_prompt_select
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_sam3_ft import COLORS, GT_ROOT, bootstrap, load_gt, prf, spec, write_csv  # noqa: E402
from vox_tracer.scoring import read_pred_boxes, score_combined  # noqa: E402

MIN_EVENT_RATIO = 0.5
THRS = np.round(np.arange(0.05, 0.96, 0.01), 2)
ALL_PROMPTS = ["traces", "white traces", "trace", "squiggle", "white trace", "line", "white line", "curve",
               "white curve", "streak", "chirp", "call", "vocalization", "ultrasonic vocalization", "whistle"]


def boot_sd(rows, n=2000, seed=0):
    """Recording-bootstrap SD of F1."""
    m = np.array(list(rows.values()))
    rng = np.random.default_rng(seed)
    return float(np.nanstd([prf(m[rng.integers(len(m), size=len(m))].sum(0))[2] for _ in range(n)]))


def rows_at(boxes, gt, thr):
    """({rec: [tp, pred_tp, fp, fn]}, predicted events / GT events) at one threshold (as compare_sam3_ft.detect_rows)."""
    out, n_pred, n_gt = {}, 0, 0
    for rel, (events, spans) in gt.items():
        if rel not in boxes:
            continue
        b = [x for x in boxes[rel] if (x[2] is None or x[2] >= thr) and any(s <= (x[0] + x[1]) / 2 <= e for s, e in spans)]
        r = score_combined(events, b)
        out[rel] = np.array([r["tp"], r["pred_tp"], r["fp"], r["fn"]])
        n_pred, n_gt = n_pred + r["n_pred"], n_gt + len(events)
    return out, n_pred / max(n_gt, 1)


def thr_of_best(picks, prompt):
    return next(r["threshold"] for r in picks if r["prompt"] == prompt)


def load_boxes(base, recs, channels):
    return {r: read_pred_boxes(Path(base) / r, channels) for r in recs if (Path(base) / r).exists()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", required=True, help="text prompts, or 'all' (the 15 of the sweep)")
    ap.add_argument("--pred-root", default="outputs/sam3_prompt_select")
    ap.add_argument("--train-recs", default="outputs/sam3_prompt_select/train_recordings.txt")
    ap.add_argument("--test-recs", default="outputs/sam3_prompt_select/test_recordings.txt")
    ap.add_argument("--ref", nargs="*", default=[], help="name=<pred base>[@score] fixed models for the test table")
    ap.add_argument("--coverage", default="outputs/pseudo_boxes/gerbil_ssl_corrected_gt_coverage.csv")
    ap.add_argument("--annotators", default="niegil,tianxiao")
    ap.add_argument("--channels", default="118,35")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.prompts == ["all"]:
        a.prompts = ALL_PROMPTS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cov, anns = pd.read_csv(a.coverage), a.annotators.split(",")
    channels = [int(c) for c in a.channels.split(",")]
    recs = {k: {x.strip() for x in open(f) if x.strip()} for k, f in (("train", a.train_recs), ("test", a.test_recs))}
    gts = {k: {ann: load_gt(GT_ROOT, ann, r, cov) for ann in anns} for k, r in recs.items()}

    curves, picks, test_rows = {}, [], []
    for p in a.prompts:
        base = Path(a.pred_root) / p.replace(" ", "_") / "gerbil_ssl"
        tr = load_boxes(base, recs["train"], channels)
        res = {ann: [rows_at(tr, gts["train"][ann], t) for t in THRS] for ann in anns}
        curves[p] = {ann: np.array([np.nan_to_num(prf(np.sum(list(r.values()), axis=0))[2]) for r, _ in res[ann]])
                     for ann in anns}
        ok = np.all([[q >= MIN_EVENT_RATIO for _, q in res[ann]] for ann in anns], axis=0)
        mean = np.mean([curves[p][ann] for ann in anns], axis=0)
        k = int(np.argmax(np.where(ok, mean, -1)))
        picks.append({"prompt": p, "threshold": THRS[k], "train_f1_mean": mean[k] if ok[k] else 0.0,
                      "collapsed": not ok.any(), "train_recordings": len(tr),
                      **{f"train_f1_{ann}": curves[p][ann][k] for ann in anns}})
        print(f"{p:24s} train-picked threshold {THRS[k]:.2f}  mean train F1 {mean[k]:.3f}"
              + ("  (no eligible threshold: collapsed)" if not ok.any() else ""))
    write_csv(out / "selection.csv", picks)
    best = max(picks, key=lambda r: r["train_f1_mean"])["prompt"]  # collapsed prompts have train_f1_mean 0
    print(f"picked prompt (best train F1): {best}")
    (out / "picked_prompt.txt").write_text(f"{best}\t{thr_of_best(picks, best):.2f}\n")

    models = [(f"text: {r['prompt']}", Path(a.pred_root) / r["prompt"].replace(" ", "_") / "gerbil_ssl", r["threshold"])
              for r in picks] + [spec(s, 0.5) for s in a.ref]
    for nm, base, thr in models:
        te = load_boxes(base, recs["test"], channels)
        for ann in anns:
            rows, ratio = rows_at(te, gts["test"][ann], thr)
            v, (lo, hi) = prf(np.sum(list(rows.values()), axis=0)), bootstrap(rows)
            test_rows.append({"model": nm, "score": thr, "gt": ann, "recordings": len(rows),
                              "precision": v[0], "recall": v[1], "f1": np.nan_to_num(v[2]), "f1_lo": lo[2],
                              "f1_hi": hi[2], "f1_sd": boot_sd(rows),
                              "event_ratio": ratio})
    write_csv(out / "test.csv", test_rows)
    for r in test_rows:
        print(f"{r['model']:24s} @{r['score']:.2f} {r['gt']:9s} P {r['precision']:.3f} R {r['recall']:.3f} "
              f"F1 {r['f1']:.3f} [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]")

    # top: train curves of the 4 best prompts per annotator; bottom: test F1 +- bootstrap SD of every model
    top = [r["prompt"] for r in sorted(picks, key=lambda r: -r["train_f1_mean"])[:4]]
    thr_of = {r["prompt"]: r["threshold"] for r in picks}
    fig = plt.figure(figsize=(16, 8.5), facecolor="#ffffff")
    gs = fig.add_gridspec(2, len(anns), height_ratios=[1, 1.1])
    axes = [fig.add_subplot(gs[0, k]) for k in range(len(anns))]
    for ax, ann in zip(axes, anns):
        for p, c in zip(top, COLORS):
            k = int(np.where(THRS == thr_of[p])[0][0])
            ax.plot(THRS, curves[p][ann], color=c, lw=2, label=f"{p}  @ {thr_of[p]:.2f}")
            ax.plot(THRS[k], curves[p][ann][k], "o", ms=8, color=c, mec="#ffffff", mew=2)
        ax.set_title(f"Train F1, {ann} (best 4 prompts)", fontsize=11)
        ax.set_xlabel("score threshold")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8, loc="lower center", title="picked threshold", title_fontsize=8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    axes[0].set_ylabel("F1")
    bx = fig.add_subplot(gs[1, :])
    order = [f"text: {r['prompt']}" for r in sorted(picks, key=lambda r: -r["train_f1_mean"])]
    names = order + [n for n in dict.fromkeys(r["model"] for r in test_rows) if n not in order]
    w = 0.8 / len(anns)
    for j, ann in enumerate(anns):
        rs = {r["model"]: r for r in test_rows if r["gt"] == ann}
        x = np.arange(len(names)) + (j - (len(anns) - 1) / 2) * w
        bx.bar(x, [rs[n]["f1"] for n in names], w * 0.92, yerr=[rs[n]["f1_sd"] for n in names],
               color=["#52514e", "#b5b4ad"][j % 2], error_kw={"lw": 1, "ecolor": "#0b0b0b"}, label=ann,
               hatch=["//" if rs[n]["event_ratio"] < MIN_EVENT_RATIO else "" for n in names], edgecolor="#ffffff")
    star = {r["model"] for r in test_rows if r["event_ratio"] < MIN_EVENT_RATIO}
    labels = [f"{n.removeprefix('text: ')}{'*' if n in star else ''} @{rs[n]['score']:.2f}" for n in names]
    bx.set_xticks(np.arange(len(names)), labels, fontsize=8, rotation=30, ha="right", rotation_mode="anchor")
    bx.get_xticklabels()[0].set_fontweight("bold")
    bx.axvline(len(order) - 0.5, color="#b5b4ad", lw=1, ls="--")
    bx.set_ylim(0, 1.05)
    bx.set_ylabel("test F1")
    bx.set_title(f"Test F1 ({len(recs['test'])} recordings), threshold picked on train, error bar = bootstrap SD; "
                 "prompts sorted by train F1; * hatched = predictions merge into < "
                 f"{MIN_EVENT_RATIO:.0%} of the GT events", fontsize=10)
    bx.legend(frameon=False, fontsize=9, loc="upper right")
    bx.grid(axis="y", color="#e6e5e0", lw=0.8)
    for ax in axes + [bx]:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "prompt_selection.png", dpi=150, facecolor="#ffffff")
    print(f"wrote {out}/selection.csv, test.csv, prompt_selection.png")


if __name__ == "__main__":
    main()
