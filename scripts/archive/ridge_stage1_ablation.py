"""
Is stage-1 component filtering actually needed, or can stage-2 do it alone?

Ablation: build ridge detections two ways over several sessions —
  (A) stage-1 ON  (current: area/aspect/freq component filter before morph-close)
  (B) stage-1 OFF (thresholded pixels -> morph-close -> every contour is a detection)
— then apply the SAME stage-2 filters to both and compare at the event level
(recording-level recall/precision via vox_tracer.scoring). Also reports raw
detection counts and pipeline runtime, since a key role of stage-1 is compute
and morph-close hygiene, not just precision.

All features here are geometric (no audio), so B's speckle explosion is cheap.

    python scripts/ridge_stage1_ablation.py --n-sessions 6
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts" / "viz"))
import ridge_pipeline_viz as rpv  # noqa: E402
from ridge_filter_experiment import discover_sessions, event_metrics  # noqa: E402
from spectral_features import bbox_to_time  # noqa: E402
from vox_tracer.spec import parse_spec_fname  # noqa: E402


def build(sessions, stage1, channels=(118, 35)):
    """Geometric detections for all sessions; returns (df, n_raw_total, seconds)."""
    p = dict(rpv.DEFAULTS)
    rows, n_raw, t0 = [], 0, time.time()
    for session, spec_dir, _, _, vox in sessions:
        for ch in channels:
            for png in sorted(spec_dir.glob(f"headmic_{ch}_*.png")):
                gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                H, W = gray.shape
                pc = dict(p)
                pc["freq_cutoff_row"] = int((H - 1) * (1.0 - p["freq_min"] / (p["sample_rate"] / 2.0)))
                res = rpv.run_pipeline(gray, pc, stage1=stage1)
                _, wt0, wt1 = parse_spec_fname(png.name)
                nyq = p["sample_rate"] / 2.0
                for comp, _, st in res["s2"]:
                    n_raw += 1
                    t_a, t_b = bbox_to_time(st["bx"], st["bw"], wt0, wt1, W)
                    bmask = comp > 0
                    rows.append(dict(
                        session=session, t0=t_a, t1=t_b,
                        label=rpv.label_tpfp(st["x0"], st["x1"], wt0, wt1, W, vox),
                        n_cols=st["n_cols"], sweep_frac=st["sweep_frac"],
                        area_frac=st["area_frac"],
                        f_high_khz=(H - 1 - st["by"]) / (H - 1) * nyq / 1e3,
                        ridge_score=float(res["response"][bmask].mean()) if bmask.any() else 0.0,
                    ))
    return pd.DataFrame(rows), n_raw, time.time() - t0


# stage-2 filters to apply identically to both variants (all geometric).
FILTERS = {
    "none (raw)":                 lambda d: pd.Series(True, index=d.index),
    "current stage-2":            lambda d: (d.area_frac <= 0.15) & (d.n_cols >= 5) & (d.sweep_frac >= 0.04),
    "n_cols>=9":                  lambda d: d.n_cols >= 9,
    "n_cols>=9 & f_high>=28kHz":  lambda d: (d.n_cols >= 9) & (d.f_high_khz >= 28),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=6)
    args = ap.parse_args()
    sessions = discover_sessions(args.n_sessions)
    print(f"{len(sessions)} sessions: {[s[0] for s in sessions]}\n")

    variants = {}
    for name, s1 in (("stage-1 ON", True), ("stage-1 OFF", False)):
        df, n_raw, secs = build(sessions, s1)
        variants[name] = df
        tp = (df.label == "tp").sum()
        print(f"[{name}] raw detections={n_raw:6d}  (TP {tp}/{len(df)})  pipeline {secs:5.1f}s")

    print(f"\n{'variant':13s} {'stage-2 filter':26s} {'recall':>7} {'prec':>7} {'F1':>7} {'FP':>6} {'kept':>7}")
    for vname, df in variants.items():
        for fname, fn in FILTERS.items():
            keep = fn(df).fillna(False)
            m = event_metrics(df, keep, sessions)
            print(f"{vname:13s} {fname:26s} {m['recall']:7.3f} {m['precision']:7.3f} "
                  f"{m['f1']:7.3f} {m['fp']:6d} {int(keep.sum()):7d}")


if __name__ == "__main__":
    main()
