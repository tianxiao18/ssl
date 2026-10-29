"""Stratified AudioMoth sample over every colony, sized to a GPU-hour budget, reusing what is already inferred.

Strata = colony x 6 h start-time bin, as sample_mongolia_am.py. Each stratum takes its recordings from
outputs/mongolia_audiomoth/campaign_sample.txt first (the 36 already run with every detector), then a seeded
shuffle of the rest; k per stratum is the largest that fits --gpu-hours, where a recording costs its duration x the
GPU rate of each detector whose output it still lacks. Rates are GPU-h per audio hour measured on 7306397 /
7282473 (A100/H100). Writes to --out:
    sample.txt          the picked recordings
    specs_<i>.txt       picks without a spectrogram .h5 (~24 h audio each)  -> mongolia_all_specs.sbatch
    gpu_tasks.tsv       one line per GPU task: detector, chunk file          -> mongolia_all_detectors.sbatch
    ridge_<i>.txt       picks without ridge output (24 per chunk)            -> mongolia_all_ridge.sbatch
    summary.json

    python scripts/dataset_prep/sample_mongolia_all.py --gpu-hours 300 --out outputs/mongolia_audiomoth/all
    python scripts/dataset_prep/sample_mongolia_all.py --all --detectors gold_ft --out outputs/mongolia_audiomoth/full
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AM = ROOT / "outputs/mongolia_audiomoth"
GPU = {"sam3_best": ("outputs/sam3_best", 0.159), "zs_traces": ("outputs/sam3_traces_mongolia_am", 0.073),
       "gold_ft": ("outputs/sam3_ft_traces_gold_v2", 0.077), "dasyolo_ft": ("outputs/sam3_ft_traces_dasyolo_labeled_v2", 0.088),
       "squeakout": ("outputs/squeakout", 0.010)}


def done(out_dir, rec):
    return (ROOT / out_dir / "mongolia_audiomoth" / rec / "coco_ch_0.json").exists()


def chunks(recs, dur, size_h):
    """Consecutive groups of about size_h audio hours."""
    out, cur, h = [], [], 0.0
    for r in recs:
        cur.append(r)
        h += dur[r]
        if h >= size_h:
            out.append(cur)
            cur, h = [], 0.0
    return out + ([cur] if cur else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-hours", type=float, default=300)
    ap.add_argument("--task-hours", type=float, default=3.0, help="target GPU hours per array task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every timed recording, no budget")
    ap.add_argument("--detectors", default=",".join(list(GPU) + ["ridge"]), help="subset of GPU detectors + ridge")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    dur = {r: float(s) / 3600 for r, s in (l.split("\t") for l in open(AM / "durations.txt").read().splitlines())}
    keep = {l.strip() for l in open(AM / "campaign_sample.txt") if l.strip()}
    strata, untimed = defaultdict(list), []
    for r in dur:
        m = re.search(r"_(\d{6})$", r)  # stem ends in the start time hhmmss
        if m:
            strata[(r.split("/")[0], int(m.group(1)[:2]) // 6)].append(r)
        else:
            untimed.append(r)
    order = {}
    for k, rs in strata.items():
        rest = sorted(set(rs) - keep)
        random.Random(f"{a.seed}-{k[0]}-{k[1]}").shuffle(rest)
        order[k] = sorted(set(rs) & keep) + rest
    gpu = {d: v for d, v in GPU.items() if d in a.detectors.split(",")}
    todo = {d: {r for r in dur if not done(p, r)} for d, (p, _) in gpu.items()}
    cost = lambda picks: sum(dur[r] * rate for d, (_, rate) in gpu.items() for r in picks if r in todo[d])
    pick_k = lambda k: sorted(r for k_, rs in order.items() for r in rs[:max(k, sum(x in keep for x in rs))])
    k = 1
    while k < max(map(len, order.values())) and cost(pick_k(k + 1)) <= a.gpu_hours:
        k += 1
    picks = sorted(r for rs in order.values() for r in rs) if a.all else pick_k(k)
    a.out.mkdir(parents=True)
    (a.out / "sample.txt").write_text("".join(f"{r}\n" for r in picks))
    spec = ROOT / "outputs/spectrograms_h5/mongolia_audiomoth"
    no_spec = [r for r in picks if not any((spec / r).glob("mic_0_*.h5"))]
    for i, c in enumerate(chunks(no_spec, dur, 24)):
        (a.out / f"specs_{i}.txt").write_text("".join(f"{r}\n" for r in c))
    lines, gpu_h = [], {}
    for d, (_, rate) in gpu.items():
        rs = [r for r in picks if r in todo[d]]
        gpu_h[d] = round(sum(dur[r] for r in rs) * rate, 1)
        for i, c in enumerate(chunks(rs, dur, a.task_hours / rate)):
            f = a.out / f"{d}_{i}.txt"
            f.write_text("".join(f"{r}\n" for r in c))
            lines.append(f"{d}\t{os.path.relpath(f, ROOT)}")
    (a.out / "gpu_tasks.tsv").write_text("".join(f"{x}\n" for x in lines))
    no_ridge = [r for r in picks if not done("outputs/ridge", r)] if "ridge" in a.detectors.split(",") else []
    n_ridge = 0
    for n_ridge, i in enumerate(range(0, len(no_ridge), 24), 1):
        (a.out / f"ridge_{n_ridge - 1}.txt").write_text("".join(f"{r}\n" for r in no_ridge[i:i + 24]))
    by_col = defaultdict(lambda: [0, 0.0])
    for r in picks:
        by_col[r.split("/")[0]][0] += 1
        by_col[r.split("/")[0]][1] += dur[r]
    n_spec = len(list(a.out.glob("specs_*.txt")))
    summary = {"per_stratum": k, "strata": len(order), "recordings": len(picks), "audio_h": round(sum(dur[r] for r in picks), 1),
               "kept_from_k6": len(keep & set(picks)), "untimed_left_out": untimed, "gpu_hours": gpu_h, "gpu_hours_total": round(sum(gpu_h.values()), 1),
               "gpu_tasks": len(lines), "spec_tasks": n_spec, "ridge_tasks": n_ridge,
               "by_colony": {c: {"recordings": n, "audio_h": round(h, 1)} for c, (n, h) in sorted(by_col.items())}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    d = os.path.relpath(a.out, ROOT)
    print(f"\nsubmit from the repo root (DIR={d}):"
          f"\n  sbatch --array=0-{max(n_spec - 1, 0)} --export=ALL,DIR={d} scripts/sbatch/mongolia_all_specs.sbatch"
          f"\n  sbatch --array=0-{len(lines) - 1} --dependency=afterok:<specs> --export=ALL,DIR={d} scripts/sbatch/mongolia_all_detectors.sbatch"
          f"\n  sbatch --array=0-{max(n_ridge - 1, 0)} --dependency=afterok:<specs> --export=ALL,DIR={d} scripts/sbatch/mongolia_all_ridge.sbatch"
          f"\n  sbatch --dependency=afterok:<detectors>:<ridge> --export=ALL,DIR={d} scripts/sbatch/build_campaign_mongolia_all.sbatch")


if __name__ == "__main__":
    main()
