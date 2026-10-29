"""Recording timeline per colony and microphone: every file in the Mongolia AudioMoth dataset, sampled ones highlighted.

    python scripts/dataset_prep/plot_mongolia_coverage.py
    -> outputs/mongolia_audiomoth/coverage.png, coverage.csv
"""
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TZ = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--durations", default=ROOT / "outputs/mongolia_audiomoth/durations.txt", type=Path)
    ap.add_argument("--sample", default=ROOT / "outputs/mongolia_audiomoth/all/sample.txt", type=Path)
    ap.add_argument("--out", default=ROOT / "outputs/mongolia_audiomoth", type=Path)
    a = ap.parse_args()
    sample = {l.rstrip("\n") for l in open(a.sample) if l.strip()}
    rows = []
    for line in open(a.durations):
        k, v = line.rstrip("\n").split("\t")
        m = re.search(r"_(\d{8}_\d{6})$", k)
        if not m:
            continue
        dev = next(re.match(r"(?:AM)?(PK\d+|[A-Z]|\d+)_", p)[1] for p in k.split("/") if re.match(r"(AM|PK)", p))
        rows.append(dict(rec=k, colony=k.split("/")[0].replace("Colony_", ""), mic=dev, hours=float(v) / 3600,
                         start=datetime.strptime(m[1], "%Y%m%d_%H%M%S") + timedelta(hours=TZ), sampled=k in sample))
    d = pd.DataFrame(rows)
    summ = d.groupby("colony").agg(mics=("mic", "nunique"), files=("rec", "size"), hours=("hours", "sum"),
                                   sampled_files=("sampled", "sum"),
                                   sampled_hours=("hours", lambda h: h[d.loc[h.index, "sampled"]].sum()),
                                   first=("start", "min"), last=("start", "max"))
    summ.round(1).to_csv(a.out / "coverage.csv")
    print(summ.round(1).to_string())

    lanes = []
    for col, g in d.groupby("colony"):
        mics = sorted(g.mic.unique(), key=lambda x: (not x.isdigit(), x.zfill(3)))
        lanes += [(col, m) for m in mics]
    y = {k: i for i, k in enumerate(lanes[::-1])}
    fig, ax = plt.subplots(figsize=(11, 0.22 * len(lanes) + 1.2), facecolor="#ffffff")
    for r in d.itertuples():
        ax.barh(y[(r.colony, r.mic)], timedelta(hours=r.hours), left=r.start, height=0.7,
                color="#2a78d6" if r.sampled else "#c9ccd1", lw=0)
    ax.set_yticks(list(y.values()), [m for _, m in y], fontsize=6.5)
    x0 = d.start.min() - timedelta(days=0.3)
    prev = None
    for (col, m), i in sorted(y.items(), key=lambda kv: -kv[1]):
        if col != prev:
            s = summ.loc[col]
            n = sum(c == col for c, _ in lanes)
            mid = i - (n - 1) / 2
            ax.annotate(f"{col}\n{int(s.mics)} mics\n{s.sampled_hours:.0f} / {s.hours:.0f} h", (0, mid),
                        xycoords=("axes fraction", "data"), xytext=(-28, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8)
            if prev is not None:
                ax.axhline(i + 0.5, color="#8a8f98", lw=0.6)
            prev = col
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel(f"date (local, UTC+{TZ})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color="#c9ccd1", label="in dataset"), Patch(color="#2a78d6", label="in analyzed sample")],
              fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.out / "coverage.png", dpi=140, facecolor="#ffffff")
    print("->", a.out / "coverage.png")


if __name__ == "__main__":
    main()
