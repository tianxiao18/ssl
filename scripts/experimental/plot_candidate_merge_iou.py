"""Single-linkage on any overlap vs single-linkage on IoU>=0.3, on six real cases.

Both lanes are single linkage; only the link predicate differs. With "overlap > 0" the
components are exactly the interval union, which is why that lane is labelled union.
Detections and candidates outside the illustrated case but inside the drawn window are
faded rather than dropped, so a call at the edge never looks undetected.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_label.candidates import find_clips, load_detections, group_detections
from vox_label.render import PngChunkSource, band_rows

TAU = 0.3
# --ghost draws detections/candidates that fall in the window but belong to a
# neighbouring case, faded. Without it the window is instead clamped so no such
# detection is inside it, which is what keeps an undrawn call off the edge.
GHOST_ON = "--ghost" in sys.argv
PAD = float(sys.argv[sys.argv.index("--pad") + 1]) if "--pad" in sys.argv else 0.05
OUT = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
       else "candidate_merge_current_vs_iou.png")
PX = 1.0 / 487
INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#ffffff", "#e5e4e0"
PROP = "#7b3fbf"
GHOST = 0.28
COLOR = {"sam3_best": "#2a78d6", "ridge": "#eb6834", "squeakout": "#1baf7a",
         "das_yolo": "#eda100", "sam3": "#e87ba4", "sam3_flatness_filtered": "#008300"}
SOURCE = {"gerbil_ssl": PngChunkSource("outputs/spectrograms", "gerbil_ssl", prefix="headmic"),
          "dryad_gerbil": PngChunkSource("outputs/spectrograms", "dryad_gerbil", prefix="mic")}
BAND, NYQ = (5.0, 60.0), 62.5
LANES = ("single-link  ov>0", f"single-link  IoU>={TAU}")


def iou(a, b):
    i = min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])
    if i <= 0:
        return 0.0
    return i / ((a["t1"] - a["t0"]) + (b["t1"] - b["t0"]) - i)


def components(dets, tau):
    """Exact single-linkage components under IoU >= tau (union-find, not a sweep).

    IoU >= tau implies positive overlap, so components refine the ov>0 groups and the
    quadratic pair scan only ever runs inside one of them.
    """
    out = []
    for grp in group_detections(dets, 0.0):
        parent = list(range(len(grp)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if iou(grp[i], grp[j]) >= tau:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        buckets = {}
        for i, d in enumerate(grp):
            buckets.setdefault(find(i), []).append(d)
        out.extend(buckets.values())
    return sorted(out, key=lambda g: min(x["t0"] for x in g))


def scan(dataset, detectors, n_clips):
    """Collect one example of each merge situation worth looking at."""
    best = {"glue": None, "chain": None, "touch": None, "onepx": None, "xchan_all": []}
    for clip in find_clips("outputs", dataset, detectors)[:n_clips]:
        raw = []
        for d in detectors:
            raw.extend(load_detections("outputs", dataset, clip, d))
        groups = group_detections(raw, 0.0)
        prev = None
        for g in groups:
            span = (min(x["t0"] for x in g), max(x["t1"] for x in g))
            rec = {"clip": clip, "dets": g, "span": span, "all": raw}
            for d in detectors:
                rest = [x for x in g if x["detector"] != d]
                if not rest or len(rest) == len(g):
                    continue
                n = len(group_detections(rest, 0.0))
                if n >= 2:
                    cand = dict(rec, glue_det=d, parts=n)
                    if best["glue"] is None or n > best["glue"]["parts"]:
                        best["glue"] = cand
            if len(g) >= 3 and not any(x["t0"] <= span[0] and x["t1"] >= span[1] for x in g):
                if best["chain"] is None or len(g) > len(best["chain"]["dets"]):
                    best["chain"] = rec
            order = sorted(g, key=lambda x: (x["t0"], x["t1"]))
            end = order[0]["t1"]
            for x in order[1:]:
                ov = end - x["t0"]
                if 0 < ov <= PX:
                    c = dict(rec, hinge=(x["t0"], end))
                    if best["onepx"] is None or len(g) > len(best["onepx"]["dets"]):
                        best["onepx"] = c
                    break
                end = max(end, x["t1"])
            if len({x["ch"] for x in g}) > 1:
                best.setdefault("xchan_all", []).append(rec)
            if prev and prev["span"][1] == span[0]:
                c = {"clip": clip, "dets": prev["dets"] + g, "cut": span[0], "all": raw,
                     "span": (prev["span"][0], span[1])}
                short = min(x["t1"] - x["t0"] for x in c["dets"])
                if best["touch"] is None or short > best["touch"]["short"]:
                    c["short"] = short
                    best["touch"] = c
            prev = rec
    return best


def spectro(ax, dataset, clip, ch, lo, hi, title=None):
    gray, t_lo, t_hi = SOURCE[dataset].crop(clip, ch, lo, hi)
    top, bot = band_rows(gray.shape[0], NYQ * 1000, BAND[0] * 1000, BAND[1] * 1000)
    ax.imshow(gray[top:bot], cmap="gray", aspect="auto", interpolation="nearest",
              extent=[t_lo, t_hi, BAND[0], BAND[1]], origin="upper")
    ax.set_ylabel(f"ch {ch}", fontsize=8, color=MUTED)
    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)


def pack(groups):
    """Greedy interval packing: sub-row per candidate, so overlaps stay visible."""
    spans = sorted(((min(x["t0"] for x in g), max(x["t1"] for x in g), g)
                    for g in groups), key=lambda s: (s[0], -(s[1] - s[0])))
    rows, out = [], []
    for t0, t1, g in spans:
        for r, end in enumerate(rows):
            if t0 >= end:
                rows[r] = t1
                out.append((r, t0, t1, g))
                break
        else:
            rows.append(t1)
            out.append((len(rows) - 1, t0, t1, g))
    return out


def window(case, pad=None):
    """The drawn window: case span plus pad, clamped off neighbouring detections."""
    pad = PAD if pad is None else pad
    t0 = min(d["t0"] for d in case["dets"])
    t1 = max(d["t1"] for d in case["dets"])
    lo, hi = t0 - pad, t1 + pad
    if not GHOST_ON:
        ids = {id(x) for x in case["dets"]}
        for x in case["all"]:
            if id(x) in ids:
                continue
            if x["t1"] <= t0:
                lo = max(lo, x["t1"])
            if x["t0"] >= t1:
                hi = min(hi, x["t0"])
    return lo, hi


def rule_rows(case, lo, hi):
    """Packed candidates per rule, restricted to the drawn window."""
    ids = {id(x) for x in case["dets"]}
    out = []
    for groups in (group_detections(case["all"], 0.0), components(case["all"], TAU)):
        vis = [g for g in groups
               if min(x["t0"] for x in g) < hi and max(x["t1"] for x in g) > lo]
        if not GHOST_ON:
            vis = [g for g in vis if any(id(x) in ids for x in g)]
        out.append([(r, t0, t1, any(id(x) in ids for x in g))
                    for r, t0, t1, g in pack(vis)])
    return out


def lanes(ax, case, detectors, lo, hi):
    """One row per (detector, channel); both rules' candidates stacked on top."""
    ids = {id(x) for x in case["dets"]}
    dets = [x for x in case["all"] if x["t0"] < hi and x["t1"] > lo
            and (GHOST_ON or id(x) in ids)]
    chans = sorted({x["ch"] for x in case["dets"]})
    rows = [(d, c) for d in detectors for c in chans]
    ys = {k: len(rows) - 1 - i for i, k in enumerate(rows)}
    for d in dets:
        if (d["detector"], d["ch"]) not in ys:
            continue
        ax.barh(ys[(d["detector"], d["ch"])], d["t1"] - d["t0"], left=d["t0"],
                height=0.66, color=COLOR[d["detector"]], edgecolor=SURFACE,
                linewidth=1.0, zorder=3, alpha=1.0 if id(d) in ids else GHOST)
    cur, new = rule_rows(case, lo, hi)
    y_cur = len(rows) + 0.45
    y_new = y_cur + max(r for r, *_ in cur) + 1.55
    top = y_new + max(r for r, *_ in new)
    for packed, base, col in ((cur, y_cur, INK), (new, y_new, PROP)):
        for r, t0, t1, focal in packed:
            ax.barh(base + r, t1 - t0, left=t0, height=0.66, color=col,
                    edgecolor=SURFACE, linewidth=1.0, zorder=3,
                    alpha=1.0 if focal else GHOST)
            if focal:
                for t in (t0, t1):
                    ax.plot([t, t], [-0.6, base + r], color=col, lw=0.7, ls=":",
                            alpha=0.4, zorder=1)
    multi = len(chans) > 1
    ax.set_yticks([y_new, y_cur] + [ys[k] for k in rows])
    ax.set_yticklabels([LANES[1], LANES[0]]
                       + [f"{d}  ch{c}" if multi else d for d, c in rows],
                       fontsize=8, color=INK)
    for lbl, col in zip(ax.get_yticklabels(), [PROP, INK]):
        lbl.set_color(col)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, top + 0.6)
    ax.set_xlabel("seconds", fontsize=8, color=MUTED)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel(fig, cell, case, dataset, detectors, title):
    lo, hi = window(case)
    chans = sorted({d["ch"] for d in case["dets"]})
    n_sub = sum(max(r for r, *_ in p) + 1 for p in rule_rows(case, lo, hi))
    n_lane = len(detectors) * len(chans) + n_sub
    inner = cell.subgridspec(len(chans) + 1, 1, hspace=0.14,
                             height_ratios=[1.6] * len(chans) + [0.42 * n_lane])
    for i, ch in enumerate(chans):
        spectro(fig.add_subplot(inner[i]), dataset, case["clip"], ch, lo, hi,
                title if i == 0 else None)
    lanes(fig.add_subplot(inner[-1]), case, detectors, lo, hi)


GD = ["sam3_best", "ridge", "squeakout", "das_yolo"]
DD = ["ridge", "sam3", "sam3_flatness_filtered"]
G = scan("gerbil_ssl", GD, 25)
D = scan("dryad_gerbil", DD, 5)

x = max((r for r in G["xchan_all"] if r["span"] != G["chain"]["span"]),
        key=lambda r: len(r["dets"]))
CASES = [
    (G["glue"], "gerbil_ssl", GD,
     f"one {G['glue']['glue_det']} detection spans {G['glue']['parts']} islands"),
    (D["glue"], "dryad_gerbil", DD,
     f"dryad: {D['glue']['glue_det']} bridges {D['glue']['parts']} islands"),
    (G["chain"], "gerbil_ssl", GD,
     f"chain: {len(G['chain']['dets'])} detections, none spans the whole"),
    (D["touch"], "dryad_gerbil", DD, "touching: 0 ms gap, two candidates"),
    (G["onepx"], "gerbil_ssl", GD, "two events joined by a 0.95 px overlap"),
    (x, "gerbil_ssl", GD, "two mics pooled, one candidate"),
]

fig = plt.figure(figsize=(17, 15.6))
outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.2,
                          left=0.165, right=0.985, top=0.95, bottom=0.085)
for i, (case, ds, dets, title) in enumerate(CASES):
    panel(fig, outer[i // 2, i % 2], case, ds, dets, title)

handles = [plt.Line2D([], [], color=COLOR[k], lw=6, label=k) for k in COLOR]
handles += [plt.Line2D([], [], color=INK, lw=6, label=f"candidate: {LANES[0]}"),
            plt.Line2D([], [], color=PROP, lw=6, label=f"candidate: {LANES[1]}")]
if GHOST_ON:
    handles.append(plt.Line2D([], [], color=MUTED, lw=6, alpha=GHOST,
                              label="outside the illustrated case"))
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.012), labelcolor=INK)
fig.patch.set_facecolor(SURFACE)
fig.savefig(OUT, dpi=140, facecolor=SURFACE)
print(f"wrote {OUT}\n")


# ── metrics ───────────────────────────────────────────────────────────────────
# One row per (case, rule), over the illustrated case's detections only. "chain" is
# span / median member duration: 1.0 means the candidate is the size of its detections,
# high means it is a bridge across separate ones.

def metrics(groups, detectors):
    rows = []
    for g in groups:
        t0, t1 = min(x["t0"] for x in g), max(x["t1"] for x in g)
        durs = sorted(x["t1"] - x["t0"] for x in g)
        m = len(durs) // 2
        med = durs[m] if len(durs) % 2 else 0.5 * (durs[m - 1] + durs[m])
        fired = {x["detector"] for x in g}
        rows.append({"t0": t0, "t1": t1, "dur": t1 - t0, "n": len(g),
                     "chain": (t1 - t0) / med if med else float("nan"),
                     "k": len(fired),
                     "z": "".join(str(int(d in fired)) for d in detectors),
                     "chans": sorted({x["ch"] for x in g})})
    return rows


SHORT = ["glue-4islands", "dryad-glue", "chain-45", "touching-0ms", "onepx-join",
         "xchan-pooled"]

print(f"{'case':16s} {'rule':22s} {'n':>3s} {'longest':>8s} {'maxchain':>9s} "
      f"{'k=1':>4s} {'kmax':>5s} {'xchan':>6s}")
print("-" * 82)
for name, (case, ds, detectors, _) in zip(SHORT, CASES):
    dets = case["dets"]
    for rule, groups in zip(LANES, (group_detections(dets, 0.0), components(dets, TAU))):
        m = metrics(groups, detectors)
        print(f"{name:16s} {rule:22s} {len(m):3d} {max(r['dur'] for r in m):8.4f} "
              f"{max(r['chain'] for r in m):9.1f} "
              f"{sum(1 for r in m if r['k'] == 1):4d} {max(r['k'] for r in m):5d} "
              f"{sum(1 for r in m if len(r['chans']) > 1):6d}")

print("\n\nper-candidate detail (z = detector fired flags, in the order listed)")
for name, (case, ds, detectors, _) in zip(SHORT, CASES):
    dets = case["dets"]
    print(f"\n{name}  [{case['clip']}]  z order: {'|'.join(detectors)}")
    for rule, groups in zip(LANES, (group_detections(dets, 0.0), components(dets, TAU))):
        m = metrics(groups, detectors)
        print(f"  {rule}: {len(m)} candidate(s)")
        for i, r in enumerate(m):
            print(f"    #{i:<2d} {r['t0']:10.4f}-{r['t1']:10.4f}  dur={r['dur']:7.4f}  "
                  f"n_det={r['n']:3d}  k={r['k']}  z={r['z']}  "
                  f"chain={r['chain']:5.1f}  ch={','.join(map(str, r['chans']))}")
