"""Candidate-merge rules vs the GUI doc scheme, one channel, on the cases that separate them.

Each merge rule takes the same detections and returns candidates as (t0, t1, dets):

    ov>0          single linkage on any positive overlap, the rule the frozen pool was
                  built with. Partitions, but one over-long detection swallows every
                  island it touches.
    IoU>=0.5      single linkage on IoU. Fragments long calls and does not partition.

The GUI doc row is not a merge rule: its unit is the raw detection. Detections are
shown in random order; a shown detection's label propagates to every unlabeled
detection (any detector) with Q >= Ts (Q read as IoU, Ts = 0.5). Bars are colored
by step: filled is shown, hollow inherits that step's label. Best case: a "multiple
calls" label (doc case 3) never propagates.
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_label.candidates import find_clips, load_detections, group_detections
from vox_label.render import PngChunkSource, band_rows

TAU = 0.5
TS = 0.5
N_ORDERS = 2000
PX = 1.0 / 487
PAD = float(sys.argv[sys.argv.index("--pad") + 1]) if "--pad" in sys.argv else 0.05
OUT = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
       else "candidate_merge_rules.png")
CH = {"gerbil_ssl": 35, "dryad_gerbil": 0}

INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#ffffff", "#e5e4e0"
BAND_BG = "#f4f3f0"
STEP_COLOR = ["#4c3fbf", "#c2185b", "#00838f", "#8d6e00", "#5d4037", "#6a1b9a",
              "#2e7d32", "#ad1457", "#37474f", "#bf360c", "#0277bd", "#827717"]
COLOR = {"sam3_best": "#2a78d6", "ridge": "#eb6834", "squeakout": "#1baf7a",
         "das_yolo": "#eda100", "sam3": "#7b3fbf", "sam3_flatness_filtered": "#008300"}
SOURCE = {"gerbil_ssl": PngChunkSource("outputs/spectrograms", "gerbil_ssl", prefix="headmic"),
          "dryad_gerbil": PngChunkSource("outputs/spectrograms", "dryad_gerbil", prefix="mic")}
BAND, NYQ = (5.0, 60.0), 62.5


def span(dets):
    return min(x["t0"] for x in dets), max(x["t1"] for x in dets)


def linkage(dets, gap=0.0):
    return [(*span(g), g) for g in group_detections(dets, gap)]


def iou(a, b):
    i = min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])
    if i <= 0:
        return 0.0
    return i / ((a["t1"] - a["t0"]) + (b["t1"] - b["t0"]) - i)


def iou_link(dets, tau=TAU):
    """Single linkage under IoU >= tau. IoU >= tau implies overlap, so components
    refine the ov>0 groups and the pair scan stays inside one of them."""
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
        out.extend((*span(g), g) for g in buckets.values())
    return sorted(out)


def gui_steps(dets, seed=0):
    """Niegil's procedure in one random order: [(step, det, src, q)], src None if shown."""
    rng = random.Random(seed)
    order = list(dets)
    rng.shuffle(order)
    log, done = [], set()
    for x in order:
        if id(x) in done:
            continue
        step = sum(1 for e in log if e[2] is None) + 1
        log.append((step, x, None, None))
        done.add(id(x))
        for y in dets:
            # Any detector, including x's own: das_yolo emits near-duplicates.
            if id(y) not in done and iou(x, y) >= TS:
                done.add(id(y))
                log.append((step, y, x, iou(x, y)))
    return log


def gui_doc(dets, seed=0):
    """Returns (t0, t1, [d], shown, step) per detection."""
    return sorted(((x["t0"], x["t1"], [x], src is None, step)
                   for step, x, src, _q in gui_steps(dets, seed)),
                  key=lambda c: (c[0], c[1]))


def mean_shown(dets):
    return sum(sum(c[3] for c in gui_doc(dets, s)) for s in range(N_ORDERS)) / N_ORDERS


RULES = [("ov>0", lambda d: linkage(d, 0.0)),
         (f"IoU>={TAU}", iou_link),
         ("from Niegil", gui_doc)]


def scan(dataset, detectors, n_clips):
    """One example of each situation that tells the rules apart."""
    ch = CH[dataset]
    best = {"glue": None, "chain": None, "touch": None, "onepx": None, "clean": None}
    for clip in find_clips("outputs", dataset, detectors)[:n_clips]:
        raw = []
        for d in detectors:
            raw.extend(load_detections("outputs", dataset, clip, d, channels={ch}))
        if not raw:
            continue
        prev = None
        for g in group_detections(raw, 0.0):
            s = span(g)
            rec = {"clip": clip, "dets": g, "span": s, "all": raw}
            for d in detectors:
                rest = [x for x in g if x["detector"] != d]
                if not rest or len(rest) == len(g):
                    continue
                n = len(group_detections(rest, 0.0))
                if n >= 2:
                    cand = dict(rec, glue_det=d, parts=n)
                    if best["glue"] is None or n > best["glue"]["parts"]:
                        best["glue"] = cand
            if len(g) >= 3 and not any(x["t0"] <= s[0] and x["t1"] >= s[1] for x in g):
                if best["chain"] is None or len(g) > len(best["chain"]["dets"]):
                    best["chain"] = rec
            order = sorted(g, key=lambda x: (x["t0"], x["t1"]))
            end = order[0]["t1"]
            for x in order[1:]:
                if 0 < end - x["t0"] <= PX:
                    if best["onepx"] is None or len(g) > len(best["onepx"]["dets"]):
                        best["onepx"] = dict(rec)
                    break
                end = max(end, x["t1"])
            if len({x["detector"] for x in g}) == len(detectors) and len(g) <= len(detectors):
                if best["clean"] is None or len(g) > len(best["clean"]["dets"]):
                    best["clean"] = rec
            if prev and abs(prev["span"][1] - s[0]) < 1e-9:
                c = {"clip": clip, "dets": prev["dets"] + g, "all": raw,
                     "span": (prev["span"][0], s[1]),
                     "short": min(x["t1"] - x["t0"] for x in prev["dets"] + g)}
                if best["touch"] is None or c["short"] > best["touch"]["short"]:
                    best["touch"] = c
            prev = rec
    return best


def spectro(ax, dataset, clip, lo, hi, title):
    gray, t_lo, t_hi = SOURCE[dataset].crop(clip, CH[dataset], lo, hi)
    top, bot = band_rows(gray.shape[0], NYQ * 1000, BAND[0] * 1000, BAND[1] * 1000)
    ax.imshow(gray[top:bot], cmap="gray", aspect="auto", interpolation="nearest",
              extent=[t_lo, t_hi, BAND[0], BAND[1]], origin="upper")
    ax.set_ylabel(f"ch {CH[dataset]}", fontsize=8, color=MUTED)
    ax.set_yticks([])
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)


def pack(cands):
    """Greedy packing into sub-rows, so candidates that overlap stay visible."""
    rows, out = [], []
    for c in sorted(cands, key=lambda s: (s[0], -(s[1] - s[0]))):
        t0, t1 = c[0], c[1]
        for r, end in enumerate(rows):
            if t0 >= end - 1e-9:
                rows[r] = t1
                out.append((r, *c))
                break
        else:
            rows.append(t1)
            out.append((len(rows) - 1, *c))
    return out


def window(case, pad=PAD):
    t0, t1 = span(case["dets"])
    lo, hi = t0 - pad, t1 + pad
    ids = {id(x) for x in case["dets"]}
    for x in case["all"]:
        if id(x) in ids:
            continue
        if x["t1"] <= t0:
            lo = max(lo, x["t1"])
        if x["t0"] >= t1:
            hi = min(hi, x["t0"])
    return lo, hi


def rule_rows(case, rules=None):
    out = []
    for name, fn, *_ in rules or RULES:
        c = fn(case["dets"])
        out.append(one_per_row(c) if c and len(c[0]) == 5 else pack(c))
    return out


def one_per_row(cands):
    """Niegil: one detection per row, step 1 on top."""
    order = sorted(cands, key=lambda c: (-c[4], c[3], -c[0]))
    return [(r, *c) for r, c in enumerate(order)]


def det_rows(case, detectors):
    """Per detector, its detections packed so overlapping ones get separate sub-rows."""
    return {d: pack([(x["t0"], x["t1"], x) for x in case["dets"] if x["detector"] == d])
            for d in detectors}


def lanes(ax, case, detectors, lo, hi, rules=None):
    """Rules are (name, fn[, color]); fn returns (t0, t1, dets[, shown, step])."""
    rules = rules or RULES
    ys, y = {}, 0.0
    for d, rows in reversed(list(det_rows(case, detectors).items())):
        top = max((r for r, *_ in rows), default=0)
        for r, t0, t1, _x in rows:
            ax.barh(y + top - r, t1 - t0, left=t0, height=0.66,
                    color=COLOR[d], edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ys[d] = y + 0.5 * top
        y += top + 1
    packed = rule_rows(case, rules)
    base, ticks, labels = y + 0.7, [], []
    for i, ((name, _fn, *col), rows) in enumerate(zip(rules, packed)):
        top = max((r for r, *_ in rows), default=0)
        # A rule needing sub-rows would otherwise leave its label floating
        # between them; the band is what groups its bars.
        if i % 2 == 0:
            ax.axhspan(base - 0.55, base + top + 0.55, color=BAND_BG, zorder=0)
        for r, t0, t1, _g, *st in rows:
            if st:
                shown, step = st
                c = STEP_COLOR[(step - 1) % len(STEP_COLOR)]
                ax.barh(base + r, t1 - t0, left=t0, height=0.62,
                        color=c if shown else SURFACE, edgecolor=c,
                        linewidth=1.2, zorder=3)
                ax.text(t0, base + r, f"{step} ", ha="right", va="center",
                        fontsize=7, color=c, zorder=4)
            else:
                ax.barh(base + r, t1 - t0, left=t0, height=0.62,
                        color=col[0] if col else INK,
                        edgecolor=SURFACE, linewidth=1.2, zorder=3)
            for t in (t0, t1):
                ax.plot([t, t], [-0.6, base + r], color=INK, lw=0.7, ls=":",
                        alpha=0.32, zorder=1)
        ticks.append(base + 0.5 * top)
        labels.append(name)
        base += top + 1.7
    ax.set_yticks(ticks + [ys[d] for d in detectors])
    ax.set_yticklabels(labels + list(detectors), fontsize=8, color=INK)
    for lbl in ax.get_yticklabels()[:len(rules)]:
        lbl.set_fontfamily("monospace")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, base + 0.2)
    ax.set_xlabel("seconds", fontsize=8, color=MUTED)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel(fig, cell, case, dataset, detectors, title, rules=None):
    lo, hi = window(case)
    n_sub = sum(max((r for r, *_ in p), default=0) + 1 for p in rule_rows(case, rules))
    n_det = sum(max((r for r, *_ in p), default=0) + 1
                for p in det_rows(case, detectors).values())
    inner = cell.subgridspec(2, 1, hspace=0.14,
                             height_ratios=[1.6, 0.40 * (n_det + n_sub + 4)])
    spectro(fig.add_subplot(inner[0]), dataset, case["clip"], lo, hi, title)
    lanes(fig.add_subplot(inner[1]), case, detectors, lo, hi, rules)


def main():
    GD = ["sam3_best", "ridge", "squeakout", "das_yolo"]
    DD = ["ridge", "sam3", "sam3_flatness_filtered"]
    G = scan("gerbil_ssl", GD, 25)
    D = scan("dryad_gerbil", DD, 5)

    CASES = [
        (G["glue"], "gerbil_ssl", GD,
         f"one {G['glue']['glue_det']} detection spans {G['glue']['parts']} islands"),
        (D["glue"], "dryad_gerbil", DD,
         f"dryad: {D['glue']['glue_det']} bridges {D['glue']['parts']} islands"),
        (G["chain"], "gerbil_ssl", GD,
         f"chain: {len(G['chain']['dets'])} detections, none spans the whole"),
        (G["onepx"], "gerbil_ssl", GD, "two events joined by a sub-pixel overlap"),
        (D["touch"], "dryad_gerbil", DD, "touching: 0 ms gap"),
        (G["clean"], "gerbil_ssl", GD, "control: one call"),
    ]
    CASES = [c for c in CASES if c[0] is not None]

    fig = plt.figure(figsize=(18.5, 18.2))
    outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.40,
                              left=0.185, right=0.985, top=0.955, bottom=0.075)
    for i, (case, ds, dets, title) in enumerate(CASES):
        panel(fig, outer[i // 2, i % 2], case, ds, dets, title)

    handles = [plt.Line2D([], [], color=COLOR[k], lw=6, label=k) for k in COLOR]
    handles.append(plt.Line2D([], [], color=INK, lw=6, label="candidate"))
    handles.append(matplotlib.patches.Patch(facecolor=MUTED, label="Niegil: shown at step n"))
    handles.append(matplotlib.patches.Patch(facecolor=SURFACE, edgecolor=MUTED,
                                            label="Niegil: label propagated"))
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.012), labelcolor=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    print(f"wrote {OUT}\n")



    SHORT = ["glue-islands", "dryad-glue", "chain", "onepx-join", "touching", "control"]
    print(f"{'case':14s} {'rule':18s} {'n':>3s} {'longest':>8s} {'overlapping':>12s}"
          f" {'E[shown]':>9s}")
    print("-" * 70)
    for name, (case, ds, detectors, _) in zip(SHORT, CASES):
        for rname, fn in RULES:
            c = sorted(fn(case["dets"]), key=lambda x: (x[0], x[1]))
            bad = sum(1 for a, b in zip(c, c[1:]) if b[0] < a[1])
            e = f"{mean_shown(case['dets']):9.2f}" if rname == "from Niegil" else ""
            print(f"{name:14s} {rname:18s} {len(c):3d} "
                  f"{max(x[1] - x[0] for x in c):8.4f} {bad:12d} {e}")


if __name__ == "__main__":
    main()
