"""Real candidates, drawn: spectrogram on top, one lane per detector below."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from vox_label.candidates import find_clips, load_detections, group_detections
from vox_label.render import PngChunkSource, band_rows

SP = "/tmp/claude-2812/-mnt-home-the10-ssl/1219c16c-24d5-48f5-b612-5af86bfbcd9d/scratchpad"
PX = 1.0 / 487
INK, MUTED, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"
COLOR = {"sam3_best": "#2a78d6", "ridge": "#eb6834", "squeakout": "#1baf7a",
         "das_yolo": "#eda100", "sam3": "#e87ba4", "sam3_flatness_filtered": "#008300"}
SOURCE = {"gerbil_ssl": PngChunkSource("outputs/spectrograms", "gerbil_ssl", prefix="headmic"),
          "dryad_gerbil": PngChunkSource("outputs/spectrograms", "dryad_gerbil", prefix="mic")}
BAND, NYQ = (5.0, 60.0), 62.5


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
            rec = {"clip": clip, "dets": g, "span": span}
            # a detector holding otherwise separate islands together
            for d in detectors:
                rest = [x for x in g if x["detector"] != d]
                if not rest or len(rest) == len(g):
                    continue
                n = len(group_detections(rest, 0.0))
                if n >= 2:
                    cand = dict(rec, glue_det=d, parts=n)
                    if best["glue"] is None or n > best["glue"]["parts"]:
                        best["glue"] = cand
            # transitive chain: nothing spans it
            if len(g) >= 3 and not any(x["t0"] <= span[0] and x["t1"] >= span[1] for x in g):
                if best["chain"] is None or len(g) > len(best["chain"]["dets"]):
                    best["chain"] = rec
            # a merge hinging on a pixel or less
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
            # two mics in one candidate (keep several: the biggest may also be the
            # chain example, and one figure should not show the same candidate twice)
            if len({x["ch"] for x in g}) > 1:
                best.setdefault("xchan_all", []).append(rec)
            # exactly touching neighbours, kept apart
            if prev and prev["span"][1] == span[0]:
                c = {"clip": clip, "dets": prev["dets"] + g, "cut": span[0],
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
    ax.set_yticks([]); ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)


def lanes(ax, dets, detectors, lo, hi):
    """One row per (detector, channel); merged candidates on top."""
    chans = sorted({d["ch"] for d in dets})
    rows = [(d, c) for d in detectors for c in chans]
    ys = {k: len(rows) - 1 - i for i, k in enumerate(rows)}
    for d in dets:
        ax.barh(ys[(d["detector"], d["ch"])], d["t1"] - d["t0"], left=d["t0"],
                height=0.66, color=COLOR[d["detector"]], edgecolor=SURFACE,
                linewidth=1.0, zorder=3)
    y = len(rows) + 0.45
    for g in group_detections(dets, 0.0):
        t0, t1 = min(x["t0"] for x in g), max(x["t1"] for x in g)
        ax.barh(y, t1 - t0, left=t0, height=0.66, color=INK, edgecolor=SURFACE,
                linewidth=1.0, zorder=3)
        for t in (t0, t1):
            ax.plot([t, t], [-0.6, y], color=MUTED, lw=0.7, ls=":", zorder=1)
    multi = len(chans) > 1
    ax.set_yticks([y] + [ys[k] for k in rows])
    ax.set_yticklabels(["candidate"] + [f"{d}  ch{c}" if multi else d for d, c in rows],
                       fontsize=8, color=INK)
    ax.set_xlim(lo, hi); ax.set_ylim(-0.6, y + 0.6)
    ax.set_xlabel("seconds", fontsize=8, color=MUTED)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(axis="x", color=RULE, lw=0.6); ax.set_axisbelow(True)
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel(fig, cell, case, dataset, detectors, title, pad=0.05):
    dets = case["dets"]
    lo = min(d["t0"] for d in dets) - pad
    hi = max(d["t1"] for d in dets) + pad
    chans = sorted({d["ch"] for d in dets})
    n_lane = len(detectors) * len(chans) + 1
    inner = cell.subgridspec(len(chans) + 1, 1, hspace=0.14,
                             height_ratios=[1.6] * len(chans) + [0.42 * n_lane])
    for i, ch in enumerate(chans):
        spectro(fig.add_subplot(inner[i]), dataset, case["clip"], ch, lo, hi,
                title if i == 0 else None)
    lanes(fig.add_subplot(inner[-1]), dets, detectors, lo, hi)


GD = ["sam3_best", "ridge", "squeakout", "das_yolo"]
DD = ["ridge", "sam3", "sam3_flatness_filtered"]
G = scan("gerbil_ssl", GD, 25)
D = scan("dryad_gerbil", DD, 5)

fig = plt.figure(figsize=(17, 15))
outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.2,
                          left=0.135, right=0.985, top=0.95, bottom=0.085)

g = G["glue"]
panel(fig, outer[0, 0], g, "gerbil_ssl", GD,
      f"one {g['glue_det']} detection spans {g['parts']} islands")
d = D["glue"]
panel(fig, outer[0, 1], d, "dryad_gerbil", DD,
      f"dryad: {d['glue_det']} bridges {d['parts']} islands")
c = G["chain"]
panel(fig, outer[1, 0], c, "gerbil_ssl", GD,
      f"chain: {len(c['dets'])} detections, none spans the whole")
t = D["touch"]
panel(fig, outer[1, 1], t, "dryad_gerbil", DD,
      "touching: 0 ms gap, two candidates")
p = G["onepx"]
panel(fig, outer[2, 0], p, "gerbil_ssl", GD,
      "two events joined by a 0.95 px overlap")
# biggest cross-channel candidate that is not already on the figure
x = max((r for r in G["xchan_all"] if r["span"] != c["span"]),
        key=lambda r: len(r["dets"]))
panel(fig, outer[2, 1], x, "gerbil_ssl", GD,
      "two mics pooled, one candidate")

handles = [plt.Line2D([], [], color=COLOR[k], lw=6, label=k) for k in COLOR]
handles.append(plt.Line2D([], [], color=INK, lw=6, label="merged candidate"))
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, 0.012), labelcolor=INK)
fig.patch.set_facecolor(SURFACE)
fig.savefig(f"{SP}/merge_cases.png", dpi=140, facecolor=SURFACE)
print("wrote merge_cases.png")
for k, v in [("gerbil glue", G["glue"]), ("dryad glue", D["glue"]),
             ("chain", G["chain"]), ("touch", D["touch"]),
             ("onepx", G["onepx"]), ("xchan", x)]:
    print(f"  {k:12s} {v['clip']:24s} t={v['span'][0]:9.3f} n={len(v['dets']):3d}")
