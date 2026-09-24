"""Per-session precision vs recall of the SAM3 dryad_gerbil_full run, and why they split.

    python scripts/plot_session_breakdown.py
    -> outputs/eval/sam3_dryad_full/per_session_breakdown.png

Input is outputs/eval/sam3_dryad_full/per_session_breakdown.csv (metrics.csv plus
duration / GT-density covariates).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8985", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "legend.frameon": False,
})

m = pd.read_csv("outputs/eval/sam3_dryad_full/per_session_breakdown.csv")
m = m[(m.gt_per_s > 0) & m.f1.notna()].reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

# --- per-session precision vs recall, colored by call density ---------------
sc = ax1.scatter(m.recall, m.precision, c=np.log10(m.gt_per_s), s=12, cmap="Blues",
                 alpha=0.75, lw=0, vmin=np.log10(0.02), vmax=np.log10(10))
for f in (0.2, 0.4, 0.6, 0.8):
    r = np.linspace(f / (2 - f) + 1e-3, 1, 200)
    ax1.plot(r, f * r / (2 * r - f), color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    ax1.annotate(f"F1 {f}", (0.97 * f / (2 * 0.97 - f), 0.975), fontsize=7.5, color=MUTED,
                 ha="center", va="top", zorder=2)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.set_xlabel("recall"); ax1.set_ylabel("precision")
cb = fig.colorbar(sc, ax=ax1, pad=0.02)
cb.set_label("GT calls/s (log)", color=INK2, fontsize=8.5)
cb.set_ticks(np.log10([0.03, 0.1, 0.3, 1, 3, 10]))
cb.set_ticklabels(["0.03", "0.1", "0.3", "1", "3", "10"])
cb.outline.set_edgecolor(MUTED)
ax1.set_title("Precision vs recall per session", color=INK, loc="left")

# --- why they split: the detector's output rate is nearly fixed --------------
ax2.scatter(m.gt_per_s, m.pred_per_s, s=9, color=BLUE, alpha=0.28, lw=0)
lim = np.array([m.gt_per_s.min() * 0.8, m.gt_per_s.max() * 1.3])
ax2.plot(lim, lim, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=3)
sl, ic = np.polyfit(np.log10(m.gt_per_s), np.log10(m.pred_per_s), 1)
xs = np.logspace(*np.log10(lim), 50)
ax2.plot(xs, 10 ** (ic + sl * np.log10(xs)), color=ORANGE, lw=2.2, zorder=4)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_title("Detection rate vs call density", color=INK, loc="left")
ax2.set_xlabel("ground-truth call density (calls/s, log)")
ax2.set_ylabel("predicted event rate (events/s, log)")
ax2.legend(handles=[Line2D([], [], color=ORANGE, lw=2.2, label=f"fit, slope {sl:.2f}"),
                    Line2D([], [], color=INK2, lw=1.4, ls=(0, (5, 3)), label="1:1")],
           loc="lower right", fontsize=8.5)

for a in (ax1, ax2):
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout(pad=1.4)
out = "outputs/eval/sam3_dryad_full/per_session_breakdown.png"
fig.savefig(out, dpi=150)
print("wrote", out)
