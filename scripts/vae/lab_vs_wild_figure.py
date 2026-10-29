"""Paper-style lab vs wild figure (reproduce_umap_figure.py layout) + interactive latent map.

UMAP is fit on lab latents only (paper latent_means, equal per cohort); wild calls (verified + filtered that passed
the --min-peak cut in lab_vs_wild_vae.py compare) are projected into it with transform().
  lab panel   the paper's 12 Figure 2 states; example = highest-z70-confidence local exemplar (model input)
  maps        main figure on the UMAP fit on lab + wild together (keeps lab-wild centroid distances far better than
              projecting wild into a lab-only UMAP); the projected version is saved as *_projected.png
  wild panel  each state matched one-to-one (Hungarian) to a wild cluster by Euclidean distance between cluster
              centroids (mean latent) in the 32-d latent space; example = wild call nearest the cluster's centroid
              (verified preferred among the 10 nearest). Each label carries that distance and the state's centroid
              distance to its nearest other lab state ("lab gap"); a wild match beyond the lab gap is no closer than
              a different lab call type. Check: nn = share of the wild cluster's calls' 15 nearest lab calls that
              belong to the state; under 10% is flagged (!), the centroids are close but the calls are not.
              Distances in state_matches.csv.
Interactive page: separate lab and wild plots, each with its own pan/zoom, switchable between this lab-fit map and a
UMAP fit on lab and wild together; hovering selects the cluster whose 2D medoid is nearest the cursor (switching only
when another medoid is clearly closer) and shows 5 members (model inputs).

    python scripts/vae/lab_vs_wild_figure.py --out outputs/ava/lab_vs_wild_snr
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/vae"))
from lab_vs_wild_vae import load_lab_df  # noqa: E402

STATES = [8, 66, 38, 17, 19, 30, 51, 59, 31, 36, 57, 5]  # paper Figure 2 border states, notebook order
BORDER = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0)]
COLORS = ["#e6194b", "#f58231", "#ffe119", "#bfef45", "#3cb44b", "#42d4f4",
          "#4363d8", "#911eb4", "#f032e6", "#a9a9a9", "#9a6324", "#000075"]
TEMPLATE = ROOT / "scripts/vae/lab_vs_wild_explorer.html"


def png_b64(spec, size=96):
    import matplotlib.pyplot as plt
    from PIL import Image
    rgb = (plt.cm.magma(np.clip(spec, 0, 1))[..., :3] * 255).astype(np.uint8)[::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((size, size), Image.BILINEAR).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def panel(fig, gs, emb, cols, picks, title, lim):
    """picks: list of (label, spec, xy, color) in BORDER order."""
    for (r, c), (label, spec, _, color) in zip(BORDER, picks):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(spec, origin="lower", cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title(label, fontsize=8, color=color)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(5)
    ax = fig.add_subplot(gs[1:3, 1:3])
    ax.scatter(emb[:, 0], emb[:, 1], s=0.5, c=cols, alpha=0.35, lw=0, rasterized=True)
    # picks sharing one example call (one wild cluster matched to several states) become concentric rings
    groups = {}
    for _, _, xy, color in picks:
        groups.setdefault(tuple(np.round(xy, 4)), []).append(color)
    for xy, colors in groups.items():
        for k, color in enumerate(colors):
            ax.plot(*xy, "o", ms=10 + 7 * (len(colors) - 1 - k), mec="k", mew=1.5, c=color, zorder=3 + k)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lim[:, 0]); ax.set_ylim(lim[:, 1]); ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/ava/lab_vs_wild_snr")
    ap.add_argument("--n-lab", type=int, default=20_000)
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    rng = np.random.default_rng(a.seed)
    lab = load_lab_df(("cohort", "z_70", "latent_means"))
    per = a.n_lab // lab.cohort.nunique()
    idx = np.concatenate([rng.choice(np.flatnonzero(lab.cohort == c), per, replace=False)
                          for c in sorted(lab.cohort.unique())])
    Zl, zl = np.stack(lab.latent_means.values[idx]).astype(np.float32), lab.z_70.values[idx]
    del lab
    w = pd.read_csv(a.out / "wild_assigned.csv", index_col="index")
    w = w[w.wild_cluster >= 0]
    Zw = np.load(a.out / "wild_latent.npy")[w.index.values]
    wc = w.wild_cluster.values
    lex = pd.read_csv(a.out / "lab_exemplars.csv", index_col="index")
    with h5py.File(a.out / "lab_exemplars.h5") as f:
        lspecs, lZ = f["specs"][:], f["latent_official"][:]

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=a.seed).fit(Zl)
    El, Ew, Elex = reducer.embedding_, reducer.transform(Zw), reducer.transform(lZ)
    np.savez(a.out / "lab_fit_umap.npz", lab=El, lab_z70=zl, wild=Ew, wild_index=w.index.values)
    jred = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=a.seed).fit(np.vstack([Zl, Zw]))
    Jl, Jw = jred.embedding_[:len(Zl)], jred.embedding_[len(Zl):]  # lab and wild fit together
    Jlex = jred.transform(lZ)
    np.savez(a.out / "joint_umap.npz", lab=Jl, lab_z70=zl, wild=Jw, wild_index=w.index.values)

    # how well each map keeps clusters together: share of a call's 15 nearest neighbours in the same cluster
    def agree(X, y, k=15):
        idx_ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[1][:, 1:]
        return float((y[idx_] == y[:, None]).mean())
    pd.DataFrame([{"data": "lab (z70)", "latent": agree(Zl, zl), "lab_fit_projected": agree(El, zl), "joint": agree(Jl, zl)},
                  {"data": "wild (wild clusters)", "latent": agree(Zw, wc), "lab_fit_projected": agree(Ew, wc),
                   "joint": agree(Jw, wc)}]).to_csv(a.out / "map_coherence.csv", index=False)

    def medoid(E):
        return E[np.argmin(((E - np.median(E, axis=0)) ** 2).sum(1))].round(3).tolist()

    lab_ids = sorted(set(zl))
    wild_ids = sorted(set(wc))
    cw = {c: Zw[wc == c].mean(0) for c in wild_ids}

    def wild_members(c, n):
        m = np.flatnonzero(wc == c)
        d = np.linalg.norm(Zw[m] - cw[c], axis=1)
        o = m[np.argsort(d)]
        near = o[:10]
        ver = near[w.set.values[near] == "verified"]
        first = ver[0] if len(ver) else o[0]
        return [first] + [i for i in o if i != first][:n - 1]

    def lab_members(z, n):
        e = lex[lex.z_70 == z].sort_values("conf", ascending=False)
        return list(e.index[:n])

    from scipy.optimize import linear_sum_assignment
    cl = {z: Zl[zl == z].mean(0) for z in lab_ids if (zl == z).sum() >= 20}
    D = np.array([[np.linalg.norm(cl[z] - cw[c]) for c in wild_ids] for z in STATES])
    r_, c_ = linear_sum_assignment(D)  # one-to-one: no wild cluster serves two states
    lab_nn = NearestNeighbors(n_neighbors=15).fit(Zl)
    rows = []
    for i, j in zip(r_, c_):
        z = STATES[i]
        gap = min(np.linalg.norm(cl[z] - cl[o]) for o in cl if o != z)
        rows.append({"z70": z, "wild_cluster": int(wild_ids[j]), "centroid_dist": float(D[i, j]), "lab_gap": float(gap),
                     "within_gap": bool(D[i, j] <= gap),
                     # share of the wild cluster's calls' 15 nearest lab calls (latent) that are this state
                     "knn_share": float((zl[lab_nn.kneighbors(Zw[wc == wild_ids[j]])[1]] == z).mean()),
                     "nearest_unconstrained": int(wild_ids[D[i].argmin()]),
                     "nearest_unconstrained_dist": float(D[i].min())})
    sm = pd.DataFrame(rows)
    sm.to_csv(a.out / "state_matches.csv", index=False)
    print(sm.round(2).to_string(index=False))

    with h5py.File(a.out / "wild_specs.h5") as f:
        ws = f["specs"]
        wspec = lambda i: ws[int(w.index.values[i])]
        # paper layout on the joint map (main figure) and on the lab-fit map with wild projected
        for fname, Lm, Wm, Lex, tl, tw in (
                ("paper_layout_lab_vs_wild.png", Jl, Jw, Jlex, "lab, UMAP fit on lab + wild", "wild, UMAP fit on lab + wild"),
                ("paper_layout_lab_vs_wild_projected.png", El, Ew, Elex, "lab, UMAP fit on lab", "wild, projected")):
            lab_picks, wild_picks = [], []
            for z, color in zip(STATES, COLORS):
                li = lab_members(z, 1)
                if li:
                    lab_picks.append((f"z70={z}", lspecs[li[0]], Lex[li[0]], color))
                else:
                    lab_picks.append((f"z70={z}\nno local example", np.zeros((128, 128)), Lm[zl == z].mean(0), color))
            for r, color in zip(sm.itertuples(), COLORS):
                i = wild_members(r.wild_cluster, 1)[0]
                flag = "" if r.knn_share >= 0.1 else " (!)"
                wild_picks.append((f"W{r.wild_cluster} (for z{r.z70}){flag}\nd {r.centroid_dist:.2f}, gap {r.lab_gap:.2f}, "
                                   f"nn {r.knn_share:.0%}", wspec(i), Wm[i], color))
            fig = plt.figure(figsize=(20, 10.5))
            outer = fig.add_gridspec(1, 2, wspace=0.08)
            lim = np.percentile(np.vstack([Lm, Wm]), [0.2, 99.8], axis=0)  # shared frame for both panels
            lim += np.array([[-1], [1]]) * 0.06 * (lim[1] - lim[0])
            panel(fig, outer[0].subgridspec(4, 4), Lm, "#555555", lab_picks, f"{tl} (n={len(Lm):,})", lim)
            panel(fig, outer[1].subgridspec(4, 4), Wm, "#555555", wild_picks, f"{tw} (n={len(Wm):,})", lim)
            fig.savefig(a.out / fname, dpi=130, facecolor="white", bbox_inches="tight")
            plt.close(fig)
        lab_pal = plt.cm.tab20(np.arange(20))

        # joint map, lab | wild, coloured by cluster, shared limits
        fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), sharex=True, sharey=True)
        jlim = np.percentile(np.vstack([Jl, Jw]), [0.2, 99.8], axis=0)
        jlim += np.array([[-1], [1]]) * 0.05 * (jlim[1] - jlim[0])
        for ax, E2, lab_, t in ((axs[0], Jl, zl, f"lab (n={len(Jl):,}), paper z70"),
                                (axs[1], Jw, wc, f"wild (n={len(Jw):,}), wild clusters")):
            ax.scatter(E2[:, 0], E2[:, 1], s=1.2, c=lab_pal[lab_ % 20], alpha=0.5, lw=0, rasterized=True)
            ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(jlim[:, 0]); ax.set_ylim(jlim[:, 1]); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(a.out / "joint_umap_lab_vs_wild.png", dpi=130, facecolor="white"); plt.close(fig)

        # interactive explorer data
        sub_l = rng.choice(len(El), min(len(El), 12_000), replace=False)
        sub_w = rng.choice(len(Ew), min(len(Ew), 12_000), replace=False)
        lab_c, wild_c = [], []
        for z in lab_ids:
            mem = lab_members(z, a.n_members)
            lab_c.append({"id": int(z), "n": int((zl == z).sum()), "share": float((zl == z).mean()),
                          "xy": medoid(El[zl == z]), "xyJ": medoid(Jl[zl == z]),
                          "imgs": [png_b64(lspecs[i]) for i in mem]})
        for c in wild_ids:
            m = wc == c
            mem = wild_members(c, a.n_members)
            wild_c.append({"id": int(c), "n": int(m.sum()), "share": float(m.mean()),
                           "verified": int((w.set.values[m] == "verified").sum()),
                           "outside99": float(w.outside99.values[m].mean()),
                           "xy": medoid(Ew[m]), "xyJ": medoid(Jw[m]),
                           "imgs": [png_b64(wspec(i)) for i in mem]})
    data = {"lab": {"x": El[sub_l, 0].round(3).tolist(), "y": El[sub_l, 1].round(3).tolist(),
                    "c": zl[sub_l].astype(int).tolist(), "n": int(len(El))},
            "wild": {"x": Ew[sub_w, 0].round(3).tolist(), "y": Ew[sub_w, 1].round(3).tolist(),
                     "c": wc[sub_w].astype(int).tolist(), "n": int(len(Ew))},
            "labClusters": lab_c, "wildClusters": wild_c}
    for k, E2, sub in (("lab", Jl, sub_l), ("wild", Jw, sub_w)):
        data[k]["xJ"], data[k]["yJ"] = E2[sub, 0].round(3).tolist(), E2[sub, 1].round(3).tolist()
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    (a.out / "explorer.html").write_text(html)
    print("wrote", a.out / "paper_layout_lab_vs_wild.png", "and", a.out / "explorer.html",
          f"({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
