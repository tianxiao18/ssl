"""Lab (dryad, Peterson et al.) vs wild (Mongolia AudioMoth) gerbil calls in the paper's pretrained VAE latent space.

Lab side = the paper's own latent_means and GMM k=70 labels (z_70) from vocalization_df; no re-encoding needed.
Wild side = Mongolia segments run through the same paper preprocessing (audio /32768, resampled 192 -> 125 kHz,
ava get_spec with dryad_to_ava_specs.PREPROCESS_PARAMS) and encoded with checkpoint_050.tar (latent = mu).
--floor perfreq (default) subtracts a per-frequency noise floor (median log|STFT| of the --ctx s of audio on either
side of the segment) before the dB window; the window start is calibrated on random lab calls so that, with the same
floor subtraction, lab spectrograms light the same share of pixels as the paper's fixed [-8, -5] window. Lab specs
and latents stay the paper's own. --floor none = the paper window on wild audio as is.
Wild sets:
  verified      human label 1 in mongolia_audiomoth_k4_hp15k (candidate span)
  noncall       human label 0 in the same campaign (control)
  filtered      sam3 gold ft hp15k detections kept by the GBM filter (keep=1) on mongolia_audiomoth_all_k4_hp15k,
                span = union of the gold ft boxes, filter peak >= --min-peak-khz
  filtered_low  same, filter peak below --min-peak-khz (the known 15-25 kHz false-positive band)
Stages (one sbatch runs all three; compare is CPU only and can be rerun alone):
  specs    wild specs + per-cluster lab exemplar specs (cohorts 2/4, top z70 confidence)
  encode   checkpoint_050 mu for wild + lab exemplars (exemplars double as a reproduction check)
  compare  UMAP fit on lab, wild projected; kNN z_70 usage; distance to lab; wild GMM clusters vs nearest lab cluster

    python scripts/vae/lab_vs_wild_vae.py {specs,encode,compare} --out outputs/ava/lab_vs_wild
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from scipy.signal import resample_poly, stft

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/vae"))
from dryad_to_ava_specs import PREPROCESS_PARAMS as P, get_spec, parse_vec  # noqa: E402

DRYAD = Path("/mnt/home/the10/ceph/dataset/dryad_gerbil")
LAB_AUDIO = ROOT / "data/dryad_gerbil_full"
WILD_AUDIO = ROOT / "data/mongolia_audiomoth"
CAMPAIGN = ROOT / "outputs/label_campaigns/mongolia_audiomoth_k4_hp15k"
POOL = ROOT / "outputs/label_campaigns/mongolia_audiomoth_all_k4_hp15k"
GOLD = ROOT / "outputs/sam3_ft_traces_gold_v2_hp15k/mongolia_audiomoth"
GOLD_DET = "sam3_ft_traces_gold_v2_hp15k"
SCORES = ROOT / ("outputs/eval/candidate_filter/mongolia_audiomoth_k4_hp15k/"
                 "apply_mongolia_audiomoth_all_k4_hp15k/scores.csv")
FS = 125_000
TF = np.linspace(P["min_freq"], P["max_freq"], P["num_freq_bins"])
warnings.filterwarnings("ignore", category=WavFileWarning)
warnings.filterwarnings("ignore", message="Found segment longer")


def gold_spans(f):
    """cand_id -> (start, end) union of its gold ft boxes."""
    from vox_tracer.scoring import bbox_to_time
    p = pd.read_csv(POOL / "candidates.csv", usecols=["cand_id", "clip", "source_anns"])
    p = p[p.cand_id.isin(f.cand_id)]
    spans = {}
    for clip, g in p.groupby("clip"):
        d = json.load(open(GOLD / clip / "coco_ch_0.json"))
        im = {i["id"]: i for i in d["images"]}
        ann = {a["id"]: a for a in d["annotations"]}
        for cid, src in zip(g.cand_id, g.source_anns):
            ts = []
            for s in json.loads(src):
                if s["detector"] == GOLD_DET:
                    a = ann[s["ann_id"]]; i = im[a["image_id"]]
                    ts.append(bbox_to_time(a["bbox"], i["window_start_sec"], i["window_end_sec"], i["width"]))
            spans[cid] = (min(t[0] for t in ts), max(t[1] for t in ts))
    return spans


def wild_segments(min_peak_khz):
    lab = {}
    for line in open(CAMPAIGN / "labels.jsonl"):
        r = json.loads(line)
        if r.get("label") is not None:
            lab[r["cand_id"]] = r["label"]  # last label wins
    c = pd.read_csv(CAMPAIGN / "candidates.csv", usecols=["cand_id", "clip", "t_start", "t_end"])
    c["label"] = c.cand_id.map(lab)
    c = c[c.label.notna()].copy()
    c["set"] = np.where(c.label == 1, "verified", "noncall")
    s = pd.read_csv(SCORES)
    f = s[s.keep == 1][["cand_id", "clip", "score", "peak_khz_mean"]].copy()
    sp = gold_spans(f)
    f["t_start"] = f.cand_id.map(lambda c: sp[c][0])
    f["t_end"] = f.cand_id.map(lambda c: sp[c][1])
    f["set"] = np.where(f.peak_khz_mean >= min_peak_khz, "filtered", "filtered_low")
    df = pd.concat([c.drop(columns="label"), f], ignore_index=True)
    df["colony"] = df["clip"].str.split("/").str[0]
    return df[(df.t_end - df.t_start) <= P["max_dur"]].reset_index(drop=True)


P_RAW = dict(P, spec_min_val=-20.0, spec_max_val=0.0)  # get_spec output * 20 - 20 = log|STFT| in [-20, 0]


def floored_log(audio, fs, t1, t2, ctx, wild):
    """log|STFT| on the get_spec grid minus the per-frequency median of the surrounding context; shoulders = -inf."""
    s0, s1 = max(0, int((t1 - ctx) * fs)), min(len(audio), int((t2 + ctx) * fs))
    x = audio[s0:s1].astype(np.float64)
    a, b = t1 - s0 / fs, t2 - s0 / fs
    if wild:
        x, fs = resample_poly(x / 32768.0, FS, fs), FS
    sp, _ = get_spec(a, b, x, P_RAW, fs, target_freqs=TF)
    L = sp * 20.0 - 20.0
    L[sp <= 0] = -np.inf
    c = np.concatenate([x[:int(a * fs)], x[int(b * fs):]])
    if len(c) < P["nperseg"]:
        return L
    f, _, S = stft(c - c.mean(), fs=fs, nperseg=P["nperseg"], noverlap=P["noverlap"])
    prof = np.interp(TF, f, np.median(np.log(np.abs(S) + 1e-12), axis=1))
    return L - prof[:, None]


def calibrate_lo(lab, n, ctx, seed):
    """Window start for floor-subtracted specs giving lab calls the paper window's median lit-pixel share."""
    r = lab.sample(min(n, len(lab)), random_state=seed)
    paper, rel = [], []
    for path, g in r.groupby("path"):
        fs, audio = wavfile.read(path, mmap=True)
        for t1, t2 in zip(g.onset, g.offset):
            sp, _ = get_spec(t1, t2, audio, P_RAW, fs, target_freqs=TF)
            paper.append(np.mean(sp * 20 - 20 > P["spec_min_val"]))
            rel.append(floored_log(audio, fs, t1, t2, ctx, wild=False).ravel())
    target = np.median(paper)
    grid = np.linspace(-1, 4, 501)
    lo = grid[np.argmin([abs(np.median([np.mean(x > g) for x in rel]) - target) for g in grid])]
    return float(lo), float(target)


def wild_spec(audio, fs, t1, t2, lo, ctx):
    if lo is None:  # paper window on the raw wild audio
        s0, s1 = max(0, int((t1 - 0.02) * fs)), int((t2 + 0.02) * fs)
        x = resample_poly(audio[s0:s1].astype(np.float64) / 32768.0, FS, fs)
        spec, _ = get_spec(t1 - s0 / fs, t2 - s0 / fs, x, P, FS, target_freqs=TF)
        return np.clip(spec, 0, 1).astype(np.float32)
    L = floored_log(audio, fs, t1, t2, ctx, wild=True)
    span = P["spec_max_val"] - P["spec_min_val"]
    return np.clip((L - lo) / span, 0, 1).astype(np.float32)


def load_lab_df(cols=("audio_filename", "onset", "offset", "cohort", "z_70", "prob_z_70", "latent_means")):
    parts = []
    for ch in pd.read_csv(DRYAD / "vocalization_df.csv", usecols=list(cols), chunksize=100_000):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["latent_means"] = df.latent_means.map(parse_vec)
    if "prob_z_70" in df:
        df["conf"] = [parse_vec(p)[z] for p, z in zip(df.prob_z_70, df.z_70)]
        df = df.drop(columns="prob_z_70")
    return df


def stage_specs(out, per_cluster, min_peak_khz, floor, ctx, seed):
    lab = load_lab_df()
    lab = lab[lab.cohort.isin(["c2", "c4"])]
    lab["path"] = [LAB_AUDIO / f"cohort{c[1]}_combined_audio" / a for c, a in zip(lab.cohort, lab.audio_filename)]
    lab = lab[[p.exists() for p in lab.path]]
    lo = None
    if floor == "perfreq":
        lo, share = calibrate_lo(lab, 1000, ctx, seed)
        (out / "floor_calibration.json").write_text(json.dumps(
            {"lo": lo, "lab_lit_share_paper_window": share, "ctx_s": ctx, "n_lab": 1000}, indent=2))
        print(f"per-frequency floor: window start {lo:.3f} above floor (lab lit share {share:.4f})")

    w = wild_segments(min_peak_khz)
    print(w.groupby("set").size().to_dict())
    specs = np.zeros((len(w), *P_SHAPE), np.float32)
    for clip, g in w.groupby("clip"):
        wav = next((WILD_AUDIO / clip).glob("mic_0_*.wav"))
        fs, audio = wavfile.read(wav, mmap=True)
        for i, r in zip(g.index, g.itertuples()):
            specs[i] = wild_spec(audio, fs, r.t_start, r.t_end, lo, ctx)
    w.to_csv(out / "wild.csv", index_label="index")
    with h5py.File(out / "wild_specs.h5", "w") as f:
        f.create_dataset("specs", data=specs)

    ex = (lab.sort_values("conf", ascending=False).groupby("z_70").head(per_cluster)
          .sort_values(["path", "onset"]).reset_index(drop=True))
    lspecs = np.zeros((len(ex), *P_SHAPE), np.float32)
    for path, g in ex.groupby("path"):
        fs, audio = wavfile.read(path, mmap=True)
        for i, r in zip(g.index, g.itertuples()):
            spec, _ = get_spec(r.onset, r.offset, audio, P, fs, target_freqs=TF)
            lspecs[i] = np.clip(spec, 0, 1)
    with h5py.File(out / "lab_exemplars.h5", "w") as f:
        f.create_dataset("specs", data=lspecs)
        f.create_dataset("latent_official", data=np.stack(ex.latent_means.values).astype(np.float32))
    ex.drop(columns=["latent_means", "path"]).to_csv(out / "lab_exemplars.csv", index_label="index")
    print(f"wild {len(w)} specs, lab exemplars {len(ex)} ({ex.z_70.nunique()} clusters)")


P_SHAPE = (P["num_freq_bins"], P["num_time_bins"])


def stage_encode(out, checkpoint, batch):
    import torch
    from train_ava_ab import StableVAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableVAE(save_dir=str(out), z_dim=32)
    model.load_state(str(checkpoint))
    model.to(dev).eval()

    def enc(specs):
        zs = []
        with torch.no_grad():
            for i in range(0, len(specs), batch):
                x = torch.from_numpy(specs[i:i + batch]).to(dev)
                zs.append(model.encode(x)[0].cpu().numpy())
        return np.concatenate(zs)

    with h5py.File(out / "wild_specs.h5") as f:
        np.save(out / "wild_latent.npy", enc(f["specs"][:]))
    with h5py.File(out / "lab_exemplars.h5") as f:
        mine, off = enc(f["specs"][:]), f["latent_official"][:]
    np.save(out / "lab_exemplar_latent.npy", mine)
    cos = (mine * off).sum(1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(off, axis=1) + 1e-9)
    chk = {"n": int(len(cos)), "cosine_mean": float(cos.mean()), "cosine_median": float(np.median(cos))}
    (out / "repro_check.json").write_text(json.dumps(chk, indent=2))
    print("lab exemplar re-encode vs official:", chk)


def wild_peak(out, k=20):
    """Per wild segment: k-th brightest pixel mapped back to log units above the noise floor (perfreq specs)."""
    cal = json.loads((out / "floor_calibration.json").read_text())
    span = P["spec_max_val"] - P["spec_min_val"]
    with h5py.File(out / "wild_specs.h5") as f:
        S = f["specs"]
        v = np.concatenate([np.sort(S[i:i + 2048].reshape(-1, S.shape[1] * S.shape[2]), axis=1)[:, -k]
                            for i in range(0, len(S), 2048)])
    return span * v + cal["lo"]


def stage_compare(out, n_per_cohort, seed, min_peak=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    lab = load_lab_df(("cohort", "z_70", "latent_means"))
    idx = np.concatenate([rng.choice(np.flatnonzero(lab.cohort == c), min(n_per_cohort, (lab.cohort == c).sum()),
                                     replace=False) for c in sorted(lab.cohort.unique())])
    rng.shuffle(idx)
    ref, held = idx[: len(idx) * 9 // 10], idx[len(idx) * 9 // 10:]
    Zl, zl, cl = np.stack(lab.latent_means.values[ref]), lab.z_70.values[ref], lab.cohort.values[ref]
    Zh = np.stack(lab.latent_means.values[held])
    w = pd.read_csv(out / "wild.csv", index_col="index")
    Zw = np.load(out / "wild_latent.npy")
    if min_peak is not None:  # drop wild segments whose call barely clears the noise floor (all sets)
        w["peak"] = wild_peak(out)
        n0 = w.groupby("set").size()
        keep = (w.peak >= min_peak).values
        w, Zw = w[keep], Zw[keep]
        print("kept after peak cut:", (w.groupby("set").size() / n0).round(3).to_dict())
    calls = w.set.isin(["verified", "filtered"]).values

    # nearest-lab distance; baseline = held-out lab calls against the same reference set
    nn = NearestNeighbors(n_neighbors=15).fit(Zl)
    dh = nn.kneighbors(Zh, 1)[0][:, 0]
    dw, iw = nn.kneighbors(Zw)
    w["nn_dist"] = dw[:, 0]
    thr = np.percentile(dh, [95, 99])
    w["outside95"], w["outside99"] = w.nn_dist > thr[0], w.nn_dist > thr[1]
    votes = zl[iw]
    w["z70_knn"] = [np.bincount(v, minlength=70).argmax() for v in votes]
    w["z70_vote"] = [np.mean(v == m) for v, m in zip(votes, w.z70_knn)]

    # side-by-side UMAP: fit on lab, project wild
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit(Zl)
    El, Ew = reducer.embedding_, reducer.transform(Zw)
    w["umap_x"], w["umap_y"] = Ew[:, 0], Ew[:, 1]
    np.save(out / "lab_umap.npy", np.column_stack([El, zl]))
    # cluster usage: lab z_70 vs wild kNN-assigned z_70
    use = pd.DataFrame({f"lab {c}": np.bincount(zl[cl == c], minlength=70) / (cl == c).sum() for c in np.unique(cl)})
    for s in ("verified", "filtered"):
        m = (w.set == s).values
        use[f"wild {s}"] = np.bincount(w.z70_knn[m], minlength=70) / m.sum()
    use["lab"] = np.bincount(zl, minlength=70) / len(zl)
    use.index.name = "z_70"
    use.to_csv(out / "cluster_usage.csv")
    order = use.lab.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(18, 4))
    x = np.arange(70)
    for col, st in [("lab", dict(color="#555", width=0.4, label="lab (all)")),
                    ("wild filtered", dict(color="#c0392b", width=0.4, label="wild filtered"))]:
        ax.bar(x + (-0.2 if col == "lab" else 0.2), use.loc[order, col], **st)
    ax.plot(x + 0.2, use.loc[order, "wild verified"], "k.", ms=4, label="wild verified")
    ax.set_xticks(x, order, fontsize=6); ax.set_xlabel("paper GMM cluster (z_70), sorted by lab usage")
    ax.set_ylabel("share of calls"); ax.legend(frameon=False); ax.margins(x=0.005)
    fig.tight_layout(); fig.savefig(out / "cluster_usage.png", dpi=150, facecolor="white"); plt.close(fig)

    # wild clusters: PCA (99% var) + GMM k by BIC on wild calls, each matched to its nearest lab z_70 centroid
    Zc = Zw[calls]
    pca = PCA(0.99, random_state=seed).fit(Zl)
    Xc = pca.transform(Zc)
    ks = list(range(4, 41, 2))
    bic = [GaussianMixture(k, covariance_type="full", random_state=seed, n_init=2).fit(Xc).bic(Xc) for k in ks]
    k = ks[int(np.argmin(bic))]
    gmm = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=4).fit(Xc)
    w["wild_cluster"] = -1
    w.loc[calls, "wild_cluster"] = gmm.predict(Xc)
    cent = np.stack([Zl[zl == z].mean(0) for z in range(70)])
    rows = []
    for c in range(k):
        m = (w.wild_cluster == c).values
        mu = Zw[m].mean(0)
        d = np.linalg.norm(cent - mu, axis=1)
        rows.append({"wild_cluster": c, "n": int(m.sum()), "share": float(m.sum() / calls.sum()),
                     "verified": int((m & (w.set == "verified").values).sum()),
                     "nearest_z70": int(d.argmin()), "centroid_dist": float(d.min()),
                     "outside99": float(w.outside99[m].mean()),
                     "knn_agree": float((w.z70_knn[m] == d.argmin()).mean())})
    wc = pd.DataFrame(rows).sort_values("n", ascending=False)
    umap_figures(out, w, El, Ew, Zw, calls, seed)
    wc.to_csv(out / "wild_clusters.csv", index=False)
    w.to_csv(out / "wild_assigned.csv", index_label="index")
    summary = {"lab_ref": int(len(ref)), "lab_heldout": int(len(held)),
               "min_peak": min_peak, "wild_n": w.groupby("set").size().to_dict(),
               "lab_nn_dist_p95_p99": thr.tolist(),
               "outside99_by_set": w.groupby("set").outside99.mean().to_dict(),
               "outside95_by_set": w.groupby("set").outside95.mean().to_dict(),
               "wild_gmm_k": k, "bic": dict(zip(ks, map(float, bic))),
               "pca_dims": int(pca.n_components_)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    exemplar_figure(out, w, wc)


def umap_figures(out, w, El, Ew, Zw, calls, seed):
    """Lab-fit UMAP: lab density | each wild set alone (no lab points). Plus a UMAP fit on wild calls only."""
    import matplotlib.pyplot as plt
    import umap
    k = int(w.wild_cluster.max()) + 1
    cmap = np.vstack([plt.cm.tab20(np.arange(20)), plt.cm.tab20b(np.arange(20))])[:max(k, 1)]
    col = lambda m: cmap[w.wild_cluster.values[m] % len(cmap)]
    lim = np.percentile(np.vstack([El, Ew]), [0.5, 99.5], axis=0)
    sets = ("verified", "filtered", "filtered_low", "noncall")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6), sharex=True, sharey=True)
    axes[0].hexbin(El[:, 0], El[:, 1], gridsize=80, bins="log", cmap="viridis", mincnt=1,
                   extent=(*lim[:, 0], *lim[:, 1]))
    axes[0].set_title(f"lab, all cohorts (n={len(El):,})")
    for ax, st in zip(axes[1:], sets):
        m = (w.set == st).values
        c = col(m) if st in ("verified", "filtered") else "#555555"
        ax.scatter(Ew[m, 0], Ew[m, 1], s=1.5 if m.sum() > 5000 else 4, c=c, alpha=0.5, lw=0, rasterized=True)
        ax.set_title(f"wild {st} (n={m.sum():,})")
    for ax in axes:
        ax.set_xlim(lim[:, 0]); ax.set_ylim(lim[:, 1]); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out / "umap_side_by_side.png", dpi=150, facecolor="white"); plt.close(fig)

    E = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(Zw[calls])
    np.save(out / "wild_umap.npy", E)
    lab_c = w.wild_cluster.values[calls]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(E[:, 0], E[:, 1], s=1, c=cmap[lab_c % len(cmap)], alpha=0.5, lw=0, rasterized=True)
    for c in range(k):
        x, y = np.median(E[lab_c == c], axis=0)
        ax.text(x, y, f"W{c}", fontsize=8, ha="center", va="center", weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"wild calls only (n={calls.sum():,})")
    fig.tight_layout(); fig.savefig(out / "umap_wild_only.png", dpi=150, facecolor="white"); plt.close(fig)


def exemplar_figure(out, w, wc, n_ex=5):
    """Row per wild cluster: wild exemplars (closest to centroid, verified first) | exemplars of nearest lab z_70."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Zw = np.load(out / "wild_latent.npy")[w.index.values]
    with h5py.File(out / "wild_specs.h5") as f:
        ws = f["specs"]
        lex = pd.read_csv(out / "lab_exemplars.csv", index_col="index")
        with h5py.File(out / "lab_exemplars.h5") as g:
            ls = g["specs"][:]
        fig, axes = plt.subplots(len(wc), 2 * n_ex + 1, figsize=(2 * n_ex + 1.5, 1.05 * len(wc)),
                                 gridspec_kw={"width_ratios": [1] * n_ex + [0.25] + [1] * n_ex})
        for r, row in enumerate(wc.itertuples()):
            m = np.flatnonzero((w.wild_cluster == row.wild_cluster).values)
            d = np.linalg.norm(Zw[m] - Zw[m].mean(0), axis=1)
            pick = m[np.lexsort((d, (w.set.values[m] != "verified")))][:n_ex]
            lpick = lex[lex.z_70 == row.nearest_z70].sort_values("conf", ascending=False).index[:n_ex]
            for j in range(2 * n_ex + 1):
                ax = axes[r, j]; ax.set_xticks([]); ax.set_yticks([])
                if j == n_ex:
                    ax.axis("off"); continue
                src = pick if j < n_ex else lpick
                k = j if j < n_ex else j - n_ex - 1
                if k < len(src):
                    ax.imshow(ws[w.index.values[src[k]]] if j < n_ex else ls[src[k]], origin="lower", cmap="magma",
                              vmin=0, vmax=1, aspect="auto")
                else:
                    ax.axis("off")
            axes[r, 0].set_ylabel(f"W{row.wild_cluster}\n{row.share:.0%}", rotation=0, ha="right", va="center",
                                  fontsize=8)
            axes[r, n_ex + 1].set_ylabel(f"z{row.nearest_z70}", rotation=0, ha="right", va="center", fontsize=8)
        axes[0, n_ex // 2].set_title("wild cluster", fontsize=9)
        axes[0, n_ex + 1 + n_ex // 2].set_title("nearest lab cluster", fontsize=9)
        fig.subplots_adjust(wspace=0.05, hspace=0.08, left=0.06, right=0.99, top=0.97, bottom=0.01)
        fig.savefig(out / "clusters_side_by_side.png", dpi=130, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("specs", "encode", "compare", "figure"))
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/ava/lab_vs_wild")
    ap.add_argument("--checkpoint", type=Path, default=DRYAD / "checkpoint_050.tar")
    ap.add_argument("--per-cluster", type=int, default=12, help="lab exemplars per z_70 cluster")
    ap.add_argument("--n-per-cohort", type=int, default=20_000, help="lab latents per cohort for UMAP/kNN")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-peak-khz", type=float, default=25.0)
    ap.add_argument("--min-peak", type=float, default=None,
                    help="compare: drop wild segments whose 20th-brightest pixel is below this many log units above "
                         "the noise floor (1.48 = lab 1st percentile)")
    ap.add_argument("--floor", choices=("perfreq", "none"), default="perfreq")
    ap.add_argument("--ctx", type=float, default=0.25, help="s of context each side for the noise floor")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "specs":
        stage_specs(a.out, a.per_cluster, a.min_peak_khz, a.floor, a.ctx, a.seed)
    elif a.stage == "encode":
        stage_encode(a.out, a.checkpoint, a.batch)
    elif a.stage == "compare":
        stage_compare(a.out, a.n_per_cohort, a.seed, a.min_peak)
    else:
        exemplar_figure(a.out, pd.read_csv(a.out / "wild_assigned.csv", index_col="index"),
                        pd.read_csv(a.out / "wild_clusters.csv"))


if __name__ == "__main__":
    main()
