# Scripts

`scripts/` is organized around the main pipeline — spectrograms → segmentation →
(optional) evaluation/comparison — which stays at the top level. Everything
else (method-specific variants, extra eval/viz tools, downstream VAE/clustering
analysis, hyperparameter search, dataset ground-truth prep, Slurm jobs, and
superseded experiments) lives in a subfolder. All top-level and stage scripts
are dataset-agnostic and method-agnostic unless noted.

## Main pipeline

### Step 1 — Generate spectrograms

```bash
python scripts/gen_spectrograms.py <recording_dir> <spec_dir> [--channels 118,35] [--chunk-sec 1.0]
```

Slices raw audio into 1-second chunks and writes each as a normalised dB spectrogram PNG.
All step-2 scripts read from `spec_dir`; run this first.

---

### Step 2 — Run segmentation (any order, all read `spec_dir`, all write `coco_ch_<ch>.json`)

All models are invoked through a single unified script:

```bash
python scripts/run.py <model> <args...>
```

**Ridge filter** (default: sato; choices: sato, meijering, frangi, hessian):
```bash
python scripts/run.py ridge <spec_dir> <out_dir> [--filter sato] [--channels 118,35]
```

**SqueakOut:**
```bash
python scripts/run.py squeakout <spec_dir> <out_dir> [--checkpoint path/to/weights.ckpt]
```

**SAM3** (requires `pip install -e sam3/ --no-deps` and checkpoint):
```bash
python scripts/run.py sam3 <spec_dir> <out_dir> [--sam3-checkpoint sam3/sam3.pt]
```

**DAS + YOLO** (needs raw audio + DAS CSV; spectrograms must be pre-generated):
```bash
python scripts/run.py das_yolo <recording_dir> <spec_dir> <out_dir>
```

Method-specific variants and post-processing tools beyond this dispatcher live in
[`segmentation/`](#segmentation).

---

### Step 3 — Evaluate (optional, requires ground truth)

Compare a step-2 output against ground truth (DAS/YOLO or a `*annotations_gt.csv`).

```bash
python scripts/evaluate.py <spec_dir> <das_dir> <recording_dir> <pred_dir> <out_dir> [--channels 118,35] [--cols 10]
```

Writes `metrics.csv` (TP/FP/FN/Recall/Precision/F1 per session/channel) and,
optionally, montage pages (`--montage-samples N`) and a full-session
visualization strip (`--viz-session`). Run once per method into its own
`out_dir` (e.g. `outputs/eval/<method>/`).

---

### Step 4 — Compare methods (optional, requires Step 3 for each method)

```bash
python scripts/compare_methods.py <eval_base> [--methods ridge,sam3,squeakout] [--out fig.png]
```

Reads each method's `metrics.csv` from `<eval_base>/<method>/` and writes one
grouped bar chart (Recall/Precision/F1 per method, micro-averaged across
recordings with per-recording std error bars).

---

## Subfolders

### `segmentation/`

Method-specific variants and post-processing tools that sit alongside `run.py`
rather than through it:
- `run_sam3_calltype.py` — SAM3 variant gated by VAE call-type match (dryad_gerbil only).
- `apply_mask_filter.py` — re-filter an existing COCO tree with stage-2 mask filters, no GPU re-run.
- `finetune_squeakout.py` — fine-tune a SqueakOut checkpoint on GT masks (run manually; no sbatch wrapper).

### `evaluation/`

Extra evaluation/visualization tools beyond `evaluate.py` + `compare_methods.py`:
- `pr_curves.py` — precision-recall curves by sweeping a per-detection score threshold post-hoc.
- `bout_analysis.py` — bout-level (runs of temporally-close GT calls) recall/precision/F1.
- `compare_bouts.py` — cross-method bout-accuracy comparison, binned by bout size.
- `viz_bouts.py` — per-bout montages (spectrogram + GT + predictions) for worst/longest bouts.
- `spectral_features.py` — librosa spectral features per detected event.

### `vae/`

Downstream analysis: fitting a VAE to detected calls (data prep, training, validation):
- `coco_to_ava_specs.py` — crop COCO mask regions into paired raw/masked AVA-format spectrogram HDF5.
- `dryad_to_ava_specs.py` — regenerate AVA-format spectrograms for dryad_gerbil, paired with the paper's own latents.
- `train_ava_ab.py` — train paired AVA VAEs (raw vs. masked) to isolate the mask-denoising effect.
- `train_qmc_ab.py` — train a QMC latent-variable model on the same data, compare against the AVA VAE.
- `confirm_vae_against_paper.py` — validate a local VAE checkpoint's latents against the paper's official latents.

### `clustering/`

Downstream analysis: clustering / UMAP visualization of fitted latents (dryad_gerbil only):
- `cluster_exemplars.py` — plot the most-representative spectrogram per z_70 GMM cluster.
- `reproduce_umap_figure.py` — reproduce the paper's UMAP figure (official vs. local-checkpoint latents).

### `hparam_search/`

- `random_search.py` — random joint hyperparameter search over the SAM3 ridge-exemplar space.
- `sweep_hparams.py` — one-at-a-time hyperparameter sweep (baseline ± single variable).
- `plot_search_by_hparam.py` — plot `random_search.py`'s PR cloud, one panel per hyperparameter.

### `dataset_prep/`

Materialize per-recording `*annotations_gt.csv` (consumed by `evaluate.py`) for datasets
without one already:
- `dryad_gt_to_csv.py` — dryad_gerbil.
- `gerbil_family_gt_to_csv.py` — gerbil_family.

### `sbatch/`

All Slurm batch scripts (13), one per pipeline/downstream/hparam-search job. Each
`cd`s to the repo root and invokes the corresponding `.py` by its path under `scripts/`
(e.g. `run_sam3.sbatch` calls `scripts/run.py sam3 ...`,
`train_ava_ab.sbatch` calls `scripts/vae/train_ava_ab.py`).

### `archive/`

Superseded ridge-tuning experiments, one-off plotting scripts, and dead code kept
for reference. Not part of the active pipeline.

---

## Library (`vox_tracer/`)

Model logic lives in the package so scripts stay thin:

| Module | Contents |
|---|---|
| `spec.py` | Spectrogram generation, audio loading, chunk slicing |
| `coco.py` | COCO JSON builders, `mask_to_polygons` |
| `montage.py` | Cell resize, overlay helpers, page saving |
| `das_yolo.py` | DAS event loading, YOLO inference, NMS |
| `squeakout_runner.py` | SqueakOut batch inference, mask cleaning |
| `ridge.py` | Ridge filter segmentation (`compute_seg_mask`) |
| `sam3_runner.py` | SAM3 inference, ridge-based candidate selection |
| `call_type_gate.py` | Gate SAM3 candidates by VAE call-type match (dryad_gerbil) |
| `scoring.py` | GT loading, 1-D time-IoU matching, event-level scoring (shared by `evaluate.py`, `evaluation/pr_curves.py`, `evaluation/bout_analysis.py`) |
| `sweep_core.py` | Shared ground-truth loading, scoring, and GPU inference pass for `hparam_search/` |
