# ssl

Detecting and segmenting rodent ultrasonic vocalizations (USVs) from multi-channel
audio, across several datasets (gerbil_ssl, dryad_gerbil, gerbil_family) and
segmentation methods (ridge filter, SqueakOut, SAM3, DAS+YOLO). Detected calls
feed into evaluation against ground truth and, downstream, into VAE-based latent
analysis and clustering/UMAP visualization.

`scripts/` follows the pipeline: spectrograms → segmentation → (optional)
evaluation/comparison at the top level, everything else in a subfolder.

## Main pipeline

### Step 1 — Generate spectrograms

```bash
python scripts/gen_spectrograms.py --dataset gerbil_ssl --all --workers 8
```

Slices raw audio into 1-second chunks and writes each as a normalised dB spectrogram PNG
to `outputs/spectrograms/<dataset>/`. `--dataset` (`gerbil_ssl`, `dryad_gerbil`, or
`gerbil_family`) is required and determines both the input (`data/<dataset>/`) and
output location; run this first.

---

### Step 2 — Run segmentation

All models are invoked through a single unified script, `scripts/run.py <model> <out_dir>
--dataset <dataset> [--all | --recording <exp>/<idx>] [options]` — `out_dir` is just the
stage/variant name (e.g. `outputs/ridge_flatness`), and the script appends `/<dataset>`
(and `/<recording>` in single-recording mode) itself, writing to
`<out_dir>/<dataset>[/<recording>]/coco_ch_<ch>.json`.

**Ridge filter** (default: sato; choices: sato, meijering, frangi, hessian):
```bash
python scripts/run.py ridge outputs/ridge_flatness --dataset gerbil_ssl --all --workers 16
```

**SqueakOut:**
```bash
python scripts/run.py squeakout outputs/squeakout --dataset gerbil_ssl --all --workers 1 --checkpoint squeakout/squeakout_weights.ckpt
```

**SAM3** (requires `pip install -e sam3/ --no-deps` and checkpoint):
```bash
python scripts/run.py sam3 outputs/sam3 --dataset gerbil_ssl --all --sam3-checkpoint sam3/sam3.pt
```

**DAS + YOLO** (needs raw audio + DAS CSV; spectrograms must be pre-generated):
```bash
python scripts/run.py das_yolo outputs/das_yolo --dataset gerbil_ssl --recording experiment_384/idx_000
```

Method-specific variants and post-processing tools beyond this dispatcher live in
[`segmentation/`](#segmentation).

---

### Step 3 — Evaluate (optional, requires ground truth)

Compare a step-2 output against ground truth (DAS/YOLO or a `*annotations_gt.csv`).
Unlike `run.py`/`gen_spectrograms.py`, `evaluate.py` has no `.sbatch` wrapper (run
manually) and takes plain paths — no `--dataset` flag, so pass the dataset segment
yourself in every path.

```bash
python scripts/evaluate.py outputs/spectrograms/gerbil_ssl outputs/das_yolo/gerbil_ssl data/gerbil_ssl outputs/squeakout/gerbil_ssl outputs/eval/gerbil_ssl/squeakout --all --workers 4
```

Writes `metrics.csv` (TP/FP/FN/Recall/Precision/F1 per session/channel) and,
optionally, montage pages (`--montage-samples N`) and a full-session
visualization strip (`--viz-session`). Run once per method into its own output dir
under `outputs/eval/<dataset>/` (e.g. `outputs/eval/gerbil_ssl/squeakout`,
`outputs/eval/gerbil_ssl/sam3`, `outputs/eval/gerbil_ssl/ridge`).

---

### Step 4 — Compare methods (optional, requires Step 3 for each method)

```bash
python scripts/compare_methods.py outputs/eval/gerbil_ssl --methods ridge,sam3,squeakout --out outputs/eval/gerbil_ssl/comparison.png
```

Reads each method's `metrics.csv` from `outputs/eval/gerbil_ssl/ridge/`,
`outputs/eval/gerbil_ssl/sam3/`, `outputs/eval/gerbil_ssl/squeakout/` (one call to
`evaluate.py` per method, per Step 3) and writes one grouped bar chart
(Recall/Precision/F1 per method, micro-averaged across recordings with
per-recording std error bars).
