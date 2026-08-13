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
python scripts/gen_spectrograms.py data/gerbil_ssl outputs/spectrograms/gerbil_ssl --all --workers 8
```

Slices raw audio into 1-second chunks and writes each as a normalised dB spectrogram PNG.
All step-2 scripts read from `spec_dir`; run this first.

---

### Step 2 — Run segmentation (any order, all read `spec_dir`, all write `coco_ch_<ch>.json`)

All models are invoked through a single unified script, `scripts/run.py`, with the
model name (`ridge`, `squeakout`, `sam3`, `das_yolo`) as the first argument:

**Ridge filter** (default: sato; choices: sato, meijering, frangi, hessian):
```bash
python scripts/run.py ridge outputs/spectrograms/gerbil_ssl outputs/ridge_flatness --all --workers 16
```

**SqueakOut:**
```bash
python scripts/run.py squeakout outputs/spectrograms/gerbil_ssl outputs/squeakout --all --workers 1 --checkpoint squeakout/squeakout_weights.ckpt
```

**SAM3** (requires `pip install -e sam3/ --no-deps` and checkpoint):
```bash
python scripts/run.py sam3 outputs/spectrograms/gerbil_ssl outputs/sam3 --sam3-checkpoint sam3/sam3.pt
```

**DAS + YOLO** (needs raw audio + DAS CSV; spectrograms must be pre-generated):
```bash
python scripts/run.py das_yolo data/gerbil_ssl/experiment_384/idx_000 outputs/spectrograms/gerbil_ssl/experiment_384/idx_000 outputs/das_yolo/experiment_384/idx_000
```

Method-specific variants and post-processing tools beyond this dispatcher live in
[`segmentation/`](#segmentation).

---

### Step 3 — Evaluate (optional, requires ground truth)

Compare a step-2 output against ground truth (DAS/YOLO or a `*annotations_gt.csv`).

```bash
python scripts/evaluate.py outputs/spectrograms/gerbil_ssl outputs/das_yolo data outputs/squeakout outputs/eval/squeakout --all --workers 4
```

Writes `metrics.csv` (TP/FP/FN/Recall/Precision/F1 per session/channel) and,
optionally, montage pages (`--montage-samples N`) and a full-session
visualization strip (`--viz-session`). Run once per method into its own
output dir (e.g. `outputs/eval/squeakout`, `outputs/eval/sam3`, `outputs/eval/ridge`).

---

### Step 4 — Compare methods (optional, requires Step 3 for each method)

```bash
python scripts/compare_methods.py outputs/eval --methods ridge,sam3,squeakout --out outputs/eval/comparison.png
```

Reads each method's `metrics.csv` from `outputs/eval/ridge/`, `outputs/eval/sam3/`,
`outputs/eval/squeakout/` (one call to `evaluate.py` per method, per Step 3) and
writes one grouped bar chart (Recall/Precision/F1 per method, micro-averaged
across recordings with per-recording std error bars).
