# ssl

Detecting and segmenting rodent ultrasonic vocalizations (USVs) from multi-channel
audio, across several datasets (gerbil_ssl, dryad_gerbil, gerbil_family) and
segmentation methods (ridge filter, SqueakOut, SAM3, DAS+YOLO). Detected calls
feed into evaluation against ground truth and, downstream, into VAE-based latent
analysis and clustering/UMAP visualization.

`scripts/` follows the pipeline: spectrograms → segmentation → (optional)
evaluation/comparison at the top level, everything else in a subfolder.

## Setup

```bash
git clone --recurse-submodules https://github.com/tianxiao18/ssl.git
cd ssl
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # Linux/CUDA
# pip install -r requirements-cpu.txt   # Mac or Windows (no CUDA)
pip install -e sam3/ --no-deps
hf auth login
hf download facebook/sam3 --local-dir sam3
pip install inference==1.2.12 --no-deps  # DAS+YOLO only
```

First-time SAM3 users must request access at https://huggingface.co/facebook/sam3 before `hf download` will work.

For SqueakOut, place the repo at `squeakout/` with weights at `squeakout/squeakout_weights.ckpt`.

The fine-tuned SAM3 weights download to `checkpoints/` on first use, from the
[`sam3-ft-v1`](https://github.com/tianxiao18/ssl/releases/tag/sam3-ft-v1) release.

## Main pipeline

### Step 1 — Generate spectrograms

```bash
python scripts/gen_spectrograms.py --dataset gerbil_ssl --workers 8
python scripts/gen_spectrograms.py --dataset my_dataset --prefix mic --lo -95 --hi -57 --workers 8
```

Reads `data/<dataset>/**/{prefix}_{ch}_*.wav` (one folder per recording, any depth) and writes
1-second dB spectrogram PNGs to `outputs/spectrograms/<dataset>/`. `--lo/--hi` set the dB range for
your rig (`--help` lists known values). `scripts/gen_spectrograms_h5.py` does the same as one HDF5
per recording, for long recordings.

---

### Step 2 — Run segmentation

All models are invoked through a single unified script,
`scripts/run.py <model> [out_dir] --dataset <dataset> [options]`. It writes
`<out_dir>/<dataset>/<recording>/coco_ch_<ch>.json`; `out_dir` defaults to `outputs/<model>`.

**Fine-tuned SAM3** (text prompt "trace", fine-tuned on gerbil_ssl; GPU recommended). Two variants:
`sam3_ft_seg` (masks from the fine-tuned mask head) and `sam3_ft_box2mask` (fine-tuned boxes, masks
from the original SAM3).
```bash
python scripts/run.py sam3_ft_seg --dataset gerbil_ssl
python scripts/run.py sam3_ft_box2mask --dataset my_dataset --prefix mic
```
`--threshold` (default 0.5) trades recall for precision.

**Ridge filter** (default: sato; choices: sato, meijering, frangi, hessian):
```bash
python scripts/run.py ridge outputs/ridge_flatness --dataset gerbil_ssl --workers 16
```

**SqueakOut:**
```bash
python scripts/run.py squeakout --dataset gerbil_ssl
```

**SAM3, zero-shot** (box prompts from the sato ridge filter):
```bash
python scripts/run.py sam3 --dataset gerbil_ssl --sam3-checkpoint sam3/sam3.pt
```

**DAS + YOLO** (needs raw audio + DAS CSV; spectrograms must be pre-generated):
```bash
python scripts/run.py das_yolo --dataset gerbil_ssl --recording experiment_384/idx_000
```

Method-specific variants and post-processing tools beyond this dispatcher live in
[`segmentation/`](#segmentation).

---

### Step 3 — Evaluate (optional, requires ground truth)

Compare a step-2 output against ground truth (DAS/YOLO or a `*annotations_gt.csv`).
`evaluate.py` takes plain paths (no `--dataset` flag), so pass the dataset segment yourself in
every path. Pointing them at a single recording's folders evaluates just that recording.

```bash
python scripts/evaluate.py outputs/spectrograms/gerbil_ssl outputs/das_yolo/gerbil_ssl data/gerbil_ssl \
    outputs/sam3_ft_seg/gerbil_ssl outputs/eval/gerbil_ssl/sam3_ft_seg --workers 4
```

Writes `metrics.csv` (TP/FP/FN/Recall/Precision/F1 per session/channel) and,
optionally, montage pages (`--montage-samples N`) and a full-session
visualization strip (`--viz-session`). Run once per method into its own output dir
under `outputs/eval/<dataset>/` (e.g. `outputs/eval/gerbil_ssl/sam3_ft_seg`,
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

---

## Evaluation GUI

Efficient benchmarking/labeling of candidate detections pooled from several detectors, used to
estimate each detector's (and each ensemble rule's) precision and relative recall with
exact confidence intervals.

### 1. Prediction format

One directory per detector, same layout as Step 2's output:

```
outputs/<detector>/<dataset>/<experiment>/<idx>/coco_ch_<ch>.json
```

Each file is [standard COCO](https://cocodataset.org/#format-data) plus
`window_start_sec` / `window_end_sec` on each image. For interval-only detectors, see
`scripts/das_to_coco.py`.

### 2. Build the candidate pool

```bash
python scripts/build_candidate_pool.py --dataset gerbil_ssl \
    --detectors sam3_best,ridge,squeakout,das_yolo \
    --out outputs/label_campaigns/<name>/candidates.csv
```

Detections from all detectors and channels are put on one time axis, and overlapping
ones are merged into one candidate. Only clips covered by every detector are used.

### 3. Make the campaign

```bash
python scripts/make_campaign.py --pool outputs/label_campaigns/<name>/candidates.csv \
    --detectors sam3_best,ridge,squeakout,das_yolo \
    --out outputs/label_campaigns/<name> --n-max 6000 --checkpoints 12
```

This freezes the annotation budget and checkpoints into `campaign.json`, which can never
be edited afterwards. Add `--plan-only` first to see what a budget buys without writing
anything. The dataset is detected from the pool; pass `--dataset` if it is ambiguous.
If spectrograms or audio live elsewhere, pass `--spectrograms <dir>` / `--audio <dir>`
(folders holding `<dataset>/`); the campaign then records absolute paths and opens
from any folder.

### 4. Launch the GUI

```bash
python scripts/label_gui.py --annotator <your_name>
```

Run it from the folder that holds `outputs/`. Open the printed URL, pick the campaign
and start; labeling resumes where it left off. Over SSH, forward the port first:
`ssh -L 8765:localhost:8765 <workstation>`.
