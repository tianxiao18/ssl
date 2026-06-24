# Scripts

## Workflow

### Step 1 — Generate spectrograms

```bash
python scripts/gen_spectrograms.py <recording_dir> <spec_dir> [--channels 118,35] [--chunk-sec 1.0]
```

Slices raw audio into 1-second chunks and writes each as a normalised dB spectrogram PNG.
All step-2 scripts read from `spec_dir`; run this first.

---

### Step 2 — Run models (any order, all read `spec_dir`, all write `coco_ch_<ch>.json`)

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

---

### Step 3 — Evaluate

Compare any two step-2 outputs. DAS/YOLO is treated as ground truth.

```bash
python scripts/evaluate.py <spec_dir> <gt_dir> <pred_dir> <out_dir> [--channels 118,35] [--cols 10]
```

Writes per-channel montage pages (3 rows: raw / GT / prediction) and prints
Precision, Recall, F1, and mean matched IoU to stdout.

---

### Training (separate concern)

```bash
python scripts/finetune_squeakout.py <recording_dir> <coco_json> <out_dir>
```

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
