"""Run a fine-tuned SAM3 ("trace" text prompt) on spectrogram windows; write coco_ch_<ch>.json per recording.
Research entry point (any checkpoint, --images-json); for new datasets use `scripts/run.py sam3_ft_seg|sam3_ft_box2mask`.

Masks come from one of:
  head  the fine-tuned model's own mask head (option 3: trained with mask loss).
  box   the detector's boxes fed as box prompts to SAM3's SAM1-style predictor (option 2); the
        predictor reads the frozen trunk features, so it is the original SAM3 segmenter.

    python scripts/segmentation/run_sam3_ft.py \
        --ckpt outputs/sam3_ft/runs/trace_seg_frozen_v1/checkpoints/checkpoint.pt --mask-mode head \
        --split-json outputs/sam3_ft/gerbil_ssl/val/_annotations.coco.json --out outputs/sam3_ft_head

Spectrograms are read from outputs/spectrograms/<dataset> (1 s PNGs) or, if that does not exist,
outputs/spectrograms_h5/<dataset> (one HDF5 per recording, e.g. dryad_gerbil_full):

    python scripts/segmentation/run_sam3_ft.py --ckpt none --mask-mode head --threshold 0.2 \
        --dataset dryad_gerbil_full --channels 0 --prefix mic --recordings-file recs.txt --out outputs/sam3_text_dryad

--images-json runs on a fine-tuning dataset's own (masked) images instead and writes COCO results
(list of detections with RLE masks) to --out, for scripts/evaluation/eval_sam3_ft.py.
--ckpt none gives zero-shot SAM3 (same prompt) as a baseline.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vox_tracer.sam3_ft_runner import detect, load_model, run_sam3_ft  # noqa: E402


def run_images(model, proc, images_json, out, prompt, mask_mode):
    """COCO results for every image of a fine-tuning dataset split."""
    c = json.load(open(images_json))
    res = []
    for im in c["images"]:
        img = Image.open(Path(images_json).parent / im["file_name"]).convert("RGB")
        for (x0, y0, x1, y1), score, mask in detect(model, proc, img, prompt, mask_mode):
            rle = mask_util.encode(np.asfortranarray(mask))
            res.append({"image_id": im["id"], "category_id": 1, "score": score,
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "segmentation": {"size": rle["size"], "counts": rle["counts"].decode()}})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"{len(res)} detections on {len(c['images'])} images -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trainer checkpoint (dict with 'model'), or 'none' for zero-shot")
    ap.add_argument("--base", default=str(ROOT / "sam3" / "sam3.pt"), help="weights not in --ckpt come from here")
    ap.add_argument("--mask-mode", choices=["head", "box"], required=True)
    ap.add_argument("--prompt", default="trace")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--dataset", default="gerbil_ssl")
    ap.add_argument("--channels", default="35,118")
    ap.add_argument("--prefix", default="headmic", help="spectrogram file prefix (mic for dryad)")
    ap.add_argument("--recordings-file", help="one recording (relative to the dataset) per line")
    ap.add_argument("--split-json", help="only recordings whose clips appear in this dataset json (e.g. val)")
    ap.add_argument("--recording", help="single experiment_X/idx_Y")
    ap.add_argument("--images-json", help="run on this dataset split's images; --out is then a results .json")
    ap.add_argument("--out", required=True, help="out_dir; <out>/<dataset>/<rel>/coco_ch_<ch>.json")
    a = ap.parse_args()

    if a.images_json:
        model, proc = load_model(a.ckpt, a.mask_mode, a.threshold, a.base)
        return run_images(model, proc, a.images_json, a.out, a.prompt, a.mask_mode)

    spec_base = ROOT / "outputs" / "spectrograms" / a.dataset
    if not spec_base.exists():
        spec_base = ROOT / "outputs" / "spectrograms_h5" / a.dataset
    if a.recording:
        rels = [a.recording]
    elif a.recordings_file:
        rels = [x.strip() for x in open(a.recordings_file) if x.strip()]
    elif a.split_json:
        clips = {im["clip"] for im in json.load(open(a.split_json))["images"]}
        rels = sorted({"experiment_{}/idx_{:03d}".format(*map(int, re.search(r"exp(\d+)_idx(\d+)", c).groups()))
                       for c in clips})
    else:
        rels = sorted(str(p.parent.relative_to(spec_base)) for p in spec_base.glob("*/*/headmic_*_chunk_00000_*.png"))
        rels = sorted(set(rels))
    load_model(a.ckpt, a.mask_mode, a.threshold, a.base)  # cached for run_sam3_ft
    channels = [int(c) for c in a.channels.split(",")]
    for i, rel in enumerate(rels):
        out = Path(a.out) / a.dataset / rel
        if all((out / f"coco_ch_{ch}.json").exists() for ch in channels):
            continue  # resumable
        run_sam3_ft(spec_base / rel, out, channels, checkpoint=a.ckpt, threshold=a.threshold, prompt=a.prompt,
                    prefix=a.prefix, mask_mode=a.mask_mode, base=a.base)
        print(f"{i + 1}/{len(rels)} {rel}", flush=True)


if __name__ == "__main__":
    main()
