"""Fine-tuned SAM3 ("trace" text prompt): seg (own mask head) or box2mask (boxes -> original SAM3 masks)."""
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vox_tracer.coco import image_entry, make_coco, mask_to_polygons, save_coco_per_channel
from vox_tracer.spec import WindowReader, spec_windows_by_channel, window_fname

ROOT = Path(__file__).resolve().parents[1]
BPE = ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
BASE = ROOT / "sam3" / "sam3.pt"
RELEASE = "https://github.com/tianxiao18/ssl/releases/download/sam3-ft-v1"
# variant -> (mask mode, default checkpoint)
VARIANTS = {"seg": ("head", ROOT / "checkpoints" / "sam3_ft_seg.pt"),
            "box2mask": ("box", ROOT / "checkpoints" / "sam3_ft_box2mask.pt")}

_MODELS = {}


def _device():
    try:
        from sam3.device_utils import DEVICE
    except ImportError:  # upstream sam3 has no device_utils
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE.type


def fetch_checkpoint(path):
    """Download a default checkpoint from the GitHub release if it is not on disk."""
    path = Path(path)
    if not path.exists():
        url = f"{RELEASE}/{path.name}"
        print(f"Downloading {url} -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    return path


def load_model(ckpt, mask_mode, threshold, base=BASE):
    """(model, processor) for SAM3 from `base` with the fine-tuned weights in `ckpt` ('none' = zero-shot); cached."""
    key = (str(ckpt), mask_mode, threshold, str(base))
    if key in _MODELS:
        return _MODELS[key]
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(bpe_path=str(BPE), checkpoint_path=str(base), load_from_HF=False,
                                   enable_segmentation=True, enable_inst_interactivity=mask_mode == "box",
                                   device=_device())
    if str(ckpt) != "none":
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)  # trainer checkpoint or weights-only export
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"{ckpt}: {len(sd)} tensors loaded, {len(missing)} kept from {base}, {len(unexpected)} unexpected")
        assert not unexpected, unexpected[:5]
    _MODELS[key] = model, Sam3Processor(model, device=_device(), confidence_threshold=threshold)
    return _MODELS[key]


@torch.inference_mode()
def detect(model, proc, img, prompt, mask_mode):
    """[(xyxy box, score, mask uint8 HxW)] for one window (PIL RGB image)."""
    dev = proc.device if isinstance(proc.device, str) else proc.device.type
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        state = proc.set_image(img)
        state = proc.set_text_prompt(prompt, state)
    boxes = state["boxes"].float().cpu().numpy()
    scores = state["scores"].float().cpu().numpy()
    if not len(boxes):
        return []
    if mask_mode == "head":
        masks = state["masks"].squeeze(1).cpu().numpy()
    else:
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            masks, _, _ = model.predict_inst(state, point_coords=None, point_labels=None,
                                             box=boxes, multimask_output=False)
        masks = np.asarray(masks).reshape(len(boxes), img.height, img.width)
    return [(b, float(s), (m > 0).astype(np.uint8)) for b, s, m in zip(boxes, scores, masks)]


def run_sam3_ft(spec_dir, out_dir, channels=None, variant="seg", checkpoint=None, threshold=0.5,
                prompt="trace", prefix="headmic", chunk_sec=1.0, overwrite=False, mask_mode=None, base=BASE):
    """Run fine-tuned SAM3 on one recording's spectrograms (PNG or HDF5); write coco_ch_<ch>.json per channel."""
    mask_mode = mask_mode or VARIANTS[variant][0]
    if checkpoint is None:
        checkpoint = fetch_checkpoint(VARIANTS[variant][1])
    by_ch = spec_windows_by_channel(spec_dir, channels, prefix=prefix, chunk_sec=chunk_sec)
    pending = {ch: e for ch, e in by_ch.items() if overwrite or not (Path(out_dir) / f"coco_ch_{ch}.json").exists()}
    if not pending:
        print(f"{out_dir}: all channels done, skipping.")
        return
    model, proc = load_model(checkpoint, mask_mode, threshold, base)
    read = WindowReader()
    coco_by_ch = {}
    for ch, entries in pending.items():
        coco = make_coco(f"SAM3 fine-tuned '{prompt}', masks={mask_mode}, ch {ch}", "vox")
        for k, (src, t0, t1) in enumerate(entries):
            gray = read(src, t0, t1)
            h, w = gray.shape
            coco["images"].append(image_entry(k, window_fname(src, t0, t1, chunk_sec), w, h,
                                              window_start_sec=t0, window_end_sec=t1))
            img = Image.fromarray(gray).convert("RGB")
            for (x0, y0, x1, y1), score, mask in detect(model, proc, img, prompt, mask_mode):
                polys = mask_to_polygons(mask, (h, w)) if mask.any() else []
                coco["annotations"].append({
                    "id": len(coco["annotations"]), "image_id": k, "category_id": 1, "iscrowd": 0,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "area": float(mask.sum()), "score": score, "segmentation": polys})
        coco_by_ch[ch] = coco
    read.close()
    save_coco_per_channel(coco_by_ch, out_dir)
