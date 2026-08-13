"""SqueakOut inference over pre-generated spectrogram PNGs."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from vox_tracer.coco import image_entry, make_coco, mask_to_polygons, poly_annotation, save_coco_per_channel
from vox_tracer.paths import recording_dir_from_spec_dir
from vox_tracer.ridge import detection_spectral_features, passes_mask_filters
from vox_tracer.spec import group_specs_by_channel, load_channel_audio

INFERENCE_BATCH_SIZE = 16


def clean_mask(mask, min_area_ratio=0.1, min_component=100, min_total=300):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas):
        return mask
    rel_threshold = max(areas) * min_area_ratio
    clean = np.zeros_like(mask)
    for label, area in enumerate(areas, start=1):
        if area >= rel_threshold and area >= min_component:
            clean[labels == label] = 255
    if (clean > 0).sum() < min_total:
        return np.zeros_like(mask)
    return clean


def _run_batch(paths, model, device, batch_size=INFERENCE_BATCH_SIZE, mask_threshold=None):
    from squeakout.data import DEFAULT_IMAGE_SIZE, load_spectrogram_tensor
    from squeakout.inference import DEFAULT_MASK_THRESHOLD, logits_to_mask

    thr = DEFAULT_MASK_THRESHOLD if mask_threshold is None else mask_threshold
    out = []   # (binary_mask, prob_map) per image, both at model resolution
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            tensors = torch.stack(
                [load_spectrogram_tensor(p, DEFAULT_IMAGE_SIZE) for p in batch]
            ).to(device)
            logits = model(tensors)
            probs  = torch.sigmoid(logits.detach())
            for l, p in zip(logits, probs):
                out.append((logits_to_mask(l, threshold=thr),
                            p.squeeze().float().cpu().numpy()))
    return out


def run_squeakout(spec_dir, out_dir, channels=None, checkpoint=None,
                  batch_size=INFERENCE_BATCH_SIZE, mask_threshold=None,
                  overwrite=False,
                  max_mask_area_frac=0.15, min_freq_sweep_frac=0.0, min_mask_cols=9,
                  min_centroid_hz=25000.0, max_flatness=None, recording_dir=None,
                  prefix="headmic"):
    """Run SqueakOut on pre-generated spectrogram PNGs; write coco_ch_{ch}.json per channel.

    Each detection stores a ``score`` = its mean mask-probability, so
    scripts/evaluation/pr_curves.py can sweep a detection threshold post-hoc. Lower
    ``mask_threshold`` (the sigmoid binarization cut) for a permissive run that
    gives the PR sweep a high-recall arm.

    Stage-2 uses the same cross-validated gate as ridge: a component is kept only
    if it spans >= min_mask_cols time columns AND its band-limited spectral centroid
    is >= min_centroid_hz (pass 0 to disable) AND, when max_flatness is set, its
    band-limited flatness is <= max_flatness. The spectral gates need the source
    audio, read from recording_dir (inferred as data/<experiment>/<idx> from spec_dir
    when not given); if the audio is missing they are skipped and only the geometric
    gates apply.

    Resumes by default: channels whose ``coco_ch_{ch}.json`` already exists are
    skipped. Pass ``overwrite=True`` to reprocess them.
    """
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "squeakout"))

    spec_dir = Path(spec_dir)
    out_dir  = Path(out_dir)
    if recording_dir is None:
        recording_dir = recording_dir_from_spec_dir(spec_dir)

    by_ch = group_specs_by_channel(spec_dir, channels, prefix=prefix)
    pending = {ch: entries for ch, entries in by_ch.items()
               if overwrite or not (out_dir / f"coco_ch_{ch}.json").exists()}
    if not pending:
        print("All channels already have output, skipping.")
        return

    from squeakout import load_model, resolve_device

    ckpt = Path(checkpoint) if checkpoint else repo_root / "squeakout" / "squeakout_weights.ckpt"
    device = resolve_device()
    model  = load_model(ckpt, device=device)
    model.eval()

    coco_by_ch = {}

    for ch, entries in pending.items():
        paths   = [e[0] for e in entries]
        results = _run_batch(paths, model, device, batch_size, mask_threshold)  # [(mask, prob)]

        loaded = load_channel_audio(recording_dir, ch, prefix=prefix)
        sr, audio = (loaded[0], loaded[1]) if loaded is not None else (None, None)
        nyquist = (sr / 2.0) if audio is not None else None
        if audio is None:
            print(f"  squeakout ch{ch}: no audio in {recording_dir} -> spectral gates skipped")

        coco = make_coco(f"SqueakOut detections — ch {ch}", "squeakout")
        for (path, t0, t1), (raw_mask, prob) in zip(entries, results):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            H, W = gray.shape
            nyq  = nyquist if nyquist is not None else (sr / 2.0 if sr else 62500.0)
            iid  = len(coco["images"])
            coco["images"].append(
                image_entry(iid, path.name, W, H, window_start_sec=t0, window_end_sec=t1)
            )
            # Score each surviving component by its mean mask-probability (prob and
            # the cleaned mask share the model-resolution grid).
            clean = clean_mask(raw_mask)
            n_lbl, labels, _, _ = cv2.connectedComponentsWithStats(clean)
            for lbl in range(1, n_lbl):
                comp  = (labels == lbl).astype(np.uint8) * 255
                score = float(prob[labels == lbl].mean())
                polys = mask_to_polygons(comp, (H, W))
                # same stage-2 filter ridge and SAM3 apply, judged on the
                # image-resolution mask so all three methods are held to it identically.
                comp_hires = np.zeros((H, W), np.uint8)
                for poly in polys:
                    if len(poly) >= 6:
                        cv2.fillPoly(comp_hires, [np.asarray(poly, np.int32).reshape(-1, 2)], 255)
                # stage-2 spectral gate: map the component bbox back to a time segment
                # + frequency band, then take band-limited features (one STFT).
                centroid, flatness = detection_spectral_features(
                    audio, sr, cv2.boundingRect(comp_hires), t0, t1, H, W, nyq)
                if not passes_mask_filters(comp_hires, H, W, max_mask_area_frac,
                                           min_freq_sweep_frac, min_mask_cols,
                                           centroid_hz=centroid, min_centroid_hz=min_centroid_hz,
                                           flatness=flatness, max_flatness=max_flatness):
                    continue
                for poly in polys:
                    coco["annotations"].append(
                        poly_annotation(len(coco["annotations"]), iid, poly,
                                        extra={"score": score, "centroid_hz": centroid,
                                               "flatness": flatness})
                    )
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)
