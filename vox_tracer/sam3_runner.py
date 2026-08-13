"""SAM3 segmentation guided by a sato ridge-filter candidate exemplar."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.filters import sato

from vox_tracer.coco import image_entry, make_coco, mask_to_polygons, poly_annotation, save_coco_per_channel
from vox_tracer.ridge import compute_seg_mask, detection_spectral_features, passes_mask_filters
from vox_tracer.spec import group_specs_by_channel, load_channel_audio


_SATO_CACHE = {}


def _gray_and_sato(path, sigmas, cache=True):
    """Return (gray uint8, sato response float64) for path.

    Returns None if the image can't be read. The response is treated as read-only
    by callers (compute_seg_mask / pick_best_candidate only read it).

    Caching (by (path, sigmas)) exists solely for the hyperparameter sweep
    (sweep_core.gpu_pass), which re-runs over the SAME PNGs every trial while
    sigmas stays fixed, so the sato response is computed once and reused across
    trials. The single-pass run_sam3 production path visits each PNG exactly once,
    so it passes cache=False: caching there never hits and would otherwise
    accumulate ~1 MB/window across all recordings until the process OOMs.
    """
    key = (str(path), tuple(sigmas))
    if cache:
        hit = _SATO_CACHE.get(key)
        if hit is not None:
            return hit
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    img_f = gray.astype(np.float64) / 255.0
    response = sato(img_f, sigmas=list(sigmas), black_ridges=False)
    if cache:
        _SATO_CACHE[key] = (gray, response)
    return gray, response


def pick_best_candidate(seg_mask, sato_response):
    """Return ((x0, y0, x1, y1), score) of the best candidate, or (None, 0.0).

    Score: mean_ridge_intensity * (1 - extent).
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_mask)
    best_box   = None
    best_score = 0.0

    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        x0     = stats[lbl, cv2.CC_STAT_LEFT]
        y0     = stats[lbl, cv2.CC_STAT_TOP]

        comp_mask = (labels == lbl).astype(np.uint8)
        mean_int  = float(sato_response[comp_mask > 0].mean())
        extent    = area / (comp_h * comp_w)

        score = mean_int * (1.0 - extent)
        if score > best_score:
            best_score = score
            best_box   = (x0, y0, x0 + comp_w, y0 + comp_h)

    return best_box, best_score


def build_processor(checkpoint=None, score_threshold=0.5):
    """Load the SAM3 image model once and wrap it in a Sam3Processor.

    Loading the model is the slow part; callers that sweep many configs should
    build the processor once and reuse it across runs.
    """
    from pathlib import Path as _Path
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # pkg_resources can't locate the asset when sam3.__file__ is None (editable install quirk)
    _bpe = str(_Path(__file__).resolve().parents[1] / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    if checkpoint:
        print(f"Loading SAM3 from {checkpoint} …")
        sam3_model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, bpe_path=_bpe)
    else:
        print("Downloading SAM3 from HuggingFace …")
        sam3_model = build_sam3_image_model(load_from_HF=True, bpe_path=_bpe)
    processor = Sam3Processor(sam3_model, confidence_threshold=score_threshold)
    print("SAM3 ready.")
    return processor


def iter_sam3_windows(
    processor,
    entries,
    sigmas=(2, 3, 4),
    threshold_pct=99.0,
    sample_rate=125000,
    freq_min=20000.0,
    min_area=30,
    vert_aspect=5.0,
    horiz_aspect=0.2,
    close_kernel=(7, 3),
    cache_sato=True,
):
    """Yield per-window ridge + raw SAM3 results, *without* post-hoc filtering.

    Each yield is a dict with keys: fname, H, W, window_start, window_end,
    seg_mask (ridge segmentation), best_box (SAM3 prompt or None), and
    raw_masks (list of (mask_u8, score); empty when best_box is None).

    This is the expensive GPU stage. Post-hoc filters (vox_tracer.ridge.passes_mask_filters)
    are applied by the caller, so a single pass can be re-scored cheaply under
    many stage-2 settings.

    cache_sato defaults True for the sweep (re-runs over the same PNGs across
    trials); run_sam3 passes False for its single pass over each PNG so the sato
    cache doesn't accumulate across all recordings (see _gray_and_sato).
    """
    sigmas = list(sigmas)
    for path, t0, t1 in entries:
        got = _gray_and_sato(path, sigmas, cache=cache_sato)
        if got is None:
            continue
        gray, response = got
        H, W  = gray.shape

        seg_mask = compute_seg_mask(response, H, W, threshold_pct, sample_rate, freq_min,
                                    min_area, vert_aspect, horiz_aspect, close_kernel)
        best_box, _ = pick_best_candidate(seg_mask, response)

        raw_masks = []
        if best_box is not None:
            x0, y0, x1, y1 = best_box
            norm_box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
            pil_rgb  = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_rgb)
                state = processor.add_geometric_prompt(box=norm_box, label=True, state=state)
            for mask_tensor, score in zip(state.get("masks", []), state.get("scores", [])):
                mask_u8 = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                raw_masks.append((mask_u8, float(score)))

        yield {"fname": path.name, "H": H, "W": W,
               "window_start": t0, "window_end": t1,
               "seg_mask": seg_mask, "best_box": best_box, "raw_masks": raw_masks}


def run_sam3(
    spec_dir,
    out_dir,
    channels=None,
    checkpoint=None,
    sigmas=(2, 3, 4),
    threshold_pct=99.0,
    score_threshold=0.5,
    sample_rate=125000,
    freq_min=20000.0,
    min_area=30,
    vert_aspect=5.0,
    horiz_aspect=0.2,
    close_kernel=(7, 3),
    max_mask_area_frac=0.15,
    min_freq_sweep_frac=0.0,
    min_mask_cols=9,
    min_centroid_hz=25000.0,
    max_flatness=None,
    recording_dir=None,
    overwrite=False,
    processor=None,
    prefix="headmic",
):
    """Run SAM3 on pre-generated spectrogram PNGs; write coco_ch_{ch}.json per channel.

    Pass a pre-built `processor` (from build_processor) to skip model loading.

    Stage-2 uses the same cross-validated gate as ridge: a detection is kept only
    if it spans >= min_mask_cols time columns AND its band-limited spectral centroid
    is >= min_centroid_hz (pass 0 to disable) AND, when max_flatness is set, its
    band-limited flatness is <= max_flatness. The spectral gates need the source
    audio, read from recording_dir (inferred as data/<experiment>/<idx> from spec_dir
    when not given); if the audio is missing they are skipped and only the geometric
    gates apply.
    """
    spec_dir = Path(spec_dir)
    if recording_dir is None:
        recording_dir = Path("data") / "gerbil_ssl" / spec_dir.parent.name / spec_dir.name

    by_ch = group_specs_by_channel(spec_dir, channels, prefix=prefix)
    coco_by_ch = {}

    pending = {ch: entries for ch, entries in by_ch.items()
               if overwrite or not (Path(out_dir) / f"coco_ch_{ch}.json").exists()}
    if not pending:
        print("All channels already have output, skipping.")
        return

    if processor is None:
        processor = build_processor(checkpoint, score_threshold)

    stage1 = dict(sigmas=sigmas, threshold_pct=threshold_pct, sample_rate=sample_rate,
                  freq_min=freq_min, min_area=min_area, vert_aspect=vert_aspect,
                  horiz_aspect=horiz_aspect, close_kernel=close_kernel)

    for ch, entries in pending.items():

        coco = make_coco(
            "SAM3 detections via sato ridge-filter exemplar", "sam3",
            extra_info={"sigmas": list(sigmas), "threshold_pct": threshold_pct,
                        "score_threshold": score_threshold,
                        "min_area": min_area, "vert_aspect": vert_aspect,
                        "horiz_aspect": horiz_aspect, "close_kernel": list(close_kernel),
                        "max_mask_area_frac": max_mask_area_frac,
                        "min_freq_sweep_frac": min_freq_sweep_frac,
                        "min_mask_cols": min_mask_cols,
                        "min_centroid_hz": min_centroid_hz,
                        "max_flatness": max_flatness},
        )
        n_skip = 0

        loaded = load_channel_audio(recording_dir, ch, prefix=prefix)
        sr, audio = (loaded[0], loaded[1]) if loaded is not None else (None, None)
        nyquist = (sr / 2.0) if audio is not None else (sample_rate / 2.0)
        if audio is None:
            print(f"  sam3 ch{ch}: no audio in {recording_dir} -> spectral gates skipped")

        for win in iter_sam3_windows(processor, entries, cache_sato=False, **stage1):
            iid = len(coco["images"])
            coco["images"].append(
                image_entry(iid, win["fname"], win["W"], win["H"],
                            window_start_sec=win["window_start"], window_end_sec=win["window_end"])
            )
            if win["best_box"] is None:
                n_skip += 1
                continue

            for mask_u8, score in win["raw_masks"]:
                # stage-2 spectral gate: map the mask bbox back to a time segment +
                # frequency band, then take band-limited features (one STFT).
                centroid, flatness = detection_spectral_features(
                    audio, sr, cv2.boundingRect(mask_u8), win["window_start"],
                    win["window_end"], win["H"], win["W"], nyquist)
                if not passes_mask_filters(mask_u8, win["H"], win["W"], max_mask_area_frac,
                                           min_freq_sweep_frac, min_mask_cols,
                                           centroid_hz=centroid, min_centroid_hz=min_centroid_hz,
                                           flatness=flatness, max_flatness=max_flatness):
                    continue
                for poly in mask_to_polygons(mask_u8, (win["H"], win["W"])):
                    coco["annotations"].append(
                        poly_annotation(len(coco["annotations"]), iid, poly,
                                        extra={"score": score, "centroid_hz": centroid,
                                               "flatness": flatness})
                    )

        n_win = len(coco["images"])
        print(f"ch {ch}: {n_win - n_skip}/{n_win} windows produced annotations")
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)
