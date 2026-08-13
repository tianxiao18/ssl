"""Ridge filter segmentation shared by run_ridge.py and sam3_runner.py."""
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi, hessian, meijering, sato

from vox_tracer.coco import image_entry, make_coco, poly_annotation, save_coco_per_channel
from vox_tracer.paths import recording_dir_from_spec_dir
from vox_tracer.spec import group_specs_by_channel, load_channel_audio

FILTER_NAMES = ("sato", "meijering", "frangi", "hessian")


def _build_filter_fn(name, sigmas):
    if name == "sato":
        return lambda img: sato(img, sigmas=sigmas, black_ridges=False)
    if name == "meijering":
        return lambda img: meijering(img, sigmas=sigmas, black_ridges=False)
    if name == "frangi":
        return lambda img: frangi(img, sigmas=sigmas, black_ridges=False)
    if name == "hessian":
        return lambda img: (1.0 - hessian(img, sigmas=sigmas, black_ridges=False)) * img
    raise ValueError(f"unknown filter: {name!r}. Choose from {FILTER_NAMES}")


def _filter_components(binary, min_area, vert_aspect, horiz_aspect, freq_cutoff_row):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    out = np.zeros_like(binary)
    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        aspect = comp_h / comp_w
        y_top  = stats[lbl, cv2.CC_STAT_TOP]
        if area < min_area:
            continue
        if aspect > vert_aspect:
            continue
        if aspect < horiz_aspect:
            continue
        if y_top > freq_cutoff_row:
            continue
        out[labels == lbl] = 255
    return out


def detection_band_features(audio, sr, t0, t1, f_low, f_high, n_fft=512):
    """Band-limited spectral features of one detection's audio segment.

    Mirrors scripts/evaluation/spectral_features.event_features: the STFT is sliced to the
    detection's [f_low, f_high] band before each feature, so they describe the
    energy WITHIN the detected call box. One STFT serves all features. Returns
    ``(centroid_hz, flatness)`` — flatness near 0 = tonal call, near 1 =
    noise-like — or ``(None, None)`` for an empty segment.
    """
    import librosa  # heavy; imported lazily so image-only callers never pay for it
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(audio), int(round(t1 * sr)))
    if i1 <= i0:
        return None, None
    seg = audio[i0:i1].astype(np.float32)
    S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=n_fft // 4)) + 1e-10
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= f_low) & (freqs <= f_high)
    if not band.any():  # band narrower than one bin: keep the nearest bin
        band[np.argmin(np.abs(freqs - 0.5 * (f_low + f_high)))] = True
    S_b = S[band]
    cent = librosa.feature.spectral_centroid(S=S_b, freq=freqs[band])
    flat = librosa.feature.spectral_flatness(S=S_b)
    return float(np.mean(cent)), float(np.mean(flat))


def detection_spectral_features(audio, sr, bbox, t0, t1, H, W, nyquist):
    """Band-limited ``(centroid_hz, flatness)`` for one detection bbox.

    Maps a detection bbox ``(x, y, w, h)`` in a spectrogram window back to a
    time segment + frequency band (the flipud pixel->physical mapping shared with
    scripts/evaluation/spectral_features.py), then defers to ``detection_band_features``.
    Returns ``(None, None)`` when ``audio`` is None so image-only callers skip the
    spectral gates. Shared by ridge/sam3/squeakout so all three compute the
    stage-2 spectral gate identically.
    """
    if audio is None:
        return None, None
    bx, by, bw, bh = bbox
    dt0 = t0 + (bx / W) * (t1 - t0)
    dt1 = t0 + ((bx + bw) / W) * (t1 - t0)
    f_hi = (H - 1 - by) / (H - 1) * nyquist
    f_lo = (H - 1 - (by + bh)) / (H - 1) * nyquist
    return detection_band_features(audio, sr, dt0, dt1, f_lo, f_hi)


def passes_mask_filters(mask_u8, H, W, max_area_frac=0.15, min_sweep_frac=0.04, min_cols=5,
                        centroid_hz=None, min_centroid_hz=25000.0,
                        flatness=None, max_flatness=None, return_reason=False):
    """Stage-2 per-detection filter shared by ridge and SAM3.

    Rejects a detection mask that is too large (SAM3 bled into background / ridge
    flooded), spans too few time columns (``min_cols``, too short to be a call),
    whose per-column median frequency barely moves (``min_sweep_frac``; pass 0 to
    disable), or that fails an enabled spectral gate:

    * centroid: reject when ``centroid_hz`` < ``min_centroid_hz`` (USVs are
      ultrasonic, so a low-centroid blob is noise);
    * flatness: reject when ``flatness`` > ``max_flatness`` (calls are tonal, so
      a spectrally flat blob is noise). Disabled unless ``max_flatness`` is set.

    Spectral values default to None so image-only callers (SAM3, SqueakOut) keep
    the geometric-only behaviour unchanged.

    Both spectral gates come from the from-scratch, session-CV filter search on
    ridge detections (scripts/archive/ridge_stage2_fromscratch.py): the pair
    ``n_cols>=9 & centroid>=25 kHz`` gives event-level F1 ~0.89, and the single
    gate ``flatness<=0.21`` on raw detections gives ~0.82 (vs ~0.74 for the old
    area/column/sweep gates).

    return_reason : bool, optional
        When True, return ``(passed, reason)`` instead of just ``passed``, where
        ``reason`` is one of ``"too_big"``, ``"too_short"``, ``"flat"``,
        ``"low_centroid"``, ``"high_flatness"``, or ``"keep"`` — which gate first
        rejected the detection (gates are checked in the order below, so only the
        first failing one is reported even if several would reject it).
    """
    def result(passed, reason):
        return (passed, reason) if return_reason else passed

    if int((mask_u8 > 0).sum()) > max_area_frac * H * W:
        return result(False, "too_big")
    ys, xs = np.where(mask_u8 > 0)
    unique_xs = np.unique(xs)
    if len(unique_xs) < min_cols:
        return result(False, "too_short")
    if min_sweep_frac > 0:
        col_meds = np.array([np.median(ys[xs == x]) for x in unique_xs])
        if (col_meds.max() - col_meds.min()) / H < min_sweep_frac:
            return result(False, "flat")
    if centroid_hz is not None and centroid_hz < min_centroid_hz:
        return result(False, "low_centroid")
    if max_flatness is not None and flatness is not None and flatness > max_flatness:
        return result(False, "high_flatness")
    return result(True, "keep")


def compute_seg_mask(
    response,
    H,
    W,
    threshold_pct=99.0,
    sample_rate=125000,
    freq_min=20000.0,
    min_area=30,
    vert_aspect=5.0,
    horiz_aspect=0.2,
    close_kernel=(7, 3),
):
    """Threshold a filter response into a clean binary segmentation mask.

    Pipeline: threshold → remove lines/speckles → morphological close → remove again.
    Lines are removed before closing so the close cannot bridge noise into vocalizations.
    """
    nyquist = sample_rate / 2.0
    freq_cutoff_row = int((H - 1) * (1.0 - freq_min / nyquist))

    thresh   = np.percentile(response, threshold_pct)
    binary   = (response >= thresh).astype(np.uint8) * 255
    filtered = _filter_components(binary, min_area, vert_aspect, horiz_aspect, freq_cutoff_row)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(close_kernel))
    closed   = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    return _filter_components(closed, min_area, vert_aspect, horiz_aspect, freq_cutoff_row)


def run_ridge(
    spec_dir,
    out_dir,
    channels=None,
    filter_name="sato",
    sigmas=(2, 3, 4),
    threshold_pct=99.0,
    sample_rate=125000,
    freq_min=20000.0,
    min_area=50,
    max_mask_area_frac=0.15,
    min_freq_sweep_frac=0.0,
    min_mask_cols=9,
    min_centroid_hz=25000.0,
    max_flatness=None,
    recording_dir=None,
    reject_out_dir=None,
    prefix="headmic",
):
    """Run a single ridge filter over all PNGs in spec_dir; write coco_ch_{ch}.json per channel.

    Stage-1 (``compute_seg_mask``) forms candidate masks; stage-2
    (``passes_mask_filters``) then keeps a detection only if it spans >= min_mask_cols
    time columns AND passes the enabled spectral gates: band-limited centroid >=
    min_centroid_hz (pass 0 to disable) and, when ``max_flatness`` is set, band-limited
    spectral flatness <= max_flatness. The spectral features need the source audio, so
    the wavs are read from ``recording_dir`` (inferred as ``data/<experiment>/<idx>``
    from spec_dir when not given). If the audio is missing the spectral gates are
    skipped and only the geometric gates apply.

    Defaults (min_area=50, min_mask_cols=9, sweep disabled, centroid>=25 kHz) are the
    cross-validated best from scripts/archive/ridge_{stage1,stage2}_fromscratch.py. The
    flatness-only alternative from the same search (F1 ~0.82) is min_mask_cols=0,
    min_centroid_hz=0, max_flatness=0.21.

    reject_out_dir : str or Path, optional
        When given, every stage-1 candidate that stage-2 *rejects* is also written
        here (same coco_ch_{ch}.json format, same image ids as ``out_dir`` so the
        two trees line up), each annotation tagged with ``extra.reject_reason``
        (see ``passes_mask_filters``). Diagnostic only — never fed to scoring, so
        the normal ``out_dir`` output (and every existing caller) is unaffected.
    """
    sigmas = list(sigmas)
    fn = _build_filter_fn(filter_name, sigmas)

    spec_dir = Path(spec_dir)
    if recording_dir is None:
        recording_dir = recording_dir_from_spec_dir(spec_dir)
    by_ch = group_specs_by_channel(spec_dir, channels, prefix=prefix)
    coco_by_ch = {}
    reject_coco_by_ch = {}

    for ch, entries in by_ch.items():
        coco = make_coco(f"ridge({filter_name}) detections", "vox")
        reject_coco = make_coco(f"ridge({filter_name}) stage-2 rejects", "vox")
        loaded = load_channel_audio(recording_dir, ch, prefix=prefix)
        sr, audio = (loaded[0], loaded[1]) if loaded is not None else (None, None)
        nyquist = (sr / 2.0) if audio is not None else (sample_rate / 2.0)
        if audio is None:
            print(f"  ridge ch{ch}: no audio in {recording_dir} -> spectral gates skipped")
        for path, t0, t1 in entries:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            H, W    = gray.shape
            img_f    = gray.astype(np.float64) / 255.0
            response = fn(img_f)
            mask     = compute_seg_mask(response, H, W, threshold_pct, sample_rate,
                                        freq_min, min_area=min_area)

            iid = len(coco["images"])
            img_kw = dict(window_start_sec=t0, window_end_sec=t1)
            coco["images"].append(image_entry(iid, path.name, W, H, **img_kw))
            if reject_out_dir is not None:
                reject_coco["images"].append(image_entry(iid, path.name, W, H, **img_kw))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                poly = cnt.flatten().tolist()
                comp = np.zeros((H, W), np.uint8)
                cv2.drawContours(comp, [cnt], -1, 1, thickness=cv2.FILLED)

                # stage-2 spectral gates: map the detection bbox back to a time
                # segment + frequency band, then take band-limited features (one STFT).
                centroid, flatness = detection_spectral_features(
                    audio, sr, cv2.boundingRect(cnt), t0, t1, H, W, nyquist)

                passed, reason = passes_mask_filters(
                    comp, H, W, max_mask_area_frac, min_freq_sweep_frac, min_mask_cols,
                    centroid_hz=centroid, min_centroid_hz=min_centroid_hz,
                    flatness=flatness, max_flatness=max_flatness, return_reason=True)
                # confidence = mean ridge-filter response inside the component, so
                # scripts/evaluation/pr_curves.py can sweep a detection threshold post-hoc.
                score = float(response[comp.astype(bool)].mean()) if comp.any() else 0.0
                if not passed:
                    if reject_out_dir is not None:
                        ann = poly_annotation(len(reject_coco["annotations"]), iid, poly,
                                              extra={"score": score, "centroid_hz": centroid,
                                                     "flatness": flatness, "reject_reason": reason})
                        ann["area"] = float(cv2.contourArea(cnt))
                        reject_coco["annotations"].append(ann)
                    continue
                ann  = poly_annotation(len(coco["annotations"]), iid, poly,
                                       extra={"score": score, "centroid_hz": centroid,
                                              "flatness": flatness})
                ann["area"] = float(cv2.contourArea(cnt))
                coco["annotations"].append(ann)
        coco_by_ch[ch] = coco
        reject_coco_by_ch[ch] = reject_coco

    save_coco_per_channel(coco_by_ch, out_dir)
    if reject_out_dir is not None:
        save_coco_per_channel(reject_coco_by_ch, reject_out_dir)
