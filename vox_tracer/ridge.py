"""Ridge filter segmentation shared by run_ridge.py and sam3_runner.py."""
import cv2
import numpy as np
from skimage.filters import frangi, hessian, meijering, sato

from vox_tracer.coco import image_entry, make_coco, poly_annotation, save_coco_per_channel
from vox_tracer.spec import group_specs_by_channel

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
):
    """Run a single ridge filter over all PNGs in spec_dir; write coco_ch_{ch}.json per channel."""
    sigmas = list(sigmas)
    fn = _build_filter_fn(filter_name, sigmas)

    by_ch = group_specs_by_channel(spec_dir, channels)
    coco_by_ch = {}

    for ch, entries in by_ch.items():
        coco = make_coco(f"ridge({filter_name}) detections", "vox")
        for path, t0, t1 in entries:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            H, W    = gray.shape
            img_f    = gray.astype(np.float64) / 255.0
            response = fn(img_f)
            mask     = compute_seg_mask(response, H, W, threshold_pct, sample_rate, freq_min)

            iid = len(coco["images"])
            coco["images"].append(
                image_entry(iid, path.name, W, H, window_start_sec=t0, window_end_sec=t1)
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                poly = cnt.flatten().tolist()
                ann  = poly_annotation(len(coco["annotations"]), iid, poly)
                ann["area"] = float(cv2.contourArea(cnt))
                coco["annotations"].append(ann)
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)
