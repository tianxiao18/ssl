"""SAM3 segmentation guided by a sato ridge-filter candidate exemplar."""
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.filters import sato

from vox_tracer.coco import image_entry, make_coco, mask_to_polygons, poly_annotation, save_coco_per_channel
from vox_tracer.ridge import _filter_components, compute_seg_mask
from vox_tracer.spec import group_specs_by_channel


def _eccentricity(comp_mask):
    """Eccentricity of the best-fit ellipse via second-order central moments."""
    M = cv2.moments(comp_mask)
    if M["m00"] < 1:
        return 0.0
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    tr   = mu20 + mu02
    disc = max((tr / 2) ** 2 - (mu20 * mu02 - mu11 ** 2), 0.0)
    lam1 = tr / 2 + disc ** 0.5
    lam2 = tr / 2 - disc ** 0.5
    return (1.0 - lam2 / lam1) ** 0.5 if lam1 > 1e-9 else 0.0


def pick_best_candidate(seg_mask, sato_response):
    """Return ((x0, y0, x1, y1), score) of the best candidate, or (None, 0.0).

    Hard filters: not too tall, not too wide, eccentricity > 0.8.
    Soft score: mean_ridge_intensity * (1 - extent) * (1 - solidity).
    """
    H, W = seg_mask.shape
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_mask)
    best_box   = None
    best_score = 0.0

    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        x0     = stats[lbl, cv2.CC_STAT_LEFT]
        y0     = stats[lbl, cv2.CC_STAT_TOP]

        if comp_h / H >= 0.35:
            continue
        if comp_w / W >= 0.30:
            continue

        comp_mask = (labels == lbl).astype(np.uint8)
        if _eccentricity(comp_mask) <= 0.8:
            continue

        # frequency trace: median y per x column
        ys, xs = np.where(comp_mask > 0)
        unique_xs = np.unique(xs)
        if len(unique_xs) < 5:
            continue
        col_meds = np.array([np.median(ys[xs == x]) for x in unique_xs])

        # reject flat horizontal lines — must sweep at least 4% of image height
        if (col_meds.max() - col_meds.min()) / H < 0.04:
            continue

        # reject jagged / right-angle shapes — second differences must be small
        if np.std(np.diff(col_meds, 2)) > 3.0:
            continue

        mean_int = float(sato_response[comp_mask > 0].mean())
        extent   = area / (comp_h * comp_w)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull      = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 1.0

        score = mean_int * (1.0 - extent) * (1.0 - solidity)
        if score > best_score:
            best_score = score
            best_box   = (x0, y0, x0 + comp_w, y0 + comp_h)

    return best_box, best_score


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
    overwrite=False,
):
    """Run SAM3 on pre-generated spectrogram PNGs; write coco_ch_{ch}.json per channel."""
    from pathlib import Path as _Path

    by_ch = group_specs_by_channel(spec_dir, channels)
    coco_by_ch = {}

    pending = {ch: entries for ch, entries in by_ch.items()
               if overwrite or not (_Path(out_dir) / f"coco_ch_{ch}.json").exists()}
    if not pending:
        print("All channels already have output, skipping.")
        return

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # pkg_resources can't locate the asset when sam3.__file__ is None (editable install quirk)
    _bpe = str(_Path(__file__).resolve().parents[1] / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    sigmas = list(sigmas)

    if checkpoint:
        print(f"Loading SAM3 from {checkpoint} …")
        sam3_model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, bpe_path=_bpe)
    else:
        print("Downloading SAM3 from HuggingFace …")
        sam3_model = build_sam3_image_model(load_from_HF=True, bpe_path=_bpe)
    processor = Sam3Processor(sam3_model, confidence_threshold=score_threshold)
    print("SAM3 ready.")

    for ch, entries in pending.items():

        coco = make_coco(
            "SAM3 detections via sato ridge-filter exemplar", "sam3",
            extra_info={"sigmas": sigmas, "threshold_pct": threshold_pct,
                        "score_threshold": score_threshold},
        )
        n_skip = 0

        for path, t0, t1 in entries:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            H, W  = gray.shape
            img_f = gray.astype(np.float64) / 255.0
            iid   = len(coco["images"])
            coco["images"].append(
                image_entry(iid, path.name, W, H, window_start_sec=t0, window_end_sec=t1)
            )

            response = sato(img_f, sigmas=sigmas, black_ridges=False)
            seg_mask = compute_seg_mask(response, H, W, threshold_pct, sample_rate, freq_min)
            best_box, _ = pick_best_candidate(seg_mask, response)

            if best_box is None:
                n_skip += 1
                continue

            x0, y0, x1, y1 = best_box
            norm_box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
            pil_rgb  = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(pil_rgb)
                state = processor.add_geometric_prompt(box=norm_box, label=True, state=state)

            for mask_tensor, score in zip(state.get("masks", []), state.get("scores", [])):
                mask_u8 = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                for poly in mask_to_polygons(mask_u8, (H, W)):
                    coco["annotations"].append(
                        poly_annotation(len(coco["annotations"]), iid, poly, extra={"score": float(score)})
                    )

        n_win = len(coco["images"])
        print(f"ch {ch}: {n_win - n_skip}/{n_win} windows produced annotations")
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)
