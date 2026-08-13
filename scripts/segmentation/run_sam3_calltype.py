"""SAM3 on dryad_gerbil, prompted with call-type-gated ridge candidates instead
of the single loudest one.

vox_tracer/sam3_runner.py's iter_sam3_windows prompts SAM3 with exactly one
ridge candidate per window -- whichever connected component scored highest on
raw ridge brightness (mean_ridge_intensity * (1 - extent)). Evaluating an early
dryad smoke test against ground truth showed this is the main recall bottleneck:
windows with >1 real call often lose
the quieter ones to a single loud broadband noise streak that outscores them.

This script is standalone rather than a change to sam3_runner.py -- it's
dryad-specific and experimental until validated, so it stays out of the path
every other dataset (gerbil_ssl, etc.) runs through. It reuses sam3_runner's
model loader and the shared ridge/coco/spec helpers, but does its own
per-window candidate selection:

  1. Ridge-filter the window as usual (vox_tracer.ridge.compute_seg_mask) to
     get ALL connected components, not just the best one.
  2. For each, re-spectrogram its time span with AVA's own preprocessing (the
     recipe checkpoint_050.tar was trained under) and encode it with that VAE.
  3. Keep it only if its latent is close enough to one of the 70 known
     dryad_gerbil call-type cluster centroids (outputs/dryad_holdout, from
     scripts/vae/dryad_to_ava_specs.py) -- see vox_tracer/call_type_gate.py.
  4. Prompt SAM3 once per surviving candidate (up to --max-candidates),
     reusing one image encoding per window (processor.reset_all_prompts
     between candidates) so cost scales with real calls, not a fixed re-encode
     per candidate.

This gates out noise (a candidate that doesn't look like any known call type
never reaches SAM3) while still letting a busy window recover several distinct
call shapes, not just one -- the two failure modes raised when reviewing the
smoke test.

    python scripts/segmentation/run_sam3_calltype.py \\
        outputs/spectrograms/dryad_gerbil/2020_07_22_15_52_33_369348_merged \\
        outputs/sam3_calltype/dryad_gerbil/2020_07_22_15_52_33_369348_merged \\
        --recording-dir data/dryad_gerbil/2020_07_22_15_52_33_369348_merged \\
        --call-type-specs outputs/dryad_holdout \\
        --call-type-checkpoint /mnt/home/the10/ceph/dataset/dryad_gerbil/checkpoint_050.tar \\
        --sam3-checkpoint sam3/sam3.pt \\
        --channels 0 --prefix mic --freq-min 500 --min-centroid-hz 0
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.filters import sato

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vox_tracer import call_type_gate
from vox_tracer.coco import image_entry, make_coco, mask_to_polygons, poly_annotation, save_coco_per_channel
from vox_tracer.ridge import compute_seg_mask, detection_spectral_features, passes_mask_filters
from vox_tracer.sam3_runner import build_processor
from vox_tracer.spec import group_specs_by_channel, load_channel_audio


def _gray_and_sato(path, sigmas):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    response = sato(gray.astype(np.float64) / 255.0, sigmas=sigmas, black_ridges=False)
    return gray, response


def _all_ridge_candidates(seg_mask, sato_response):
    """[(box, score), ...] for every ridge connected component, highest score first.

    Same score as sam3_runner.pick_best_candidate (mean_ridge_intensity * (1 -
    extent)) but returns every component instead of only the argmax, so the
    call-type gate below has a full set of candidates to choose from.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_mask)
    out = []
    for lbl in range(1, n_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        comp_h = stats[lbl, cv2.CC_STAT_HEIGHT]
        comp_w = max(stats[lbl, cv2.CC_STAT_WIDTH], 1)
        x0     = stats[lbl, cv2.CC_STAT_LEFT]
        y0     = stats[lbl, cv2.CC_STAT_TOP]
        comp_mask = (labels == lbl).astype(np.uint8)
        mean_int  = float(sato_response[comp_mask > 0].mean())
        extent    = area / (comp_h * comp_w)
        out.append(((x0, y0, x0 + comp_w, y0 + comp_h), mean_int * (1.0 - extent)))
    out.sort(key=lambda bs: -bs[1])
    return out


def run_sam3_calltype(
    spec_dir, out_dir, recording_dir,
    call_type_specs_dir, call_type_checkpoint,
    channels=None, prefix="headmic",
    sam3_checkpoint=None, score_threshold=0.5,
    sigmas=(2, 3, 4), threshold_pct=99.0,
    sample_rate=125000, freq_min=20000.0,
    min_area=30, vert_aspect=5.0, horiz_aspect=0.2, close_kernel=(7, 3),
    max_mask_area_frac=0.15, min_freq_sweep_frac=0.0, min_mask_cols=9,
    min_centroid_hz=25000.0, max_flatness=None,
    max_candidates=5, gate_pct=95.0,
    overwrite=False,
):
    spec_dir, out_dir, recording_dir = Path(spec_dir), Path(out_dir), Path(recording_dir)
    sigmas = list(sigmas)

    by_ch = group_specs_by_channel(spec_dir, channels, prefix=prefix)
    pending = {ch: entries for ch, entries in by_ch.items()
               if overwrite or not (out_dir / f"coco_ch_{ch}.json").exists()}
    if not pending:
        print("All channels already have output, skipping.")
        return

    processor = build_processor(sam3_checkpoint, score_threshold)

    print("Loading call-type gate …")
    centroids, cluster_ids, threshold = call_type_gate.load_cluster_centroids(call_type_specs_dir, pct=gate_pct)
    vae = call_type_gate.build_vae(call_type_checkpoint, save_dir=out_dir)
    print(f"  {len(cluster_ids)} clusters, distance threshold={threshold:.3f}")

    coco_by_ch = {}
    for ch, entries in pending.items():
        coco = make_coco(
            "SAM3 detections via call-type-gated ridge candidates", "sam3",
            extra_info={"sigmas": sigmas, "threshold_pct": threshold_pct,
                        "score_threshold": score_threshold,
                        "min_area": min_area, "vert_aspect": vert_aspect,
                        "horiz_aspect": horiz_aspect, "close_kernel": list(close_kernel),
                        "max_mask_area_frac": max_mask_area_frac,
                        "min_freq_sweep_frac": min_freq_sweep_frac,
                        "min_mask_cols": min_mask_cols,
                        "min_centroid_hz": min_centroid_hz,
                        "max_flatness": max_flatness,
                        "call_type_gate_threshold": threshold,
                        "max_candidates": max_candidates},
        )
        n_skip = 0

        loaded = load_channel_audio(recording_dir, ch, prefix=prefix)
        sr, audio = (loaded[0], loaded[1]) if loaded is not None else (None, None)
        nyquist = (sr / 2.0) if audio is not None else (sample_rate / 2.0)
        if audio is None:
            print(f"  ch{ch}: no audio in {recording_dir} -- call-type gate needs it, skipping channel")
            continue

        for path, t0, t1 in entries:
            got = _gray_and_sato(path, sigmas)
            if got is None:
                continue
            gray, response = got
            H, W = gray.shape
            seg_mask = compute_seg_mask(response, H, W, threshold_pct, sample_rate, freq_min,
                                        min_area, vert_aspect, horiz_aspect, close_kernel)

            iid = len(coco["images"])
            coco["images"].append(image_entry(iid, path.name, W, H, window_start_sec=t0, window_end_sec=t1))

            boxes = []
            for box, _score in _all_ridge_candidates(seg_mask, response):
                z = call_type_gate.encode_bbox(vae, audio, sr, box, t0, t1, W)
                if z is None:
                    continue
                _cluster_id, dist = call_type_gate.nearest_cluster(z, centroids, cluster_ids)
                if dist <= threshold:
                    boxes.append((box, dist))
            boxes.sort(key=lambda bd: bd[1])
            boxes = [box for box, _dist in boxes[:max_candidates]]

            if not boxes:
                n_skip += 1
                continue

            pil_rgb, state = None, None
            for box in boxes:
                x0, y0, x1, y1 = box
                norm_box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if state is None:
                        pil_rgb = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
                        state = processor.set_image(pil_rgb)
                    else:
                        processor.reset_all_prompts(state)
                    state = processor.add_geometric_prompt(box=norm_box, label=True, state=state)
                for mask_tensor, score in zip(state.get("masks", []), state.get("scores", [])):
                    mask_u8 = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255
                    score = float(score)
                    centroid, flatness = detection_spectral_features(
                        audio, sr, cv2.boundingRect(mask_u8), t0, t1, H, W, nyquist)
                    if not passes_mask_filters(mask_u8, H, W, max_mask_area_frac,
                                               min_freq_sweep_frac, min_mask_cols,
                                               centroid_hz=centroid, min_centroid_hz=min_centroid_hz,
                                               flatness=flatness, max_flatness=max_flatness):
                        continue
                    for poly in mask_to_polygons(mask_u8, (H, W)):
                        coco["annotations"].append(
                            poly_annotation(len(coco["annotations"]), iid, poly,
                                            extra={"score": score, "centroid_hz": centroid,
                                                   "flatness": flatness}))

        n_win = len(coco["images"])
        print(f"ch {ch}: {n_win - n_skip}/{n_win} windows produced annotations")
        coco_by_ch[ch] = coco

    save_coco_per_channel(coco_by_ch, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--recording-dir", required=True)
    ap.add_argument("--call-type-specs", required=True,
                    help="dir of *.hdf5 from scripts/vae/dryad_to_ava_specs.py (z_70 + latent_means_official)")
    ap.add_argument("--call-type-checkpoint", required=True, help="AVA VAE checkpoint (e.g. checkpoint_050.tar)")
    ap.add_argument("--channels", default="118,35")
    ap.add_argument("--prefix", default="headmic")
    ap.add_argument("--sam3-checkpoint", default=None, help="leave unset to download from HuggingFace")
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--sigmas", default="2,3,4")
    ap.add_argument("--threshold-pct", type=float, default=99.0)
    ap.add_argument("--sample-rate", type=int, default=125000)
    ap.add_argument("--freq-min", type=float, default=20000.0)
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--max-mask-area-frac", type=float, default=0.15)
    ap.add_argument("--min-freq-sweep-frac", type=float, default=0.0)
    ap.add_argument("--min-mask-cols", type=int, default=9)
    ap.add_argument("--min-centroid-hz", type=float, default=25000.0)
    ap.add_argument("--max-flatness", type=float, default=None)
    ap.add_argument("--max-candidates", type=int, default=5,
                    help="cap on gated candidates prompted to SAM3 per window")
    ap.add_argument("--gate-pct", type=float, default=95.0,
                    help="percentile of within-cluster distance used as the call-type gate threshold")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_sam3_calltype(
        args.spec_dir, args.out_dir, args.recording_dir,
        args.call_type_specs, args.call_type_checkpoint,
        channels=[int(c) for c in args.channels.split(",")], prefix=args.prefix,
        sam3_checkpoint=args.sam3_checkpoint, score_threshold=args.score_threshold,
        sigmas=tuple(int(s) for s in args.sigmas.split(",")), threshold_pct=args.threshold_pct,
        sample_rate=args.sample_rate, freq_min=args.freq_min,
        min_area=args.min_area,
        max_mask_area_frac=args.max_mask_area_frac, min_freq_sweep_frac=args.min_freq_sweep_frac,
        min_mask_cols=args.min_mask_cols, min_centroid_hz=args.min_centroid_hz, max_flatness=args.max_flatness,
        max_candidates=args.max_candidates, gate_pct=args.gate_pct,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
