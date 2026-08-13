"""Apply the shared stage-2 passes_mask_filters to an EXISTING coco output tree,
post-hoc, without re-running the model.

Each annotation's polygon is rasterized to its image (H, W) grid and passed
through vox_tracer.ridge.passes_mask_filters — the same function ridge, sam3 and
squeakout apply at emit time — so a set of already-computed detections can be
filtered to match a fair comparison without a GPU re-run.

    python scripts/segmentation/apply_mask_filter.py outputs/squeakout outputs/squeakout_filtered \
        --channels 118,35

Only the mask geometry matters, so this reproduces exactly what the runner would
have emitted whenever each component maps to a single polygon (the usual case).

--max-flatness / --min-centroid-hz add the spectral flatness / centroid gates on
top, still without a GPU re-run: they re-derive each detection's band-limited
flatness/centroid from the bbox already stored in the annotation (same math as
sam3_runner/ridge's detection_spectral_features) plus the ORIGINAL audio, and
drop anything too noise-like (flat) or too low-pitched to be a real call. This is
the fix for the SAM3 dryad runs, which never had a flatness gate wired in (see
vox_tracer.ridge.detection_band_features):

    python scripts/segmentation/apply_mask_filter.py outputs/sam3/dryad_gerbil outputs/sam3_flatness_filtered/dryad_gerbil \
        --channels 0 --flat-layout --recording-dir-base data/dryad_gerbil --prefix mic \
        --min-mask-cols 0 --min-freq-sweep-frac 0 --max-flatness 0.21
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_tracer.ridge import detection_spectral_features, passes_mask_filters
from vox_tracer.spec import load_channel_audio


def _poly_mask(seg, H, W):
    m = np.zeros((H, W), np.uint8)
    for poly in seg:
        if len(poly) >= 6:
            cv2.fillPoly(m, [np.asarray(poly, np.int32).reshape(-1, 2)], 255)
    return m


def filter_coco(coco, max_area_frac, min_sweep_frac, min_cols,
                max_flatness=None, min_centroid_hz=None, audio=None, sr=None, nyquist=None):
    """Return (new_coco, n_kept, n_dropped); images are preserved, annotations filtered.

    max_flatness/min_centroid_hz on one side and audio/sr/nyquist on the other are
    all-or-nothing: pass audio/sr/nyquist plus either (or both) spectral gate to
    also gate on band-limited spectral flatness/centroid (re-derived from each
    annotation's stored bbox + its image's window_start_sec/window_end_sec), or
    leave audio at None to keep the original geometry-only behavior.
    """
    img_by_id = {im["id"]: im for im in coco["images"]}
    kept = []
    for ann in coco["annotations"]:
        im = img_by_id[ann["image_id"]]
        H, W = im["height"], im["width"]
        mask = _poly_mask(ann.get("segmentation", []), H, W)
        centroid = flatness = None
        if audio is not None and (max_flatness is not None or min_centroid_hz is not None):
            centroid, flatness = detection_spectral_features(
                audio, sr, ann["bbox"], im["window_start_sec"], im["window_end_sec"], H, W, nyquist)
        if passes_mask_filters(mask, H, W, max_area_frac, min_sweep_frac, min_cols,
                               centroid_hz=(centroid if min_centroid_hz is not None else None),
                               min_centroid_hz=min_centroid_hz,
                               flatness=(flatness if max_flatness is not None else None),
                               max_flatness=max_flatness):
            kept.append(ann)
    dropped = len(coco["annotations"]) - len(kept)
    for new_id, ann in enumerate(kept):
        ann["id"] = new_id
    return {**coco, "annotations": kept}, len(kept), dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_base")
    ap.add_argument("out_base")
    ap.add_argument("--channels", default="118,35")
    ap.add_argument("--max-mask-area-frac",  type=float, default=0.15)
    ap.add_argument("--min-freq-sweep-frac", type=float, default=0.04)
    ap.add_argument("--min-mask-cols",       type=int,   default=5)
    ap.add_argument("--max-flatness", type=float, default=None,
                    help="also gate on band-limited spectral flatness (e.g. 0.21); "
                         "requires --recording-dir-base for the source audio")
    ap.add_argument("--min-centroid-hz", type=float, default=None,
                    help="also gate on band-limited spectral centroid (e.g. 25000); "
                         "requires --recording-dir-base for the source audio")
    ap.add_argument("--recording-dir-base", default=None,
                    help="root containing per-recording audio, e.g. data/dryad_gerbil "
                         "(with --flat-layout) or data/gerbil_ssl (without)")
    ap.add_argument("--prefix", default="headmic")
    ap.add_argument("--sample-rate", type=int, default=125000,
                    help="nyquist fallback if a recording's audio can't be found")
    ap.add_argument("--flat-layout", action="store_true",
                    help="in_base is <recording>/coco_ch_*.json (dryad_gerbil) instead of "
                         "the default experiment_*/idx_*/coco_ch_*.json (gerbil_ssl)")
    args = ap.parse_args()

    channels = [int(c) for c in args.channels.split(",")]
    in_base, out_base = Path(args.in_base), Path(args.out_base)

    glob_pat = "*/coco_ch_*.json" if args.flat_layout else "experiment_*/idx_*/coco_ch_*.json"
    files = sorted(in_base.glob(glob_pat))
    files = [f for f in files if int(f.stem.split("_")[-1]) in channels]
    print(f"{len(files)} coco files under {in_base}")

    needs_audio = args.max_flatness is not None or args.min_centroid_hz is not None
    if needs_audio and not args.recording_dir_base:
        raise SystemExit("--max-flatness/--min-centroid-hz need --recording-dir-base (source audio)")

    def _audio_for(rec_dir, ch):
        # No cache: each (rec_dir, ch) is visited exactly once below, so caching would
        # only accumulate every recording's raw audio in memory for no benefit.
        loaded = load_channel_audio(rec_dir, ch, prefix=args.prefix)
        return (loaded[0], loaded[1]) if loaded is not None else (None, None)

    tot_kept = tot_drop = 0
    for f in files:
        rel = f.relative_to(in_base)
        ch = int(f.stem.split("_")[-1])
        audio = sr = nyquist = None
        if needs_audio:
            rec_dir = (Path(args.recording_dir_base) / rel.parts[0] if args.flat_layout
                      else Path(args.recording_dir_base) / rel.parts[0] / rel.parts[1])
            sr, audio = _audio_for(rec_dir, ch)
            nyquist = (sr / 2.0) if audio is not None else (args.sample_rate / 2.0)
            if audio is None:
                print(f"  {rel}: no audio in {rec_dir} -> spectral gates skipped for this file")

        with open(f) as fh:
            coco = json.load(fh)
        new_coco, kept, dropped = filter_coco(
            coco, args.max_mask_area_frac, args.min_freq_sweep_frac, args.min_mask_cols,
            max_flatness=(args.max_flatness if audio is not None else None),
            min_centroid_hz=(args.min_centroid_hz if audio is not None else None),
            audio=audio, sr=sr, nyquist=nyquist)
        tot_kept += kept
        tot_drop += dropped
        out_path = out_base / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(new_coco, fh)

    total = tot_kept + tot_drop
    print(f"kept {tot_kept}/{total} annotations, dropped {tot_drop} "
          f"({tot_drop / total:.1%})  →  {out_base}")


if __name__ == "__main__":
    main()
