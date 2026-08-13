"""Convert SAM3 COCO segmentation masks into AVA-format spectrogram HDF5 datasets.

For every SAM3 annotation we crop the *same* patch out of the spectrogram image
SAM3 ran on and emit it into two paired datasets that differ ONLY by the mask:

    raw/     crop as-is (background retained)                 = "before denoising"
    masked/  same crop, pixels outside the SAM3 polygon zeroed = "after denoising"

Because both conditions come from one resized raw patch (the mask is applied
*after* the identical crop+resize), a VAE trained on each isolates the effect of
mask-denoising and nothing else. The output HDF5 files are drop-in for
``ava.models.vae_dataset`` (a single ``specs`` dataset of shape (N, 128, 128),
float32 in [0, 1]), so training is stock AVA.

The images are the cached PNGs written by ``vox_tracer/spec.py`` (dB spectrogram
already normalised to [0, 1] and stored as uint8), so dividing by 255 reproduces
the exact fixed global scale SAM3 saw -- no per-image renormalisation, which
would otherwise leak into the A/B contrast.

Example
-------
    python scripts/vae/coco_to_ava_specs.py \
        --coco 'outputs/sam3_best/experiment_384/idx_000/coco_ch_*.json' \
        --out  outputs/ava/experiment_384_idx000
"""
import argparse
import glob as globmod
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

X_SHAPE = (128, 128)  # AVA VAE input size (ava.models.vae.X_SHAPE)


def resolve_spec_dir(coco_path: Path, override: Path | None) -> Path:
    """Directory holding the source PNGs for a coco file.

    Follows the repo convention outputs/<tool>/<exp>/<idx> -> outputs/spectrograms/gerbil_ssl/<exp>/<idx>.
    """
    if override is not None:
        return override
    parts = list(coco_path.parent.parts)
    if "sam3_best" in parts:
        i = parts.index("sam3_best")
        parts[i:i + 1] = ["spectrograms", "gerbil_ssl"]
        return Path(*parts)
    # Fallback: assume PNGs live next to the json.
    return coco_path.parent


def recording_id(coco_path: Path) -> str:
    """Stable 'experiment_N/idx_M' id for a coco file, or the parent dir name."""
    parts = coco_path.parent.parts
    for i, p in enumerate(parts):
        if p.startswith("experiment_") and i + 1 < len(parts):
            return f"{p}/{parts[i + 1]}"
    return coco_path.parent.name


def poly_mask(segmentation, bbox, h, w):
    """Binary uint8 mask (h, w) filled inside the SAM3 polygon(s), or the bbox if absent."""
    mask = np.zeros((h, w), np.uint8)
    polys = [np.asarray(p, np.float64).reshape(-1, 2) for p in (segmentation or []) if len(p) >= 6]
    if polys:
        cv2.fillPoly(mask, [p.round().astype(np.int32) for p in polys], 1)
    else:  # bbox-only annotation: fall back to the rectangle
        x, y, bw, bh = bbox
        cv2.rectangle(mask, (int(x), int(y)), (int(x + bw), int(y + bh)), 1, -1)
    return mask


def crop_box(bbox, h, w, pad_frac):
    """Padded, clipped integer crop box (x0, y0, x1, y1) from a COCO xywh bbox."""
    x, y, bw, bh = bbox
    pad = pad_frac * max(bw, bh)
    x0 = int(np.floor(x - pad)); y0 = int(np.floor(y - pad))
    x1 = int(np.ceil(x + bw + pad)); y1 = int(np.ceil(y + bh + pad))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _write_file(out_file, specs, prov):
    """Write one AVA-compatible HDF5 (specs + provenance) for a single chunk."""
    with h5py.File(out_file, "w") as f:
        f.create_dataset("specs", data=np.stack(specs), dtype="float32")
        f.create_dataset("ann_id", data=np.array([p["ann_id"] for p in prov]))
        f.create_dataset("score", data=np.array([p["score"] for p in prov], dtype="float32"))
        f.create_dataset("file_name", data=np.array([p["file_name"] for p in prov], dtype="S128"))
        f.create_dataset("coco", data=np.array([p["coco"] for p in prov], dtype="S256"))
        f.create_dataset("recording", data=np.array([p["recording"] for p in prov], dtype="S64"))
        f.create_dataset("bbox", data=np.array([p["bbox"] for p in prov], dtype="float32"))
        f.create_dataset("window_start_sec", data=np.array([p["window_start_sec"] for p in prov], dtype="float32"))
        f.create_dataset("window_end_sec", data=np.array([p["window_end_sec"] for p in prov], dtype="float32"))


class ChunkWriter:
    """Streams specs into equal-count HDF5 files (AVA requires uniform sylls_per_file).

    Files hold exactly `sylls_per_file` specs. The trailing remainder is written
    only if it would otherwise be the *sole* file (so a small dataset keeps every
    call); once >=1 full file exists the remainder is dropped to keep files
    uniform. `add()` is called on the raw and masked writers in lockstep, so
    file k, row i is the same annotation in both arms.
    """

    def __init__(self, out_dir, sylls_per_file):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spf = sylls_per_file
        self.buf_specs, self.buf_prov = [], []
        self.file_num = 0
        self.n_written = 0

    def add(self, spec, prov):
        self.buf_specs.append(spec)
        self.buf_prov.append(prov)
        if len(self.buf_specs) >= self.spf:
            self._flush(self.spf)

    def _flush(self, n):
        out = self.dir / f"syllables_{self.file_num:04d}.hdf5"
        _write_file(out, self.buf_specs[:n], self.buf_prov[:n])
        self.file_num += 1
        self.n_written += n
        self.buf_specs = self.buf_specs[n:]
        self.buf_prov = self.buf_prov[n:]

    def close(self):
        """Flush/drop the remainder; return the number of dropped specs."""
        rem = len(self.buf_specs)
        if rem == 0:
            return 0
        if self.file_num == 0:  # whole dataset < one chunk: keep all in one file
            self._flush(rem)
            return 0
        self.buf_specs, self.buf_prov = [], []  # drop to keep files uniform
        return rem


def build_specs(coco_paths, out_dir, spec_dir_override, pad_frac, min_score,
                min_crop_px, sylls_per_file):
    """Stream every kept annotation into {out_dir}/raw and {out_dir}/masked.

    Returns (n_written, dropped, skipped). Memory stays flat: only the current
    chunk (`sylls_per_file` specs per arm) plus one coco file's PNGs are held.
    """
    raw_w = ChunkWriter(Path(out_dir) / "raw", sylls_per_file)
    masked_w = ChunkWriter(Path(out_dir) / "masked", sylls_per_file)
    skipped = {"missing_png": 0, "small_crop": 0, "low_score": 0}

    for ci, coco_path in enumerate(coco_paths):
        coco_path = Path(coco_path)
        spec_dir = resolve_spec_dir(coco_path, spec_dir_override)
        data = json.loads(coco_path.read_text())
        images = {im["id"]: im for im in data["images"]}
        # PNGs are only referenced within their own coco file, so cache per-file
        # to bound memory when scanning thousands of recordings.
        img_cache: dict[str, np.ndarray] = {}
        if ci % 50 == 0:
            print(f"  [{ci}/{len(coco_paths)}] {coco_path}  (kept so far: {raw_w.n_written})")

        for ann in data["annotations"]:
            if min_score is not None and ann.get("score", 1.0) < min_score:
                skipped["low_score"] += 1
                continue
            im_meta = images[ann["image_id"]]
            key = str(spec_dir / im_meta["file_name"])
            if key not in img_cache:
                arr = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
                img_cache[key] = None if arr is None else arr.astype(np.float32) / 255.0
            img = img_cache[key]
            if img is None:
                skipped["missing_png"] += 1
                continue
            h, w = img.shape

            x0, y0, x1, y1 = crop_box(ann["bbox"], h, w, pad_frac)
            if x1 - x0 < min_crop_px or y1 - y0 < min_crop_px:
                skipped["small_crop"] += 1
                continue

            patch = img[y0:y1, x0:x1]
            mask = poly_mask(ann.get("segmentation"), ann["bbox"], h, w)[y0:y1, x0:x1]

            spec_raw = cv2.resize(patch, X_SHAPE[::-1], interpolation=cv2.INTER_AREA)
            mask_rs = cv2.resize(mask, X_SHAPE[::-1], interpolation=cv2.INTER_NEAREST)
            spec_masked = spec_raw * mask_rs  # ONLY difference between the two arms

            prov = {
                "coco": str(coco_path),
                "recording": recording_id(coco_path),
                "file_name": im_meta["file_name"],
                "ann_id": int(ann["id"]),
                "bbox": [float(v) for v in ann["bbox"]],
                "score": float(ann.get("score", 1.0)),
                "window_start_sec": float(im_meta.get("window_start_sec", np.nan)),
                "window_end_sec": float(im_meta.get("window_end_sec", np.nan)),
            }
            raw_w.add(np.clip(spec_raw, 0.0, 1.0).astype(np.float32), prov)
            masked_w.add(np.clip(spec_masked, 0.0, 1.0).astype(np.float32), prov)

    dropped = raw_w.close()
    masked_w.close()  # same buffer length -> drops the identical remainder
    assert raw_w.n_written == masked_w.n_written, "arms desynced"
    return raw_w.n_written, dropped, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", nargs="+", required=True,
                    help="coco_ch_*.json path(s) or glob(s).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir; writes {out}/raw/syllables_NNNN.hdf5 and {out}/masked/...")
    ap.add_argument("--spec-dir", type=Path, default=None,
                    help="Override PNG directory (default: auto sam3_best->spectrograms/gerbil_ssl).")
    ap.add_argument("--pad-frac", type=float, default=0.1,
                    help="Context padding around each bbox, as a fraction of its longer side.")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Drop annotations below this SAM3 score before building.")
    ap.add_argument("--min-crop-px", type=int, default=4,
                    help="Skip crops smaller than this in either dimension.")
    ap.add_argument("--sylls-per-file", type=int, default=128,
                    help="Specs per HDF5 chunk (must be uniform for AVA's loader). Default 128.")
    args = ap.parse_args()

    coco_paths = sorted({p for pat in args.coco for p in globmod.glob(pat, recursive=True)})
    if not coco_paths:
        ap.error(f"No coco files matched: {args.coco}")
    print(f"{len(coco_paths)} coco file(s)")

    n, dropped, skipped = build_specs(
        coco_paths, args.out, args.spec_dir, args.pad_frac, args.min_score,
        args.min_crop_px, args.sylls_per_file)
    if n == 0:
        ap.error(f"No usable annotations. Skipped: {skipped}")

    print(f"Wrote {n} paired specs -> {args.out}/{{raw,masked}}/ "
          f"({n // args.sylls_per_file if n >= args.sylls_per_file else 1} files/arm, "
          f"dropped remainder {dropped}, skipped {skipped})")


if __name__ == "__main__":
    main()
