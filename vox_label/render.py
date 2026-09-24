"""Render a spectrogram crop for an arbitrary time window, as JPEG bytes.

Two storage layouts exist in this repo and a labeling campaign should not care which
one its corpus uses, so both sit behind the same `crop()` signature:

* `PngChunkSource` -- per-second chunk PNGs under outputs/spectrograms/<dataset>/...
  This is what gerbil_ssl has (17 GB of them). A candidate straddling a chunk boundary
  needs its covering chunks stitched before slicing, which is why this is more than a
  file read. `scripts/archive/ridge_failure_viz.py:170` sidesteps the problem by
  showing only the chunk containing the candidate's midpoint; that silently truncates
  anything crossing a boundary, which for a labeling tool would mean showing half a
  call and asking whether it is real.
* `H5Source` -- one HDF5 per recording under outputs/spectrograms_h5/<dataset>/...
  (dryad_gerbil_full). Holds one open handle and a cached time axis, the pattern at
  `scripts/viz_recording_html.py:163-176`, rather than the reopen-and-reread-`t`-every-
  call of `vox_tracer.spec.read_h5_window`, which costs a 4.7 MB read per crop.

Both return `(gray, t_lo, t_hi)`: the uint8 (freq, time) crop and the true time span it
covers, which is not exactly the requested span because both backends can only cut on
column boundaries. The caller needs the true span to place the candidate bracket
correctly -- assuming the requested span would drift the marker by up to a column.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from vox_tracer.spec import group_specs_by_channel


def band_rows(height, nyquist_hz, f_lo_hz, f_hi_hz):
    """Row slice (top, bottom) covering [f_lo, f_hi] Hz.

    The spectrogram is stored flipped (`vox_tracer/spec.py:113`), so row 0 is the
    Nyquist frequency and row height-1 is DC. Cropping the band matters for a labeling
    GUI beyond saving pixels: gerbil calls sit around 20-35 kHz, and the bottom few kHz
    is broadband cage noise that costs vertical space without carrying any of the
    evidence the annotator is judging.
    """
    top = int(round((1 - f_hi_hz / nyquist_hz) * (height - 1)))
    bot = int(round((1 - f_lo_hz / nyquist_hz) * (height - 1)))
    top = max(0, min(top, height - 2))
    bot = max(top + 1, min(bot, height - 1))
    return top, bot + 1


@lru_cache(maxsize=256)
def _imread_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_jpeg(gray, disp_w=900, jpeg_q=72):
    """Resize to `disp_w` preserving aspect, encode as grayscale JPEG, return bytes.

    INTER_AREA is the right filter for downscaling a spectrogram: it averages over the
    source footprint instead of point-sampling, so a one-pixel-wide harmonic dims
    rather than disappearing at some widths and not others.
    """
    if gray.size == 0:
        gray = np.zeros((8, 8), np.uint8)
    h, w = gray.shape
    dh = max(1, int(round(h * disp_w / w)))
    disp = cv2.resize(gray, (disp_w, dh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


class PngChunkSource:
    """Crops stitched from per-second chunk PNGs."""

    def __init__(self, spec_root, dataset, prefix="headmic"):
        self.root = Path(spec_root) / dataset
        self.prefix = prefix
        self._index = {}

    def channels(self, clip):
        return sorted(self._clip_index(clip))

    def _clip_index(self, clip):
        if clip not in self._index:
            self._index[clip] = group_specs_by_channel(self.root / clip, prefix=self.prefix)
        return self._index[clip]

    def crop(self, clip, ch, t0, t1):
        chunks = self._clip_index(clip).get(ch)
        if not chunks:
            raise KeyError(f"no spectrogram chunks for {clip} ch{ch}")
        covering = [c for c in chunks if c[2] > t0 and c[1] < t1]
        if not covering:
            # Requested window falls outside the clip; clamp to the nearest chunk so
            # the GUI shows something rather than erroring out mid-campaign.
            covering = [chunks[0] if t1 <= chunks[0][1] else chunks[-1]]
        mosaic = np.hstack([_imread_gray(p) for p, _, _ in covering])
        m_lo, m_hi = covering[0][1], covering[-1][2]
        px_per_sec = mosaic.shape[1] / (m_hi - m_lo)
        c0 = int(np.floor((t0 - m_lo) * px_per_sec))
        c1 = int(np.ceil((t1 - m_lo) * px_per_sec))
        c0 = max(0, min(c0, mosaic.shape[1] - 1))
        c1 = max(c0 + 1, min(c1, mosaic.shape[1]))
        return mosaic[:, c0:c1], m_lo + c0 / px_per_sec, m_lo + c1 / px_per_sec


class H5Source:
    """Crops column-sliced from a per-recording HDF5 spectrogram."""

    def __init__(self, spec_root, dataset, suffix=".h5"):
        self.root = Path(spec_root) / dataset
        self.suffix = suffix
        self._open = {}

    def channels(self, clip):
        import re
        out = set()
        for p in (self.root / clip).glob(f"*{self.suffix}"):
            m = re.search(r"_(\d+)_", p.name)
            if m:
                out.add(int(m.group(1)))
        return sorted(out) or [0]

    def _handle(self, clip, ch):
        key = (clip, ch)
        if key not in self._open:
            import h5py
            matches = sorted((self.root / clip).glob(f"*_{ch}_*{self.suffix}"))
            if not matches:
                matches = sorted((self.root / clip).glob(f"*{self.suffix}"))
            if not matches:
                raise KeyError(f"no HDF5 spectrogram for {clip} ch{ch}")
            f = h5py.File(matches[0], "r")
            self._open[key] = (f, f["t"][:])
        return self._open[key]

    def crop(self, clip, ch, t0, t1):
        f, t_axis = self._handle(clip, ch)
        c0, c1 = np.searchsorted(t_axis, [t0, t1])
        c0 = max(0, min(int(c0), len(t_axis) - 1))
        c1 = max(c0 + 1, min(int(c1), len(t_axis)))
        return f["spec"][:, c0:c1], float(t_axis[c0]), float(t_axis[c1 - 1])

    def close(self):
        for f, _ in self._open.values():
            f.close()
        self._open.clear()


def make_source(spec_root, dataset, kind="auto", **kw):
    """Pick a backend. `auto` prefers HDF5 when outputs/spectrograms_h5/<dataset> exists."""
    if kind == "png":
        return PngChunkSource(spec_root, dataset, **kw)
    if kind == "h5":
        return H5Source(spec_root, dataset, **kw)
    h5_root = Path("outputs/spectrograms_h5") / dataset
    if h5_root.exists():
        return H5Source("outputs/spectrograms_h5", dataset, **kw)
    return PngChunkSource(spec_root, dataset, **kw)


def clip_channels(source, clip):
    """Every channel the clip has, not merely the ones some detector fired on.

    The GUI shows all of them. Showing only the channels that produced a detection
    would both look arbitrary (one panel or two depending on the candidate) and leak
    part of the detection pattern the annotator is meant to be blind to.
    """
    fn = getattr(source, "channels", None)
    return fn(clip) if fn else []
