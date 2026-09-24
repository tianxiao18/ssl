"""Spectrogram generation utilities shared across all runners."""
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

SPEC_LO, SPEC_HI = -70, 0
NPERSEG, NOVERLAP = 512, 256
SYNC_PAD = 0.05  # seconds of padding added around each DAS event (split before/after)

def write_spectrogram_img(audio, sr, path, lo=SPEC_LO, hi=SPEC_HI):
    """Save raw audio as a normalised dB spectrogram PNG; return (h, w)."""
    _, _, Pxx = spectrogram(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Pxx_dB = np.flipud(10 * np.log10(Pxx + 1e-12))
    a = np.clip(Pxx_dB, lo, hi)
    a = (a - lo) / (hi - lo + 1e-12)
    a = (a * 255).astype(np.uint8)
    cv2.imwrite(str(path), a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return a.shape  # (h, w)


def make_spectrogram_array(audio, sr, lo=SPEC_LO, hi=SPEC_HI):
    """Build a normalised dB spectrogram as a BGR uint8 array; return (img, freqs)."""
    freqs, _, Pxx = spectrogram(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Pxx_dB = np.flipud(10 * np.log10(Pxx + 1e-12))
    a = np.clip(Pxx_dB, lo, hi)
    a = (a - lo) / (hi - lo + 1e-12)
    a = (a * 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), freqs


def parse_spec_fname(fname, prefix="headmic"):
    """Extract (channel, t0, t1) from a ``{prefix}_{ch}_..._t{t0}-{t1}.png`` chunk
    filename, or (None, None, None).

    ``prefix`` names the recording stream (``"headmic"`` for the multi-mic
    gerbil_ssl rig; a single-stream dataset can pass its own label, e.g.
    ``"mic"``, and use channel ``0``) so new datasets don't have to alias
    themselves as headmic mics.
    """
    m = re.match(rf'{re.escape(prefix)}_(\d+)_.*_t([\d.]+)-([\d.]+)\.png$', Path(fname).name)
    if m:
        return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


def load_channel_audio(recording_dir, ch, prefix="headmic"):
    """Return (sr, audio, base_name) for {prefix}_{ch}_*.wav, or None if not found.

    Integer-PCM wavs are rescaled to the float [-1, 1] convention the float
    datasets use, so one (lo, hi) dB scale applies to every dataset (raw int32
    sits +186.6 dB above it and clips to solid white).
    """
    matches = sorted(glob(str(Path(recording_dir) / f"{prefix}_{ch}_*.wav")))
    if not matches:
        return None
    sr, audio = wavfile.read(matches[0])
    if np.issubdtype(audio.dtype, np.signedinteger):
        audio = audio.astype(np.float32) / float(2 ** (8 * audio.dtype.itemsize - 1))
    elif audio.dtype == np.uint8:  # 8-bit PCM is the one unsigned case, centred on 128
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    return sr, audio, Path(matches[0]).stem


def write_chunk_spectrograms(audio, sr, spec_dir, base_name, chunk_sec=1.0, lo=SPEC_LO, hi=SPEC_HI):
    """Slice audio into fixed-length chunks and write each as a spectrogram PNG.

    lo/hi override the module-default dB clipping window — pass a recording-specific
    window for audio whose amplitude scale doesn't match the default calibration (e.g.
    a different recording rig than the one SPEC_LO/SPEC_HI were tuned for).

    Skips chunks whose PNG already exists on disk.
    Returns (paths, shapes) in chunk order.
    """
    spec_dir = Path(spec_dir)
    chunk_len = int(chunk_sec * sr)
    n_chunks = len(audio) // chunk_len
    paths, shapes = [], []
    for i in range(n_chunks):
        start_i = i * chunk_len
        stop_i = start_i + chunk_len
        t0, t1 = start_i / sr, stop_i / sr
        path = spec_dir / f"{base_name}_chunk_{i:05d}_t{t0:.2f}-{t1:.2f}.png"
        if path.exists():
            shape = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape
        else:
            shape = write_spectrogram_img(audio[start_i:stop_i], sr, path, lo=lo, hi=hi)
        paths.append(path)
        shapes.append(shape)
    return paths, shapes


def write_recording_spectrogram_h5(audio, sr, h5_path, lo=SPEC_LO, hi=SPEC_HI, chunk_sec=1.0,
                                    compression="gzip", compression_opts=4):
    """Compute ONE continuous STFT over the whole recording (no per-chunk truncation
    at the STFT-input stage) and write it to a single HDF5 file per recording.

    The dataset is chunked internally at ~chunk_sec of columns (HDF5's own on-disk
    block layout, unrelated to the STFT computation) so a later window read only
    decompresses the block(s) it needs instead of the whole array.

    Returns (h5_path, (freq_bins, n_time_bins)).

    Computed in float32 with in-place ops throughout: a full-recording STFT in
    float64 needs ~4-5 live copies of a (freq_bins x n_time_bins) array at once
    (power, dB, clipped, normalised, ...), which for a long recording (e.g. a
    60-min/125kHz dryad_gerbil_full session) peaks around 15-18GB RSS. float32 +
    in-place halves the per-array cost and avoids the extra copies.
    """
    audio = np.asarray(audio, dtype=np.float32)
    freqs, t, Pxx = spectrogram(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    del audio
    Pxx += np.float32(1e-12)
    np.log10(Pxx, out=Pxx)
    Pxx *= 10  # Pxx now holds dB values in place
    Pxx = np.flipud(Pxx)  # view, not a copy
    np.clip(Pxx, lo, hi, out=Pxx)
    Pxx -= lo
    Pxx /= (hi - lo + 1e-12)
    Pxx *= 255
    img_u8 = Pxx.astype(np.uint8)

    hop = NPERSEG - NOVERLAP
    cols_per_chunk = max(1, round(chunk_sec * sr / hop))
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset(
            "spec", data=img_u8,
            chunks=(img_u8.shape[0], min(cols_per_chunk, img_u8.shape[1])),
            compression=compression, compression_opts=compression_opts,
        )
        ds.attrs["sr"] = sr
        ds.attrs["nperseg"] = NPERSEG
        ds.attrs["noverlap"] = NOVERLAP
        ds.attrs["lo"] = lo
        ds.attrs["hi"] = hi
        f.create_dataset("t", data=t)  # time-bin centers (sec), global to the recording
    return h5_path, img_u8.shape


def read_h5_window(h5_path, t0, t1):
    """Column-slice a recording's HDF5 spectrogram for real time window [t0, t1).

    Returns a uint8 (freq, time) array -- same pixel convention as
    write_spectrogram_img/make_spectrogram_array.
    """
    with h5py.File(h5_path, "r") as f:
        t = f["t"][:]
        c0, c1 = np.searchsorted(t, [t0, t1])
        return f["spec"][:, c0:c1]


def group_specs_by_channel(spec_dir, channels=None, prefix="headmic"):
    """Scan spec_dir for {prefix}_{ch}_..._t{t0}-{t1}.png PNGs; return dict ch → [(path, t0, t1)], sorted by channel."""
    by_ch = defaultdict(list)
    for p in sorted(Path(spec_dir).glob("*.png")):
        ch, t0, t1 = parse_spec_fname(p.name, prefix=prefix)
        if ch is not None and (channels is None or ch in channels):
            by_ch[ch].append((p, t0, t1))
    return dict(sorted(by_ch.items()))


def h5_windows(h5_path, chunk_sec=1.0):
    """[(h5_path, t0, t1), ...] tiling an HDF5 recording's full duration at chunk_sec."""
    with h5py.File(h5_path, "r") as f:
        duration = float(f["t"][-1])
    n_chunks = int(duration // chunk_sec)
    return [(h5_path, round(i * chunk_sec, 2), round((i + 1) * chunk_sec, 2)) for i in range(n_chunks)]


def spec_windows_by_channel(spec_dir, channels=None, prefix="headmic", chunk_sec=1.0):
    """group_specs_by_channel, falling back to {prefix}_{ch}_*.h5 windows for channels with no PNGs."""
    by_ch = group_specs_by_channel(spec_dir, channels, prefix=prefix)
    for ch in (channels or []):
        if by_ch.get(ch):
            continue
        h5_matches = sorted(glob(str(Path(spec_dir) / f"{prefix}_{ch}_*.h5")))
        if h5_matches:
            by_ch[ch] = h5_windows(Path(h5_matches[0]), chunk_sec=chunk_sec)
    return dict(sorted(by_ch.items()))


def window_fname(source, t0, t1, chunk_sec=1.0):
    """COCO file_name for a window: the PNG's name, or the chunk name an HDF5 window would have had."""
    if str(source).endswith(".h5"):
        idx = round(t0 / chunk_sec) if chunk_sec else 0
        return f"{Path(source).stem}_chunk_{idx:05d}_t{t0:.2f}-{t1:.2f}.png"
    return Path(source).name


class WindowReader:
    """Read a gray window from a PNG path or an HDF5 (t0, t1) slice, keeping the last HDF5 open."""

    def __init__(self):
        self._path, self._f, self._t = None, None, None

    def __call__(self, source, t0, t1):
        if not str(source).endswith(".h5"):
            return cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if source != self._path:
            self.close()
            self._f = h5py.File(source, "r")
            self._t = self._f["t"][:]
            self._path = source
        c0, c1 = np.searchsorted(self._t, [t0, t1])
        return self._f["spec"][:, c0:c1]

    def close(self):
        if self._f is not None:
            self._f.close()
        self._path, self._f, self._t = None, None, None


def calibrate_db_range(audio, sr, lo_pct=1.0, hi_pct=99.9, n_windows=6, window_sec=20.0):
    """Percentile-based (lo, hi) dB window for one recording's own amplitude scale.

    Sampled from several windows spread across the recording (not just the
    start) so a quiet intro or a loud bout doesn't skew the calibration. Use
    this instead of the module default SPEC_LO/SPEC_HI for audio recorded on a
    rig with a different amplitude scale (SPEC_LO/SPEC_HI were tuned for the
    gerbil_ssl headmic setup and can clip an unrelated dataset to near-black).
    """
    win = int(window_sec * sr)
    n = len(audio)
    starts = np.linspace(0, max(n - win, 0), n_windows, dtype=int)
    dbs = []
    for s in starts:
        seg = audio[s:s + win]
        if len(seg) < NPERSEG:
            continue
        _, _, Pxx = spectrogram(seg, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
        dbs.append(10 * np.log10(Pxx + 1e-12).ravel())
    allb = np.concatenate(dbs)
    return float(np.percentile(allb, lo_pct)), float(np.percentile(allb, hi_pct))
