"""Spectrogram generation utilities shared across all runners."""
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import cv2
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


def parse_spec_fname(fname):
    """Extract (channel, t0, t1) from a headmic chunk filename, or (None, None, None)."""
    m = re.match(r'headmic_(\d+)_.*_t([\d.]+)-([\d.]+)\.png$', Path(fname).name)
    if m:
        return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


def load_channel_audio(recording_dir, ch):
    """Return (sr, audio, base_name) for headmic_{ch}_*.wav, or None if not found."""
    matches = sorted(glob(str(Path(recording_dir) / f"headmic_{ch}_*.wav")))
    if not matches:
        return None
    sr, audio = wavfile.read(matches[0])
    return sr, audio, Path(matches[0]).stem


def write_chunk_spectrograms(audio, sr, spec_dir, base_name, chunk_sec=1.0):
    """Slice audio into fixed-length chunks and write each as a spectrogram PNG.

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
            shape = write_spectrogram_img(audio[start_i:stop_i], sr, path)
        paths.append(path)
        shapes.append(shape)
    return paths, shapes


def group_specs_by_channel(spec_dir, channels=None):
    """Scan spec_dir for headmic PNGs; return dict ch → [(path, t0, t1)], sorted by channel."""
    by_ch = defaultdict(list)
    for p in sorted(Path(spec_dir).glob("*.png")):
        ch, t0, t1 = parse_spec_fname(p.name)
        if ch is not None and (channels is None or ch in channels):
            by_ch[ch].append((p, t0, t1))
    return dict(sorted(by_ch.items()))


def default_spec_dir(out_dir):
    """Infer shared spectrogram directory from an output directory.

    Follows the convention outputs/<tool>/<exp_id> → outputs/spectrograms/<exp_id>.
    """
    out_dir = Path(out_dir)
    return out_dir.parent.parent / "spectrograms" / out_dir.name
