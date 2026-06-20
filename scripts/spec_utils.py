"""Shared spectrogram helpers used by run_squeakout_raw.py, run_yolo_vox.py, etc."""
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import spectrogram

SPEC_LO, SPEC_HI = -70, 0
NPERSEG, NOVERLAP = 512, 256
SYNC_PAD = 0.05  # seconds added around each DAS event window (split before/after)


def write_spectrogram_img(audio, sr, path, lo=SPEC_LO, hi=SPEC_HI):
    """Save raw audio as a normalised dB spectrogram PNG; return (h, w)."""
    _, _, Pxx = spectrogram(audio, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Pxx_dB = np.flipud(10 * np.log10(Pxx + 1e-12))
    a = np.clip(Pxx_dB, lo, hi)
    a = (a - lo) / (hi - lo + 1e-12)
    a = (a * 255).astype(np.uint8)
    cv2.imwrite(str(path), a, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return a.shape  # (h, w)


def default_spec_dir(out_dir):
    """Shared spectrograms directory inferred from out_dir.

    Both run_squeakout_raw and run_yolo_vox follow the convention
    outputs/<tool>/<exp_id>, so their natural shared cache is
    outputs/spectrograms/<exp_id>.
    """
    out_dir = Path(out_dir)
    return out_dir.parent.parent / "spectrograms" / out_dir.name
