"""Playback of a candidate: band-passed, optionally slowed down, as WAV bytes.

Slowing by `slow` divides every frequency by it (10x puts 25 kHz at 2.5 kHz). Output
is resampled to 48 kHz, since some browsers (Safari) reject odd rates like 7812 Hz,
so at slow=1 anything above 24 kHz is dropped.
"""
import io
import math

import numpy as np
import soundfile as sf
from fractions import Fraction

from scipy.signal import butter, resample_poly, sosfiltfilt

OUT_SR = 48000


def snippet_wav(path, t0, t1, f_lo_hz, f_hi_hz, slow=1):
    """(wav_bytes, t_lo, t_hi): [t0, t1] of `path`, clamped to the file."""
    info = sf.info(str(path))
    sr = info.samplerate
    i0 = max(0, int(math.floor(t0 * sr)))
    i1 = min(info.frames, int(math.ceil(t1 * sr)))
    x, _ = sf.read(str(path), start=i0, stop=max(i1, i0 + 1), dtype="float64",
                   always_2d=True)
    x = x[:, 0]
    # Same band as the displayed spectrogram, so what is heard is what is shown.
    hi = min(f_hi_hz, 0.98 * sr / 2)
    sos = butter(4, [max(f_lo_hz, 1.0), hi], btype="band", fs=sr, output="sos")
    if len(x) > 3 * (2 * len(sos) + 1):
        x = sosfiltfilt(sos, x)
    # Level on the 99.5th percentile, not the peak: one click would otherwise set the
    # gain and leave the call near silent. tanh soft-clips what lies above it.
    ref = np.percentile(np.abs(x), 99.5)
    if ref > 0:
        x = np.tanh(0.5 * x / ref)
    fade = min(len(x) // 2, int(0.003 * sr))
    if fade:
        ramp = np.linspace(0, 1, fade)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    ratio = Fraction(OUT_SR * slow / sr).limit_denominator(1000)
    x = np.clip(resample_poly(x, ratio.numerator, ratio.denominator), -1, 1)
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), OUT_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue(), i0 / sr, i1 / sr
