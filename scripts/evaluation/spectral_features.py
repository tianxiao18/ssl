"""
Compute librosa spectral features for detected vocalization events.

Each detection is a COCO annotation: a bbox in spectrogram-pixel space on a
1-second chunk whose absolute span is (window_start_sec, window_end_sec). We map
each bbox back to
  * a time segment      [t0, t1]      of the source headmic audio, and
  * a frequency band    [f_low, f_high]
then compute librosa spectral features *restricted to that band* (the STFT
magnitude is sliced to the band's bins before every feature, and the
band-limited ZCR is taken on a brick-wall band-passed copy of the segment). So a
feature such as spectral_centroid describes energy WITHIN the detected call box,
not the whole 0-62.5 kHz spectrum.

Pixel -> physical mapping (must match vox_tracer/spec.py, which builds each PNG
with scipy.signal.spectrogram(nperseg=512, noverlap=256) then np.flipud):
  time  : t0 = window_start + (x   / W) * (window_end - window_start)
          t1 = window_start + (x+w / W) * (window_end - window_start)
  freq  : row 0 is the TOP of the flipped image = Nyquist; row H-1 = 0 Hz, so
          f_high = (H-1 - y    )/(H-1) * nyquist   (box top)
          f_low  = (H-1 - (y+h))/(H-1) * nyquist   (box bottom)

Output: one CSV per method, one row per detected event, provenance columns +
<feat>_mean / <feat>_std aggregated over the event's STFT frames.

Example
-------
    python scripts/evaluation/spectral_features.py \
        --methods sam3_best ridge squeakout --dataset gerbil_ssl \
        --out-dir outputs/spectral_features
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vox_tracer.scoring import iou_1d, load_gt_from_csv  # noqa: E402
from vox_tracer.spec import NPERSEG, load_channel_audio, parse_spec_fname  # noqa: E402


@lru_cache(maxsize=8)
def gt_vox_intervals(recording_dir):
    """Cached list of (start, stop) 'vox' GT events for a recording, or None if no GT csv.

    GT is recording-level (channel-agnostic) and covers the whole recording, so a
    detection is TP if it time-overlaps any vox interval and FP otherwise.
    """
    try:
        vox, _ = load_gt_from_csv(recording_dir)
        return vox
    except FileNotFoundError:
        return None

import warnings  # noqa: E402

import librosa  # noqa: E402

# Many detected syllables are shorter than one STFT window; librosa zero-pads
# them (fine here) but warns each time — silence just that message.
warnings.filterwarnings("ignore", message="n_fft=.* is too large")

# STFT grid for feature extraction. NPERSEG (512) matches the grid the detector
# saw, giving ~244 Hz bins at 125 kHz; a 1/4 hop is librosa's usual default.
N_FFT = NPERSEG
HOP = N_FFT // 4
EPS = 1e-10

# Feature name -> aggregated over frames as mean & std.
FEATURES = ["centroid", "bandwidth", "rolloff", "flatness", "rms", "zcr", "peak_freq"]


def bbox_to_band(y, h, H, nyquist):
    """Detection bbox rows -> (f_low, f_high) in Hz (image is flipud, so row 0 = Nyquist)."""
    f_high = (H - 1 - y) / (H - 1) * nyquist
    f_low = (H - 1 - (y + h)) / (H - 1) * nyquist
    lo, hi = sorted((max(0.0, f_low), min(nyquist, f_high)))
    return lo, hi


def bbox_to_time(x, w, win_start, win_end, W):
    """Detection bbox columns -> (t0, t1) absolute seconds within the recording."""
    dur = win_end - win_start
    return win_start + (x / W) * dur, win_start + ((x + w) / W) * dur


def event_features(audio, sr, t0, t1, f_low, f_high):
    """Band-restricted librosa spectral features for one event -> {feat: (mean, std)}.

    Returns None if the audio segment is empty. All spectral features are taken
    on the STFT sliced to [f_low, f_high]; ZCR is taken on a brick-wall
    band-passed copy of the segment.
    """
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(audio), int(round(t1 * sr)))
    if i1 <= i0:
        return None
    seg = audio[i0:i1].astype(np.float32)

    S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP)) + EPS  # (n_bins, n_frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    mask = (freqs >= f_low) & (freqs <= f_high)
    if not mask.any():  # band narrower than one bin: keep the nearest bin
        mask[np.argmin(np.abs(freqs - 0.5 * (f_low + f_high)))] = True
    S_b = S[mask]
    f_b = freqs[mask]

    cent = librosa.feature.spectral_centroid(S=S_b, freq=f_b)
    feats = {
        "centroid": cent,
        "bandwidth": librosa.feature.spectral_bandwidth(S=S_b, freq=f_b, centroid=cent),
        "rolloff": librosa.feature.spectral_rolloff(S=S_b, freq=f_b, roll_percent=0.85),
        "flatness": librosa.feature.spectral_flatness(S=S_b),
        # In-band RMS per frame from the sliced magnitudes (librosa.feature.rms
        # needs a full-length spectrum, so compute it directly). Parseval: the
        # 2x accounts for the dropped negative frequencies of the real FFT.
        "rms": np.sqrt(2.0 * np.sum(S_b ** 2, axis=0))[None, :] / N_FFT,
        "peak_freq": f_b[np.argmax(S_b, axis=0)][None, :],
    }

    # Band-limited ZCR: brick-wall bandpass in the frequency domain, then ZCR.
    Z = np.fft.rfft(seg)
    zf = np.fft.rfftfreq(len(seg), 1.0 / sr)
    Z[(zf < f_low) | (zf > f_high)] = 0.0
    seg_bp = np.fft.irfft(Z, n=len(seg)).astype(np.float32)
    fl = min(N_FFT, max(2, len(seg_bp)))
    feats["zcr"] = librosa.feature.zero_crossing_rate(seg_bp, frame_length=fl, hop_length=HOP)

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in feats.items()}


def process_recording(coco_path, recording_dir, ch, method):
    """All detections in one coco file -> list of feature rows (audio loaded once)."""
    coco = json.loads(Path(coco_path).read_text())
    if not coco["annotations"]:
        return [], 0
    loaded = load_channel_audio(recording_dir, ch)
    if loaded is None:
        return None, 0  # signal missing audio
    sr, audio, _ = loaded
    nyquist = sr / 2.0
    img_by_id = {im["id"]: im for im in coco["images"]}
    vox = gt_vox_intervals(str(recording_dir))  # None if this recording has no GT csv

    rows, degenerate = [], 0
    exp, idx = Path(coco_path).parts[-3], Path(coco_path).parts[-2]
    for ann in coco["annotations"]:
        im = img_by_id[ann["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = ann["bbox"]
        t0, t1 = bbox_to_time(x, w, im["window_start_sec"], im["window_end_sec"], W)
        f_low, f_high = bbox_to_band(y, h, H, nyquist)

        # TP = detection overlaps any GT vox event; FP otherwise; None if no GT.
        if vox is None:
            label = None
        else:
            label = "tp" if any(iou_1d(t0, t1, gs, ge) > 0 for gs, ge in vox) else "fp"

        row = {
            "method": method, "experiment": exp, "idx": idx, "channel": ch,
            "ann_id": ann["id"], "file_name": im["file_name"], "score": ann.get("score"),
            "label": label,
            "t_start_sec": t0, "t_stop_sec": t1, "duration_sec": t1 - t0,
            "f_low_hz": f_low, "f_high_hz": f_high, "f_band_hz": f_high - f_low,
        }
        feats = event_features(audio, sr, t0, t1, f_low, f_high)
        if feats is None:
            degenerate += 1
            for name in FEATURES:
                row[f"{name}_mean"] = row[f"{name}_std"] = np.nan
        else:
            for name, (m, s) in feats.items():
                row[f"{name}_mean"], row[f"{name}_std"] = m, s
        rows.append(row)
    return rows, degenerate


def discover(method_root, dataset):
    """Yield (coco_path, recording_dir, channel) for every coco_ch_*.json under a method root."""
    for coco in sorted(Path(method_root).glob("experiment_*/idx_*/coco_ch_*.json")):
        ch = int(coco.stem.split("_")[-1])
        exp, idx = coco.parts[-3], coco.parts[-2]
        yield coco, Path("data") / dataset / exp / idx, ch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs", help="dir holding <method>/<dataset>/ trees")
    ap.add_argument("--methods", nargs="+",
                    default=["sam3_best", "ridge_filtered", "squeakout_filtered"])
    ap.add_argument("--dataset", required=True, choices=("gerbil_ssl", "dryad_gerbil", "gerbil_family"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/spectral_features"))
    ap.add_argument("--limit", type=int, default=None, help="cap coco files per method (debug)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        root = Path(args.outputs_root) / method / args.dataset
        tasks = list(discover(root, args.dataset))
        if args.limit:
            tasks = tasks[:args.limit]
        print(f"\n[{method}] {len(tasks)} coco files")

        rows, degenerate, missing, done = [], 0, 0, 0
        for coco, rec_dir, ch in tasks:
            r, deg = process_recording(coco, rec_dir, ch, method)
            if r is None:
                missing += 1
                continue
            rows.extend(r)
            degenerate += deg
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)}  events so far: {len(rows)}")

        out = args.out_dir / f"{method}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[{method}] wrote {len(rows)} events -> {out} "
              f"(missing-audio recordings: {missing}, degenerate bboxes: {degenerate})")


if __name__ == "__main__":
    main()
