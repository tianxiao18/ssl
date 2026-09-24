"""What is on disk: spectrogram datasets, frozen pools, campaigns in progress.

Read-only. Backs the setup page so dataset, pool and campaign are picked from
what exists rather than typed as paths.
"""
import csv
import json
import struct
from pathlib import Path

PNG_ROOT = "outputs/spectrograms"
H5_ROOT = "outputs/spectrograms_h5"
CAMPAIGN_ROOT = "outputs/label_campaigns"
DATA_ROOT = "data"


def _subdirs(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)


def _first_file(root, suffix, max_depth=4):
    """First file with `suffix` under root, breadth-first and depth-capped.

    Depth-capped rather than rglob: a dataset directory may be a symlink onto
    bulk storage holding millions of files.
    """
    frontier = [Path(root)]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                entries = sorted(d.iterdir())
            except OSError:
                continue
            for p in entries:
                if p.is_file() and p.suffix == suffix:
                    return p
                if p.is_dir():
                    nxt.append(p)
        frontier = nxt
    return None


def _wav_sample_rate(path):
    """Sample rate from a WAV header, or None.

    Parses the fmt chunk directly: stdlib `wave` rejects float32 WAVs
    (format tag 3), which several datasets here use.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(head):
        cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"fmt " and i + 16 <= len(head):
            return struct.unpack("<I", head[i + 12:i + 16])[0]
        i += 8 + size + (size & 1)
    return None


def _h5_sample_rate(dataset, h5_root=H5_ROOT):
    p = _first_file(Path(h5_root) / dataset, ".h5")
    if p is None:
        return None
    try:
        import h5py
        with h5py.File(p, "r") as f:
            return int(f["spec"].attrs["sr"])
    except Exception:
        return None


def nyquist_khz(dataset, backend, png_root=PNG_ROOT, h5_root=H5_ROOT,
                data_root=DATA_ROOT):
    """sr/2 in kHz for a dataset, or None if it cannot be read.

    Worth getting from the data: the band crop in render.band_rows is in
    fractions of Nyquist, so a wrong value silently shows the wrong frequencies.
    """
    sr = _h5_sample_rate(dataset, h5_root) if backend == "h5" else None
    if sr is None:
        wav = _first_file(Path(data_root) / dataset, ".wav")
        sr = _wav_sample_rate(wav) if wav else None
    return round(sr / 2000, 4) if sr else None


def png_prefix(dataset, png_root=PNG_ROOT):
    """Chunk-PNG stream prefix for a dataset ("headmic", "mic", ...), or None.

    Datasets disagree (gerbil_ssl is headmic_<ch>_, dryad_gerbil is mic_<ch>_)
    and the wrong prefix finds zero chunks, so it is detected not assumed.
    """
    png = _first_file(Path(png_root) / dataset, ".png", max_depth=3)
    if png is None:
        return None
    stem = png.name.split("_chunk_")[0]
    parts = stem.split("_")
    for i, part in enumerate(parts[1:], start=1):
        if part.isdigit():
            return "_".join(parts[:i])
    return None


def datasets(png_root=PNG_ROOT, h5_root=H5_ROOT, data_root=DATA_ROOT):
    """Every dataset with spectrograms, with its backend and display defaults."""
    found = {}
    for p in _subdirs(h5_root):
        found[p.name] = "h5"
    for p in _subdirs(png_root):
        found.setdefault(p.name, "png")
    out = []
    for name in sorted(found):
        backend = found[name]
        nyq = nyquist_khz(name, backend, png_root, h5_root, data_root)
        root = Path(h5_root if backend == "h5" else png_root) / name
        out.append({
            "name": name,
            "backend": backend,
            "spec_root": h5_root if backend == "h5" else png_root,
            "n_clips": len(_subdirs(root)),
            "nyquist_khz": nyq,
            "prefix": png_prefix(name, png_root) if backend == "png" else None,
        })
    return out


def pool_detectors(pool_csv):
    """Detector names in pool order, from the z_* columns of a frozen pool."""
    try:
        with open(pool_csv, newline="") as fh:
            header = next(csv.reader(fh))
    except (OSError, StopIteration):
        return []
    return [c[2:] for c in header if c.startswith("z_")]


def pools(root=CAMPAIGN_ROOT):
    """Frozen candidate pools, newest first."""
    out = []
    for d in _subdirs(root):
        csv_path = d / "candidates.csv"
        if not csv_path.exists():
            continue
        out.append({
            "path": csv_path.as_posix(),
            "dir": d.name,
            "detectors": pool_detectors(csv_path),
            "mtime": csv_path.stat().st_mtime,
            "size_mb": round(csv_path.stat().st_size / 1e6, 1),
        })
    return sorted(out, key=lambda r: -r["mtime"])


def count_labels(labels_path):
    """Labels currently standing in an append-only log, retractions applied.

    Replays ids only, so a campaign can be listed without loading its pool.
    """
    path = Path(labels_path)
    if not path.exists():
        return 0
    live = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("retracted"):
                live.discard(rec.get("cand_id"))
            else:
                live.add(rec.get("cand_id"))
    return len(live)


def guess_dataset(spec, campaign_dir, known):
    """Dataset for a campaign spec, for specs frozen before the field existed.

    Longest known dataset name appearing in the campaign or pool path. A guess:
    the setup page shows it as an editable field.
    """
    if spec.get("dataset"):
        return spec["dataset"]
    hay = f"{campaign_dir} {spec.get('pool_csv', '')} {spec.get('name', '')}"
    hits = [d for d in known if d in hay]
    return max(hits, key=len) if hits else None


def campaigns(root=CAMPAIGN_ROOT, known_datasets=None):
    """Frozen campaigns with their progress, so a half-labeled one can be resumed."""
    known = known_datasets if known_datasets is not None else [
        d["name"] for d in datasets()]
    out = []
    for d in _subdirs(root):
        spec_path = d / "campaign.json"
        if not spec_path.exists():
            continue
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        n_labeled = count_labels(d / "labels.jsonl")
        n_max = spec.get("n_max", 0)
        out.append({
            "dir": d.as_posix(),
            "name": spec.get("name", d.name),
            "dataset": guess_dataset(spec, d.as_posix(), known),
            "dataset_known": bool(spec.get("dataset")),
            "n_labeled": n_labeled,
            "n_max": n_max,
            "n_candidates": spec.get("n_candidates"),
            "detectors": spec.get("detectors", []),
            "rules": spec.get("rules", []),
            "J": spec.get("J"),
            "alpha": spec.get("alpha"),
            "alpha_effective": spec.get("alpha_effective"),
            "seed": spec.get("seed"),
            "recall_stream_max": spec.get("recall_stream_max"),
            "pool_csv": spec.get("pool_csv"),
            "pool_exists": bool(spec.get("pool_csv")) and Path(spec["pool_csv"]).exists(),
            "pct": round(100 * n_labeled / n_max, 1) if n_max else None,
        })
    return sorted(out, key=lambda r: -r["n_labeled"])
