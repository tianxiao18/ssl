"""Symlink the AudioMoth recordings into data/mongolia_audiomoth/<path in Colony_Vox>/mic_0_<stem>.wav.

Recording ids are their path inside Colony_Vox without the extension (e.g. Colony_1A/20260628/AM10_20260628/<stem>),
so every tree keyed by recording mirrors the lab's original folder; the per-recording folder only holds the
mic_<ch>_ file the pipeline expects.

Source: every colony folder in Audiomoth_Files/Colony_Vox on ceph. Skips non-WAV files, empty/truncated WAVs
(< 1 MB), *_clipped.wav (excerpts of a full WAV), the Audiomoth_Regions excerpts (= data/mongolia_wild_data), the
Audiomoth_Marked_Regions copies (marked duplicates of AMPK1/AMPK2, used by marked_regions_to_csv.py), folders
flagged corrupt or with a reset clock (19700101), and the call-example clips ("... calls ...wav"), which go to
data/mongolia_audiomoth_examples/<slug>/ as a positive control.
Writes outputs/mongolia_audiomoth/{recordings,durations,examples,skipped}.txt and shard_<i>.txt (by audio duration).
"""
import argparse
import re
from pathlib import Path

import soundfile as sf

SRC = Path("/mnt/ceph/users/neurostatslab/saneslab_data/mongolia/Audiomoth_Files/Colony_Vox")
BAD_DIRS = ("may_be_corrupt", "19700101", "Audiomoth_Regions", "Audiomoth_Marked_Regions")


def wavs(colony):
    """Every *.wav up to 5 levels below a colony (date dir / device dir / [day dir]; marked copies one deeper)."""
    for pat in ("*/*.wav", "*/*/*.wav", "*/*/*/*.wav", "*/*/*/*/*.wav", "*/*/*/*/*/*.wav",
                "*/*.WAV", "*/*/*.WAV", "*/*/*/*.WAV", "*/*/*/*/*.WAV", "*/*/*/*/*/*.WAV"):
        yield from (SRC / colony).glob(pat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=16)
    a = ap.parse_args()
    data, out = Path("data/mongolia_audiomoth"), Path("outputs/mongolia_audiomoth")
    out.mkdir(parents=True, exist_ok=True)
    recs, examples, skipped = [], [], []
    colonies = sorted(d.name for d in SRC.iterdir() if d.is_dir() and d.name.startswith("Colony_"))
    for colony in colonies:
        for p in sorted(set(wavs(colony))):
            if p.name.startswith("._"):  # macOS AppleDouble metadata
                continue
            rel = p.relative_to(SRC / colony)
            if any(b in str(rel) for b in BAD_DIRS):
                skipped.append(p)
            elif "calls" in p.name.lower():
                examples.append(p)
            elif p.stat().st_size < 1e6 or p.stem.endswith("_clipped"):
                skipped.append(p)
            else:
                recs.append((f"{colony}/{rel.with_suffix('').as_posix()}", p, sf.info(str(p)).duration))
    for rel, p, _ in recs:
        d = data / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / f"mic_0_{p.stem}.wav").unlink(missing_ok=True)
        (d / f"mic_0_{p.stem}.wav").symlink_to(p)
    ex_rels = []
    for p in examples:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", p.stem).strip("_")
        d = Path("data/mongolia_audiomoth_examples") / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / f"mic_0_{slug}.wav").unlink(missing_ok=True)
        (d / f"mic_0_{slug}.wav").symlink_to(p)
        ex_rels.append(slug)
    (out / "recordings.txt").write_text("".join(f"{r}\n" for r, _, _ in recs))
    (out / "examples.txt").write_text("".join(f"{r}\n" for r in ex_rels))
    (out / "durations.txt").write_text("".join(f"{r}\t{d:.1f}\n" for r, _, d in recs))
    (out / "skipped.txt").write_text("".join(f"{p}\n" for p in skipped))
    loads = [[0.0, []] for _ in range(a.shards)]
    for rel, _, dur in sorted(recs, key=lambda x: -x[2]):
        s = min(loads, key=lambda x: x[0])
        s[0] += dur
        s[1].append(rel)
    for i, (_, rels) in enumerate(loads):
        (out / f"shard_{i}.txt").write_text("".join(f"{r}\n" for r in sorted(rels)))
    print(f"{len(colonies)} colonies, {len(recs)} recordings, {sum(d for *_, d in recs) / 3600:.1f} h; "
          f"{len(examples)} examples; skipped {len(skipped)} (outputs/mongolia_audiomoth/skipped.txt)")


if __name__ == "__main__":
    main()
