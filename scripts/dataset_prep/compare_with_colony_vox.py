"""Check data/mongolia_audiomoth against the lab's Colony_Vox tree, file by file.

Walks every WAV in Colony_Vox and reports each as linked (data/<same path>/mic_0_<stem>.wav points at it), skipped on
purpose (outputs/mongolia_audiomoth/skipped.txt / examples.txt), or MISSING; also links that point anywhere else.

    python scripts/dataset_prep/compare_with_colony_vox.py
"""
import os
from collections import Counter
from pathlib import Path

SRC = Path("/mnt/ceph/users/neurostatslab/saneslab_data/mongolia/Audiomoth_Files/Colony_Vox")
DATA = Path("data/mongolia_audiomoth")


def main():
    listed = set()
    for f in ("skipped.txt", "examples.txt"):
        p = Path("outputs/mongolia_audiomoth") / f
        if p.exists():
            listed |= {l.strip() for l in open(p) if l.strip()}
    status, missing = Counter(), []
    for dirpath, _, files in os.walk(SRC):
        for name in files:
            if not name.lower().endswith(".wav") or name.startswith("._"):
                continue
            p = Path(dirpath) / name
            rid = p.relative_to(SRC).with_suffix("").as_posix()
            link = DATA / rid / f"mic_0_{p.stem}.wav"
            if link.is_symlink() and Path(os.path.realpath(link)) == p:
                status["linked, same path"] += 1
            elif str(p) in listed:
                status["skipped on purpose"] += 1
            else:
                status["MISSING"] += 1
                missing.append(rid)
    stray = [l for l in DATA.rglob("mic_0_*.wav")
             if not Path(os.path.realpath(l)).is_relative_to(SRC)
             or Path(os.path.realpath(l)).relative_to(SRC).with_suffix("").as_posix() != l.parent.relative_to(DATA).as_posix()]
    for k, v in status.items():
        print(f"{k:22} {v}")
    print(f"{'links elsewhere':22} {len(stray)}")
    for r in missing[:20]:
        print("  missing:", r)


if __name__ == "__main__":
    main()
