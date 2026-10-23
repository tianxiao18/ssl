"""One-off: re-key mongolia_audiomoth recordings from <colony>/<device>/<stem> to their Colony_Vox path.

New id = the WAV's path inside Colony_Vox without extension (e.g. Colony_1A/20260628/AM10_20260628/<stem>). Moves the
per-recording folder in data/ and in every outputs/<tree>/mongolia_audiomoth (no recompute: contents do not depend on
the path), rewrites ids in list / result files, and re-keys the Mongolia campaigns with candidate order, cand_ids and
labels unchanged (only the clip strings and the pool hash change; originals archived first). Dry run unless --apply.

    python scripts/dataset_prep/migrate_mongolia_layout.py            # report only
    python scripts/dataset_prep/migrate_mongolia_layout.py --apply
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = "mongolia_audiomoth"
SRC = Path("/mnt/ceph/users/neurostatslab/saneslab_data/mongolia/Audiomoth_Files/Colony_Vox")
OLD_ID = re.compile(r"Colony_[0-9A-Z]+/(?:AM|PK)[^/\s\"',]*/[0-9A-F]{16}_\d{8}_\d{6}")
TEXT_DIRS = ["outputs/mongolia_audiomoth", "outputs/eval", "experiments"]
TEXT_SUFFIXES = {".txt", ".csv", ".tsv", ".json", ".psv", ".html"}
CAMPAIGNS = ["outputs/label_campaigns/mongolia_audiomoth_k6", "outputs/label_campaigns/mongolia_audiomoth_all_k6",
             "outputs/label_campaigns/mongolia_audiomoth_k4_hp10k", "outputs/label_campaigns/mongolia_audiomoth_k4_hp15k",
             "outputs/label_campaigns/mongolia_audiomoth_k4_hp24k", "outputs/archive/label_campaigns/mongolia_audiomoth_k4"]
ODD = []
BACKUP = ROOT / "outputs/archive/pre_colony_vox_layout"


def id_map():
    """old id -> new id, from the data/ links (each points at its Colony_Vox WAV)."""
    m = {}
    for old in [l.strip() for l in open(ROOT / f"outputs/{DS}/recordings.txt") if l.strip()]:
        stem = old.split("/")[-1]
        target = Path(os.path.realpath(ROOT / f"data/{DS}/{old}/mic_0_{stem}.wav"))
        new = target.relative_to(SRC).with_suffix("").as_posix()
        assert new.split("/")[0] == old.split("/")[0] and new.endswith(stem), (old, new)
        m[old] = new
    assert len(set(m.values())) == len(m), "new ids collide"
    return m


def trees():
    out = [ROOT / f"data/{DS}"]
    out += sorted(p for p in (ROOT / "outputs").glob(f"*/{DS}") if p.is_dir())
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rewrite_text(path, m, apply):
    s = path.read_text()
    n = 0

    def rep(mo):
        nonlocal n
        if mo.group(0) in m:
            n += 1
            return m[mo.group(0)]
        return mo.group(0)
    t = OLD_ID.sub(rep, s)
    for k in ODD:  # ids the pattern cannot match (e.g. spaces in the name)
        if k in t:
            n += t.count(k)
            t = t.replace(k, m[k])
    if n and apply:
        dst = BACKUP / path.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(t)
        tmp.replace(path)
    return n, len({x for x in OLD_ID.findall(t) if x not in m.values()})


def rekey_campaign(d, m, apply):
    d = ROOT / d
    spec = json.loads((d / "campaign.json").read_text())
    pool = ROOT / spec["pool_csv"] if not Path(spec["pool_csv"]).is_absolute() else Path(spec["pool_csv"])
    if pool.parent != d:  # archived campaign: its spec still names the old location
        pool = d / "candidates.csv"
    csv.field_size_limit(1 << 30)
    new_csv = d / "candidates.csv.rekeyed"
    n = unknown = 0
    with open(pool, newline="") as fi, open(new_csv, "w", newline="") as fo:
        r = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if row["clip"] not in m:
                unknown += 1
            row["clip"] = m.get(row["clip"], row["clip"])
            w.writerow(row)
            n += 1
    if unknown or not apply:
        new_csv.unlink()
        return n, unknown
    arch = ROOT / "outputs/archive/label_campaigns" / (d.name + "__old_layout")
    assert not arch.exists(), arch
    shutil.copytree(d, arch)
    digest = sha256(new_csv)
    new_csv.replace(d / "candidates.csv")
    spec["pool_sha256"] = digest
    spec["rekeyed_from"] = {"pool_sha256": json.loads((arch / "campaign.json").read_text())["pool_sha256"],
                            "archive": arch.relative_to(ROOT).as_posix(),
                            "note": "recording ids re-keyed to Colony_Vox paths; rows, order and cand_ids unchanged"}
    (d / "campaign.json").write_text(json.dumps(spec, indent=1))
    lab = d / "labels.jsonl"
    if lab.exists():
        lines = []
        for line in lab.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if "clip" in rec:
                    rec["clip"] = m.get(rec["clip"], rec["clip"])
                if "campaign" in rec:
                    rec["campaign"] = digest[:12]
                lines.append(json.dumps(rec))
        lab.write_text("".join(l + "\n" for l in lines))
    return n, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.apply and subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-f", "^python[0-9.]* .*label_gui\\.py"],
                                  capture_output=True).stdout.strip():
        raise SystemExit("a label_gui.py is running; close it first")
    os.chdir(ROOT)
    m = id_map()
    global ODD
    ODD = [k for k in m if not OLD_ID.fullmatch(k)]
    print("ids outside the pattern:", ODD)
    print(f"{len(m)} recordings, e.g. {next(iter(m.items()))}")
    if a.apply:
        BACKUP.mkdir(parents=True, exist_ok=True)
        (BACKUP / "id_map_old_to_colony_vox.tsv").write_text(
            "old_id\tcolony_vox_id\n" + "".join(f"{k}\t{v}\n" for k, v in m.items()))

    # 1. folders
    for t in trees():
        lvl3 = [p.relative_to(t).as_posix() for p in t.glob("Colony_*/*/*") if p.is_dir()]
        olds = sorted(o for o in lvl3 if o in m)
        unknown = [o for o in lvl3 if o not in m and OLD_ID.fullmatch(o)]
        clash = [o for o in olds if o in m and (t / m[o]).exists()]
        print(f"{t.relative_to(ROOT)}: {len(olds)} to move" + (f", {len(unknown)} UNKNOWN" if unknown else "")
              + (f", {len(clash)} TARGET EXISTS" if clash else ""))
        if unknown or clash:
            raise SystemExit(f"stop: {(unknown or clash)[:3]}")
        if a.apply:
            for o in olds:
                (t / m[o]).parent.mkdir(parents=True, exist_ok=True)
                os.rename(t / o, t / m[o])
            for dev in sorted(t.glob("Colony_*/*")):  # old device dirs, now empty
                if dev.is_dir() and not any(dev.iterdir()):
                    dev.rmdir()

    # 2. list / result files
    tot = files = 0
    for top in TEXT_DIRS:
        for p in sorted((ROOT / top).rglob("*")):
            if p.is_file() and p.suffix in TEXT_SUFFIXES and "/figs/" not in p.as_posix():
                n, left = rewrite_text(p, m, a.apply)
                if n:
                    files += 1
                    tot += n
    print(f"text: {tot} ids in {files} files")

    # 3. campaigns
    for d in CAMPAIGNS:
        if (ROOT / d / "campaign.json").exists():
            n, unknown = rekey_campaign(d, m, a.apply)
            print(f"{d}: {n} candidates" + (f", {unknown} with UNKNOWN clip -> skipped" if unknown else ""))
    print("applied." if a.apply else "dry run: nothing changed (pass --apply).")


if __name__ == "__main__":
    main()
