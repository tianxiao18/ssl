"""Match every Mongolia AudioMoth recording folder to its Vox deployment row in "Experiment Log.xlsx".

Join key: colony + device id + start time (first file's local time vs logged start, closest within 2 days).

    python scripts/dataset_prep/mongolia_deployments.py
    -> outputs/mongolia_audiomoth/deployments.csv (one row per device folder) + printed period x colony table
"""
import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TZ = 8
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_sheet(path, sheet=1):
    """{row: {col letter: str}} for one worksheet, without openpyxl."""
    z = zipfile.ZipFile(path)
    ss = [''.join(t.text or '' for t in si.iter(NS + "t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")]
    rows = {}
    for r in ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml")).iter(NS + "row"):
        d = {}
        for c in r.findall(NS + "c"):
            v = c.find(NS + "v")
            if v is not None:
                d[re.match(r"[A-Z]+", c.get("r"))[0]] = ss[int(v.text)] if c.get("t") == "s" else v.text
            elif (is_ := c.find(NS + "is")) is not None:
                d[re.match(r"[A-Z]+", c.get("r"))[0]] = ''.join(t.text or '' for t in is_.iter(NS + "t"))
        rows[int(r.get("r"))] = {k: v.strip() for k, v in d.items() if v.strip()}
    return rows


def devices(text):
    """'AM 9 & 14' / 'AMA-B, 2, 4' / 'AM PK1' / 'AM1; 512 TB' -> ['9', '14'] / ['A', 'B', '2', '4'] / ['PK1'] / ['1']."""
    text = re.sub(r"\(.*?\)", "", text.split(";")[0]).replace("AMs", "")
    out = []
    for tok in re.split(r"[,&\s]+", re.sub(r"\bAM\s*", " ", text)):
        if m := re.fullmatch(r"(\w+)-(\w+)", tok):
            a, b = m.groups()
            out += [str(i) for i in range(int(a), int(b) + 1)] if a.isdigit() else [chr(c) for c in range(ord(a), ord(b) + 1)]
        elif tok:
            out.append(tok)
    return out


def log_deployments(path):
    rows = read_sheet(path)
    day, prev_colony, out = None, None, []
    for i in sorted(rows):
        r = rows[i]
        a = r.get("A", "")
        if re.fullmatch(r"4\d{4}(\.0)?", a):
            day = datetime(1899, 12, 30) + timedelta(days=float(a))
            continue
        colony = r.get("C", "").split("\n")[0].replace("Colony ", "").strip() or None
        if r.get("E") != "Vox":
            prev_colony = colony or prev_colony
            continue
        colony = colony or prev_colony  # 06-28/06-29 Vox rows sit under their colony's Behavior row
        try:
            start = day + timedelta(days=float(a))
        except ValueError:
            start = day + timedelta(hours=12)  # ">12:00 PM"
        f = r.get("F", "")
        devs = devices(r.get("G", "") if f == "AMs" else f)
        gain = "med-high" if "gain med-high" in r.get("G", "") else "not logged"
        note = "; ".join(x for x in [r.get("G", "") if f != "AMs" else "", r.get("J", "")] if x and "gain" not in x)
        out += [dict(log_row=i, colony=colony, device=d, start=start, gain=gain, note=note) for d in devs]
    return pd.DataFrame(out)


def data_folders(durations):
    """One row per device folder: colony, device, first file local time, n files, hours."""
    g = defaultdict(list)
    for line in open(durations):
        k, v = line.rstrip("\n").split("\t")
        m = re.search(r"_(\d{8}_\d{6})$", k)
        if not m:
            continue
        parts = k.split("/")
        dev_i = next(j for j, p in enumerate(parts) if re.match(r"(AM|PK)", p))
        g["/".join(parts[:dev_i + 1])].append((datetime.strptime(m[1], "%Y%m%d_%H%M%S") + timedelta(hours=TZ), float(v)))
    out = []
    for folder, files in g.items():
        dev = re.match(r"(?:AM)?(PK\d+|[A-Z]|\d+)_", folder.split("/")[-1])[1]
        out.append(dict(folder=folder, colony=folder.split("/")[0].replace("Colony_", ""), device=dev,
                        first_local=min(t for t, _ in files), files=len(files), hours=sum(d for _, d in files) / 3600))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=ROOT / "Experiment Log.xlsx", type=Path)
    ap.add_argument("--durations", default=ROOT / "outputs/mongolia_audiomoth/durations.txt", type=Path)
    ap.add_argument("--sample", default=ROOT / "outputs/mongolia_audiomoth/all/sample.txt", type=Path)
    ap.add_argument("--out", default=ROOT / "outputs/mongolia_audiomoth/deployments.csv", type=Path)
    a = ap.parse_args()
    log, data = log_deployments(a.log), data_folders(a.durations)
    sample = [l.rstrip("\n") for l in open(a.sample) if l.strip()]
    data["sampled_recs"] = [sum(s.startswith(f + "/") for s in sample) for f in data.folder]

    rows = []
    for r in data.itertuples():
        c = log[(log.colony == r.colony) & (log.device == r.device)]
        dt = (c.start - r.first_local).abs()
        c = c[dt <= timedelta(days=2)]
        m = c.loc[(c.start - r.first_local).abs().idxmin()] if len(c) else None
        rows.append(r._asdict() | (dict(log_row=m.log_row, log_start=m.start, gain=m.gain, note=m.note) if m is not None
                                   else dict(log_row=None, log_start=None, gain=None, note="NOT IN LOG")))
    d = pd.DataFrame(rows).drop(columns="Index").sort_values(["first_local", "colony", "device"])
    d["offset_min"] = ((d.first_local - pd.to_datetime(d.log_start)).dt.total_seconds() / 60).round()
    d.to_csv(a.out, index=False)
    m = d[d.log_row.notna()]
    used = set(zip(m.log_row.astype(int), m.device))
    missing = log[[(r, dv) not in used for r, dv in zip(log.log_row, log.device)]]
    missing = missing.drop_duplicates(["colony", "device", "start"])
    print(d[["first_local", "colony", "device", "files", "hours", "sampled_recs", "log_row", "offset_min", "gain", "note"]]
          .to_string(index=False, max_colwidth=60))
    print("\nlogged deployments with no data folder:")
    print(missing.to_string(index=False, max_colwidth=60))
    print("->", a.out)


if __name__ == "__main__":
    main()
