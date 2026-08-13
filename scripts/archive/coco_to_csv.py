"""Convert SAM3 COCO JSON annotation file(s) to the vox CSV timestamp format."""

import argparse
import json
from pathlib import Path

import pandas as pd


def coco_to_df(coco_path: Path) -> pd.DataFrame:
    with open(coco_path) as f:
        data = json.load(f)

    img_lookup = {img["id"]: img for img in data["images"]}

    # infer channel from filename (e.g. coco_ch_35.json -> 35)
    stem = coco_path.stem  # e.g. "coco_ch_35"
    try:
        channel = float(stem.split("_ch_")[-1])
    except ValueError:
        channel = float("nan")

    rows = []
    for ann in data["annotations"]:
        img = img_lookup[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        duration = img["window_end_sec"] - img["window_start_sec"]
        start = img["window_start_sec"] + (x / img["width"]) * duration
        stop = img["window_start_sec"] + ((x + w) / img["width"]) * duration
        rows.append(
            {
                "name": "vox",
                "start_seconds": start,
                "stop_seconds": stop,
                "channel": channel,
                "label": "",
                "method": "",
            }
        )

    df = pd.DataFrame(rows, columns=["name", "start_seconds", "stop_seconds", "channel", "label", "method"])
    df = df.sort_values("start_seconds").reset_index(drop=True)
    return df


def process_one(coco_path: Path, output_csv: Path) -> int:
    df = coco_to_df(coco_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Convert SAM3 COCO JSON to vox CSV timestamps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Root directory to scan for coco_ch_*.json files (writes CSV alongside each JSON)")
    group.add_argument("--input", type=Path, help="Single COCO JSON input file")
    parser.add_argument("--output", type=Path, help="Output CSV path (required with --input)")
    args = parser.parse_args()

    if args.dir:
        json_files = sorted(args.dir.rglob("coco_ch_*.json"))
        if not json_files:
            print(f"No coco_ch_*.json files found under {args.dir}")
            return
        total_ann = 0
        for path in json_files:
            out = path.with_suffix(".csv")
            n = process_one(path, out)
            total_ann += n
            print(f"  {out.relative_to(args.dir)}  ({n} annotations)")
        print(f"\nDone: {len(json_files)} files, {total_ann} total annotations")
    else:
        if args.output is None:
            parser.error("--output is required when using --input")
        n = process_one(args.input, args.output)
        print(f"Wrote {n} annotations to {args.output}")


if __name__ == "__main__":
    main()
