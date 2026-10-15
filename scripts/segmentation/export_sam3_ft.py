"""Write weights-only copies of the fine-tuned SAM3 checkpoints (drops optimizer state) for the GitHub release.

    python scripts/segmentation/export_sam3_ft.py   # -> checkpoints/sam3_ft_seg.pt, checkpoints/sam3_ft_box2mask.pt
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "sam3_ft" / "runs"
SOURCES = {"sam3_ft_seg.pt": RUNS / "trace_seg_frozen_v1" / "checkpoints" / "best.pt",
           "sam3_ft_box2mask.pt": RUNS / "trace_frozen_v1" / "checkpoints" / "best.pt"}

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "checkpoints"
out_dir.mkdir(parents=True, exist_ok=True)
for name, src in SOURCES.items():
    sd = torch.load(src, map_location="cpu", weights_only=False)["model"]
    torch.save({"model": sd}, out_dir / name)
    print(f"{src} -> {out_dir / name} ({(out_dir / name).stat().st_size / 1e6:.0f} MB, {len(sd)} tensors)")
