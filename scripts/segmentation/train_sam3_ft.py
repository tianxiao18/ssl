"""Fine-tune SAM3 with configs kept in this repo (configs/sam3_ft), using SAM3's own trainer unchanged.

Same as `python sam3/sam3/train/train.py -c <config> ...`, except hydra reads the config from configs/sam3_ft
instead of SAM3's packaged configs, so nothing is added to the sam3 checkout.

    python scripts/segmentation/train_sam3_ft.py -c gerbil_trace_ft --use-cluster 0 --num-gpus 1
"""
import sys
from pathlib import Path

from hydra import initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # for vox_tracer.sam3_ft (configs' _target_s)

from sam3.train import train  # noqa: E402
from sam3.train.utils.train_utils import register_omegaconf_resolvers  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "sam3_ft"


def parse_args(argv=None):
    """sam3/train/train.py's arguments."""
    from argparse import ArgumentParser
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True, help="config name in configs/sam3_ft (e.g. gerbil_trace_ft)")
    ap.add_argument("--use-cluster", type=int, default=None, help="0: run locally (in this allocation)")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--qos", default=None)
    ap.add_argument("--num-gpus", type=int, default=None)
    ap.add_argument("--num-nodes", type=int, default=None)
    a = ap.parse_args(argv)
    a.use_cluster = bool(a.use_cluster) if a.use_cluster is not None else None
    return a


if __name__ == "__main__":
    args = parse_args()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        register_omegaconf_resolvers()
        train.main(args)
