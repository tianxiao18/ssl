"""SAM3 fine-tuning helpers (referenced from configs/sam3_ft/*.yaml)."""
import torch
from torch import nn

from sam3.model_builder import build_sam3_image_model

FROZEN = ("backbone.vision_backbone.trunk.", "backbone.language_backbone.")


def _freeze(module):
    """No grads, forward under no_grad, always eval mode (no dropout / drop-path).

    no_grad is also required: the SAM3.1 ViT MLP (perflib.fused.addmm_act) raises if grad is enabled.
    """
    module.requires_grad_(False)
    forward = module.forward

    def frozen_forward(*args, **kwargs):
        with torch.no_grad():
            return forward(*args, **kwargs)

    module.forward = frozen_forward
    module.train = lambda mode=True: nn.Module.train(module, False)
    module.eval()
    if hasattr(module, "use_act_checkpoint"):
        module.use_act_checkpoint = False


def build_frozen_sam3(frozen=FROZEN, **kw):
    """build_sam3_image_model with the modules at the `frozen` name prefixes frozen."""
    model = build_sam3_image_model(**kw)
    for prefix in frozen:
        _freeze(model.get_submodule(prefix.rstrip(".")))
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM3: {n_train / 1e6:.1f}M trainable, {n_frozen / 1e6:.1f}M frozen ({', '.join(frozen)})")
    return model


def best_val_trainer(**kw):
    """sam3 Trainer that also saves checkpoints/best.pt (+ best.json) whenever val bbox AP improves."""
    import json
    import os

    from sam3.train.trainer import Trainer

    key = "Meters_train/val_gerbil/detection/coco_eval_bbox_AP"

    class BestValTrainer(Trainer):
        best_ap = -1.0

        def _log_meters_and_save_best_ckpts(self, phases):
            out = super()._log_meters_and_save_best_ckpts(phases)
            ap = out.get(key)
            if ap is not None and ap > self.best_ap:
                self.best_ap = float(ap)
                self.save_checkpoint(self.epoch + 1, ["best"])
                if self.distributed_rank == 0:
                    with open(os.path.join(self.checkpoint_conf.save_dir, "best.json"), "w") as f:
                        json.dump({"epoch": self.epoch + 1, "bbox_AP": self.best_ap,
                                   **{k.rsplit("/", 1)[-1]: float(v) for k, v in out.items() if "val_gerbil" in k}}, f)
            return out

    return BestValTrainer(**kw)
