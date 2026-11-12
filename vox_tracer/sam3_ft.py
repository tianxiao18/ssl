"""SAM3 fine-tuning helpers (referenced from configs/sam3_ft/*.yaml)."""
import logging

import torch
from torch import nn

from sam3.model_builder import build_sam3_image_model

FROZEN = ("backbone.vision_backbone.trunk.", "backbone.language_backbone.")


def _patch_focal_loss():
    """Send gamma < 1 focal losses (the presence loss, gamma 0) through sam3's PyTorch branch.

    sam3's Triton backward computes pow(1 - p_t, gamma - 1) * (1 - p_t), which is inf * 0 = NaN once
    |logit| > ~17 makes 1 - p_t round to 0.
    """
    from sam3.train.loss import loss_fns

    orig = loss_fns.sigmoid_focal_loss
    if getattr(orig, "nan_safe", False):
        return

    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2, loss_on_multimask=False, reduce=True,
                           triton=True):
        return orig(inputs, targets, num_boxes, alpha, gamma, loss_on_multimask, reduce, triton and gamma >= 1)

    sigmoid_focal_loss.nan_safe = True
    loss_fns.sigmoid_focal_loss = sigmoid_focal_loss


class FixedScaler(torch.amp.GradScaler):
    """GradScaler held at scale 1 (bf16 needs no loss scaling) that still skips steps with non-finite grads.

    The stock scaler halves on every skipped step; at 2**-128, 1/scale overflows to inf, the bad step is no
    longer caught and the weights go NaN.
    """

    def __init__(self, device="cuda"):
        super().__init__(device, init_scale=1.0)
        self.skipped = 0

    def update(self, new_scale=None):
        if self._enabled and any(bool(v.item()) for s in self._per_optimizer_states.values()
                                 for v in s["found_inf_per_device"].values()):
            self.skipped += 1
            logging.warning(f"non-finite grads, optimizer step skipped ({self.skipped} so far)")
        super().update(1.0)

    def load_state_dict(self, state_dict):
        super().load_state_dict({**state_dict, "scale": 1.0})


def nan_fix(trainer):
    """Both fixes on a sam3 Trainer whose components are set up."""
    _patch_focal_loss()
    amp = trainer.optim_conf.amp if trainer.optim_conf else None
    if amp is not None and amp.enabled and amp.amp_dtype == "bfloat16":
        trainer.scaler = FixedScaler(trainer.device)
    logging.info(f"NaN fix: focal loss patched, scaler {type(trainer.scaler).__name__}")


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


def _match_in_eval(model):
    """Run the target matcher in eval mode too, so the val loss gets out["indices"].

    sam3 only matches when training or num_interactive_steps_val > 0 (sam3_image.py forward_grounding).
    """
    grounding = model.forward_grounding

    def forward_grounding(backbone_out, find_input, find_target, *args, **kwargs):
        out = grounding(backbone_out, find_input, find_target, *args, **kwargs)
        if not model.training and find_target is not None and "indices" not in out:
            model._compute_matching(out, model.back_convert(find_target))
        return out

    model.forward_grounding = forward_grounding


def build_frozen_sam3(frozen=FROZEN, **kw):
    """build_sam3_image_model with the modules at the `frozen` name prefixes frozen."""
    model = build_sam3_image_model(**kw)
    for prefix in frozen:
        _freeze(model.get_submodule(prefix.rstrip(".")))
    _match_in_eval(model)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM3: {n_train / 1e6:.1f}M trainable, {n_frozen / 1e6:.1f}M frozen ({', '.join(frozen)})")
    return model


class SafeLoss(nn.Module):
    """Wraps a loss for the val phase: returns 0 (and logs once) if it raises, so a val-loss bug can't kill a run."""

    def __init__(self, loss):
        super().__init__()
        self.loss, self.failed = loss, False

    def forward(self, *args, **kwargs):
        try:
            return self.loss(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if not self.failed:
                import logging
                logging.exception(f"val loss failed, logging 0 from now on: {e}")
            self.failed = True
            return torch.zeros(())


def best_val_trainer(max_nan_skips=50, **kw):
    """sam3 Trainer that also
    - saves checkpoints/best.pt (+ best.json) whenever val bbox AP improves,
    - keeps each epoch's val predictions as dumps/val/epoch_<N>.json (sam3 overwrites one file),
    - skips a train batch with a non-finite loss instead of stopping (raises after max_nan_skips in a row),
    - applies nan_fix (the gold-run NaN: presence focal loss backward + bf16 loss-scale collapse).
    """
    import json
    import os
    import shutil

    from sam3.train.trainer import Trainer

    key = "Meters_train/val_gerbil/detection/coco_eval_bbox_AP"

    class BestValTrainer(Trainer):
        best_ap = -1.0
        nan_run = 0
        nan_total = 0

        def _setup_components(self):
            super()._setup_components()
            nan_fix(self)

        def _run_step(self, batch, phase, loss_mts, extra_loss_mts, raise_on_error=True):
            if getattr(self, "_step_epoch", None) != self.epoch:
                self._step_epoch, self._step_i = self.epoch, -1
            self._step_i += 1  # index into this epoch's sampler order (batch size 1 = dataset position)
            before = sum(m.count for m in loss_mts.values())
            super()._run_step(batch, phase, loss_mts, extra_loss_mts, raise_on_error=False)
            if sum(m.count for m in loss_mts.values()) == before:  # loss was non-finite, no backward ran
                self.nan_run += 1
                self.nan_total += 1
                import logging
                logging.error(f"non-finite loss skipped: epoch {self.epoch} step {self._step_i}")
                if self.nan_run > max_nan_skips:
                    raise FloatingPointError(f"{self.nan_run} non-finite losses in a row")
            else:
                self.nan_run = 0

        def _log_meters_and_save_best_ckpts(self, phases):
            out = super()._log_meters_and_save_best_ckpts(phases)
            ap = out.get(key)
            if ap is None:
                return out
            run_dir = os.path.dirname(self.checkpoint_conf.save_dir)
            if self.distributed_rank == 0:
                dump = os.path.join(run_dir, "dumps", "val", "coco_predictions_bbox.json")
                if os.path.exists(dump):
                    shutil.copy(dump, os.path.join(os.path.dirname(dump), f"epoch_{self.epoch + 1}.json"))
                with open(os.path.join(run_dir, "logs", "nan_skips.json"), "a") as f:
                    f.write(json.dumps({"epoch": self.epoch + 1, "nan_skipped_total": self.nan_total}) + "\n")
            if ap > self.best_ap:
                self.best_ap = float(ap)
                self.save_checkpoint(self.epoch + 1, ["best"])
                if self.distributed_rank == 0:
                    with open(os.path.join(self.checkpoint_conf.save_dir, "best.json"), "w") as f:
                        json.dump({"epoch": self.epoch + 1, "bbox_AP": self.best_ap,
                                   **{k.rsplit("/", 1)[-1]: float(v) for k, v in out.items() if "val_gerbil" in k}}, f)
            return out

    return BestValTrainer(**kw)
