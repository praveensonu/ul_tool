"""Gradient Difference trainer."""

from __future__ import annotations

import copy

from unlearning.base import UnlearnTrainer
from unlearning.gd.losses import (
    compute_forget_loss,
    compute_kl_divergence,
    to_model_inputs,
)


class GradDiffTrainer(UnlearnTrainer):
    """Base for the retain-aware methods: owns the reference model and retain loss."""

    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        **hf_trainer_kwargs,
    ):
        super().__init__(**hf_trainer_kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "NLL":
            return model(**retain_inputs).loss
        if self.retain_loss_type == "KL":
            kl_loss, _ = compute_kl_divergence(model, self.ref_model, retain_inputs)
            return kl_loss
        raise NotImplementedError(
            f"{self.retain_loss_type} not implemented for retain set"
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs, retain_inputs = inputs

        forget_loss, forget_outputs = compute_forget_loss(
            model, to_model_inputs(forget_inputs)
        )
        retain_loss = self.compute_retain_loss(model, to_model_inputs(retain_inputs))

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
