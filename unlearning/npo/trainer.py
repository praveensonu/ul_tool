"""NPO trainer."""

from __future__ import annotations

from unlearning.gd.losses import to_model_inputs
from unlearning.gd.trainer import GradDiffTrainer
from unlearning.npo.losses import compute_forget_loss


class NPOTrainer(GradDiffTrainer):
    def __init__(self, beta=0.1, **hf_trainer_kwargs):
        super().__init__(**hf_trainer_kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs, retain_inputs = inputs

        forget_loss, forget_outputs = compute_forget_loss(
            model=model,
            ref_model=self.ref_model,
            forget_inputs=to_model_inputs(forget_inputs),
            beta=self.beta,
        )
        retain_loss = self.compute_retain_loss(model, to_model_inputs(retain_inputs))

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
