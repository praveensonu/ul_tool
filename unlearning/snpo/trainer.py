"""SimNPO trainers."""

from __future__ import annotations

from unlearning.base import UnlearnTrainer
from unlearning.gd.losses import to_model_inputs
from unlearning.gd.trainer import GradDiffTrainer
from unlearning.snpo.losses import compute_forget_loss


class SimNPOForgetOnlyTrainer(UnlearnTrainer):
    def __init__(self, delta=0.0, beta=4.5, **hf_trainer_kwargs):
        super().__init__(**hf_trainer_kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        forget_loss, forget_outputs = compute_forget_loss(
            model, forget_inputs, beta=self.beta, delta=self.delta
        )
        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss


class SimNPOForgetRetainTrainer(GradDiffTrainer):
    def __init__(self, delta=0.0, beta=4.5, **hf_trainer_kwargs):
        super().__init__(**hf_trainer_kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs, retain_inputs = inputs

        forget_loss, forget_outputs = compute_forget_loss(
            model, to_model_inputs(forget_inputs), beta=self.beta, delta=self.delta
        )
        retain_loss = self.compute_retain_loss(model, to_model_inputs(retain_inputs))

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
