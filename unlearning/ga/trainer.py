"""Gradient Ascent trainer."""

from __future__ import annotations

from unlearning.base import UnlearnTrainer
from unlearning.ga.losses import compute_forget_loss


class GradAscentTrainer(UnlearnTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        loss, outputs = compute_forget_loss(model, forget_inputs)
        return (loss, outputs) if return_outputs else loss
