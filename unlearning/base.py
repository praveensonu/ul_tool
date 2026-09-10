"""Shared base for the unlearning trainers."""

from __future__ import annotations

from types import MethodType

from transformers import Trainer


class UnlearnTrainer(Trainer):
    """A `Trainer` that behaves the same whichever batch format a method uses.

    Clearing `model_accepts_loss_kwargs` keeps the forget-only and forget/retain batch
    formats normalising by `gradient_accumulation_steps` identically, and
    `prediction_step` routes back to the stock loss because an eval batch is a plain
    language-modelling batch rather than a forget/retain structure.
    """

    def __init__(self, gamma=1.0, alpha=1.0, **hf_trainer_kwargs):
        super().__init__(**hf_trainer_kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.model_accepts_loss_kwargs = False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        self.compute_loss = MethodType(Trainer.compute_loss, self)
        try:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
        finally:
            del self.compute_loss
