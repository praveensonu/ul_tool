"""NPO losses."""

from __future__ import annotations

from unlearning.gd.losses import compute_dpo_loss


def compute_forget_loss(model, ref_model, forget_inputs, beta):
    # NPO is the preference loss with no preferred response
    return compute_dpo_loss(
        model=model,
        ref_model=ref_model,
        win_inputs=None,
        lose_inputs=forget_inputs,
        beta=beta,
    )
