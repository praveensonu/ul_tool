"""DPO losses."""

from __future__ import annotations

from unlearning.gd.losses import compute_dpo_loss


def compute_forget_loss(model, ref_model, forget_inputs, alternate_inputs, beta):
    # the alternate (idk) answer is preferred over the original forget answer
    return compute_dpo_loss(
        model=model,
        ref_model=ref_model,
        win_inputs=alternate_inputs,
        lose_inputs=forget_inputs,
        beta=beta,
    )
