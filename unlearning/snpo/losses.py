"""SimNPO losses."""

from __future__ import annotations

import torch.nn.functional as F

from unlearning.gd.losses import compute_batch_nll


def compute_forget_loss(model, inputs, beta, delta):
    loss_mask = inputs["labels"] != -100
    forget_loss, forget_outputs = compute_batch_nll(model, inputs)
    forget_loss = forget_loss / loss_mask.sum(-1).clamp_min(1) - delta
    forget_loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta
    return forget_loss, forget_outputs
