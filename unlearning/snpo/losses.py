"""SimNPO loss helpers."""

from __future__ import annotations


def get_batch_loss(output, labels):
    import torch
    import torch.nn as nn

    # when passed a ModelOutput or tuple, extract the first item
    if not torch.is_tensor(output):
        if hasattr(output, "logits"):
            output = output.logits
        else:
            output = output[0]

    shifted_labels = labels[..., 1:].contiguous()
    output         = output[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss    = loss_fn(output.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def retain_loss(model, retain_inputs):
    retain_outputs = model(**retain_inputs)
    retain_loss = 0.0
    retain_loss += retain_outputs.loss
    return retain_loss


def compute_forget_loss(
    model,
    inputs,
    return_outputs=False,
    num_items_in_batch=None,
    *,
    beta: float = 3.5,
    delta: float = 0.0,
):
    import torch.nn.functional as F

    input_ids = inputs['input_ids']
    labels = inputs['labels']
    attention_mask = inputs['attention_mask']
    loss_mask = labels != -100
    f_outputs = model(input_ids = input_ids, labels = labels, attention_mask = attention_mask)
    f_loss = get_batch_loss(f_outputs, labels)
    f_loss = f_loss / loss_mask.sum(-1).clamp_min(1) - delta
    f_loss = -F.logsigmoid(beta * f_loss).mean() * 2 / beta
    loss =  f_loss 
    return (loss, f_outputs) if return_outputs else loss


def compute_retain_loss(
    model,
    inputs,
    return_outputs=False,
    num_items_in_batch=None,
    *,
    beta: float = 3.5,
    delta: float = 0.0,
    retain_weight: float = 1.0,
):
    import torch.nn.functional as F

    forget_inputs, retain_inputs = inputs
    f_input_ids, f_labels, f_attention_mask = forget_inputs
    loss_mask = f_labels != -100
    f_outputs = model(input_ids=f_input_ids, labels=f_labels, attention_mask=f_attention_mask)
    f_loss = get_batch_loss(f_outputs, f_labels)
    f_loss = f_loss / loss_mask.sum(-1).clamp_min(1) - delta
    f_loss = -F.logsigmoid(beta * f_loss).mean() * 2 / beta

    r_input_ids, r_labels, r_attention_mask = retain_inputs
    r_outputs = model(input_ids=r_input_ids, labels=r_labels, attention_mask=r_attention_mask)
    r_loss = r_outputs.loss
    loss = f_loss + retain_weight * r_loss
    return (loss, f_outputs) if return_outputs else loss
