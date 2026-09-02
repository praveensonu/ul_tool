"""Gradient Ascent losses."""

from __future__ import annotations


def compute_forget_loss(model, inputs):
    outputs = model(**inputs)
    return -outputs.loss, outputs
