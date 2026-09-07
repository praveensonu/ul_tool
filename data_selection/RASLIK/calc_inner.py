#! /usr/bin/env python3

import torch
import gc
import random
import time
from torch.autograd import grad
import torch
from copy import copy
from .utils import display_progress
from .data_loader import IGNORE_INDEX
import random
from torch.utils.data import default_collate
import torch.nn.functional as F

params = None

def get_params(model, create_if_not_exist=True):
    global params
    if params is not None:
        return params
    if create_if_not_exist == False:
        return None 

    params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            params.append(p)
    return params


def normalize(x):
    return F.normalize(x, p=2, dim=0)


def pad(x):
    D = len(x)
    K = 2**24
    new_D = ((D - 1)//K + 1)*K
    x = F.pad(x, (0, new_D - D), "constant", 0)
    return x


def reshape(x):
    step = 421527552
    n_step = (len(x) - 1)//step + 1
    x = x.reshape((n_step, -1))
    return x


def s_test(z_test, t_test, input_len, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000, need_reshape=True):
    params = get_params(model)

    v = grad_z(z_test, t_test, input_len, model, gpu, need_reshape=False)
    h_estimate = copy(v)

    min_nan_depth = recursion_depth
    has_nan = False
    for i in range(recursion_depth):
        start_time = time.time()
        idx = random.randint(0, len(z_loader) - 1)
        x, t, _, _ = z_loader[idx]
        x = default_collate([x])
        t = default_collate([t])

        if gpu >= 0:
            x, t = x.cuda(gpu), t.cuda(gpu)

        y = model(x)
        y = y.logits
        loss = calc_loss(y, t)

        hv = hvp(loss, params, h_estimate)

        h_estimate_temp = v + (1 - damp) * h_estimate - hv / scale

        if torch.isnan(h_estimate_temp).any() == True:
            print(f"h_estimate has Nan. depth = {i}")
            min_nan_depth = min(min_nan_depth, i)
            has_nan = True
            break

        if has_nan:
            break
        h_estimate = copy(h_estimate_temp)

    h_estimate = pad(h_estimate)
    if need_reshape == True:
        h_estimate = reshape(h_estimate)

    return h_estimate, min_nan_depth


def calc_loss(y, t):
    bs, _, vocab_size = y.shape
    y = y.reshape(-1, vocab_size)
    t = t.reshape(-1)

    loss = torch.nn.functional.cross_entropy(y, t)

    return loss


def grad_z(z, t, input_len, model, gpu=-1, return_words_loss=False, s_test_vec=None, need_reshape=True, use_deepspeed=False):
    z = default_collate([z])
    t = default_collate([t])
    if z.dim() > 2:
        z = torch.squeeze(z, 0)
    if t.dim() > 2:
        t = torch.squeeze(t, 0)

    if gpu >= 0:
        z, t = z.cuda(gpu), t.cuda(gpu)

    y = model(z)
    y = y.logits
    loss = calc_loss(y, t) # batch_size = 1

    params = get_params(model, create_if_not_exist=False)
    if params is not None:
        grad_loss = torch.cat([x.reshape(-1) for x in list(grad(loss, params))])
        model.zero_grad(set_to_none=True)
    else:
        grad_loss = None
        if use_deepspeed == True:
            model.backward(loss)
            grad_loss = torch.cat([normalize(model.optimizer.fp32_partitioned_groups_flat[group_idx].grad.narrow(0, dest_offset, num_elements)) \
                    for group_idx, dest_offset, num_elements in model.optimizer.grad_position.values()])
            model.optimizer.zero_grad()
        else:
            loss.backward()
            grad_loss = torch.cat([normalize(p.grad.reshape(-1)) for p in model.parameters() if p.grad is not None])
            model.zero_grad(set_to_none=True)

    grad_loss = pad(grad_loss)
    if need_reshape == True:
        grad_loss = reshape(grad_loss)

    return grad_loss


def hvp(y, w, v):
    # First backprop
    first_grads = torch.cat([x.reshape(-1) for x in grad(y, w, create_graph=True)])

    # Elementwise products
    elemwise_products = torch.sum(first_grads * v)

    # Second backprop
    return_grads = torch.cat([x.reshape(-1) for x in grad(elemwise_products, w)])

    return return_grads


def grad_z_batch(samples, model, device, pad_token_id):
    """Vectorized per-example gradients; never differentiate a batch-mean loss.

    Preserve grad_z's existing token alignment and per-parameter normalization.
    The leading gradient dimension identifies samples and is kept through RapidGrad.
    """
    from torch.nn.utils.rnn import pad_sequence

    inputs = pad_sequence([sample[0].reshape(-1) for sample in samples], batch_first=True, padding_value=pad_token_id).to(device)
    targets = pad_sequence([sample[1].reshape(-1) for sample in samples], batch_first=True, padding_value=IGNORE_INDEX).to(device)
    lengths = torch.tensor([sample[0].numel() for sample in samples], device=device)
    mask = torch.arange(inputs.shape[1], device=device)[None, :] < lengths[:, None]
    logits = model(inputs, attention_mask=mask, use_cache=False).logits
    counts = (targets != IGNORE_INDEX).sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Gradient samples must contain at least one target token.")
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=IGNORE_INDEX).sum(dim=1) / counts
    cached_params = get_params(model, create_if_not_exist=False)
    parameters = cached_params if cached_params is not None else [p for p in model.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(
        losses, parameters, grad_outputs=torch.eye(len(samples), device=device, dtype=losses.dtype),
        is_grads_batched=True, allow_unused=True,
    )
    flattened = [g.reshape(len(samples), -1) for g in gradients if g is not None]
    if cached_params is None:
        flattened = [F.normalize(g, p=2, dim=1) for g in flattened]
    del gradients, logits, losses
    vectors = torch.cat(flattened, dim=1)
    del flattened
    block = 2**24
    return F.pad(vectors, (0, ((vectors.shape[1] - 1) // block + 1) * block - vectors.shape[1]))
