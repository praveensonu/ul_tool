from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from scipy.stats import hmean
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer


def eval_rouge_recall(generated_output: str, ground_truth: str):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(ground_truth, generated_output)
    return scores["rouge1"].recall, scores["rougeL"].recall


def eval_cosine_similarity_batched(
    generated_outputs: List[str],
    ground_truths: List[str],
    embedding_model: Any,
    batch_size: int = 32,
):
    """Calculate paired cosine similarity without an NxN similarity matrix."""

    with torch.no_grad():
        generated_embeddings = embedding_model.encode(
            generated_outputs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        truth_embeddings = embedding_model.encode(
            ground_truths,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        paired_scores = F.cosine_similarity(
            generated_embeddings, truth_embeddings
        )
        return torch.clamp(paired_scores, min=0).tolist()


@torch.no_grad()
def get_probs_ppl_batch(questions, answers, model, tokenizer, device):
    """Return per-row exp(-mean answer NLL) and perplexity, excluding padding."""
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        question_lengths = [len(ids) for ids in tokenizer(
            questions, add_special_tokens=False
        )["input_ids"]]
        encoded = tokenizer(
            [q + a + (tokenizer.eos_token or "") for q, a in zip(questions, answers)],
            add_special_tokens=False, padding=True, return_tensors="pt",
        ).to(device)
    finally:
        tokenizer.padding_side = previous_side
    labels = encoded["input_ids"].clone()
    for row, length in enumerate(question_lengths):
        labels[row, :length] = -100
    labels[encoded["attention_mask"] == 0] = -100
    output = model(**encoded, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    targets = labels[:, 1:].to(logits.device)
    counts = (targets != -100).sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each evaluation sample must contain at least one scored answer token.")
    losses = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none", ignore_index=-100
    ).sum(dim=1) / counts
    return torch.exp(-losses).tolist(), torch.exp(losses).tolist()


@torch.no_grad()
def generate_outputs_batch(questions, model, tokenizer, device, token_limits):
    """Left-pad decoder-only prompts and retain each row's requested token cap."""
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            questions, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(device)
    finally:
        tokenizer.padding_side = previous_side
    output = model.generate(
        **inputs, max_new_tokens=max(token_limits), do_sample=False,
        return_dict_in_generate=False, pad_token_id=tokenizer.pad_token_id,
    )
    prompt_width = inputs["input_ids"].shape[1]
    return [tokenizer.decode(
        tokens[prompt_width:prompt_width + limit], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ) for tokens, limit in zip(output, token_limits)]


def get_probs_ppl(question, answer, model, tokenizer, device):
    probabilities, perplexities = get_probs_ppl_batch(
        [question], [answer], model, tokenizer, device
    )
    return probabilities[0], perplexities[0]


def generate_outputs(question, model, tokenizer, device, max_new_tokens=50):
    return generate_outputs_batch(
        [question], model, tokenizer, device, [int(max_new_tokens)]
    )[0]


def compute_model_outputs(
    df: pd.DataFrame, model, tokenizer, device, dataset_name: str,
    phase: str = "both", batch_size: int = 4, progress_callback=None,
) -> pd.DataFrame:
    """Batch model calls while retaining per-example metrics and input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if phase not in {"both", "metrics", "generation"}:
        raise ValueError(f"Unknown evaluation phase: {phase}")
    result = df.copy()
    generated_answers, probabilities, perplexities = [], [], []
    for start in tqdm(range(0, len(result), batch_size), desc=f"Scoring {dataset_name}"):
        batch = result.iloc[start:start + batch_size]
        questions = batch["question"].astype(str).tolist()
        if progress_callback:
            progress_callback("evaluating_batch", f"{dataset_name}: rows {start + 1}–{start + len(batch)} of {len(result)} ({phase}).")
        if phase != "generation":
            probs, ppls = get_probs_ppl_batch(
                questions, batch["answer"].astype(str).tolist(), model, tokenizer, device
            )
            probabilities.extend(probs)
            perplexities.extend(ppls)
        if phase != "metrics":
            generated_answers.extend(generate_outputs_batch(
                questions, model, tokenizer, device, batch["num_tokens"].astype(int).tolist()
            ))
    if phase != "metrics":
        result["gen_answer"] = generated_answers
    if phase != "generation":
        result["conditional_probability"] = probabilities
        result["perplexity"] = perplexities
    return result


def save_evaluation_jsonl(result: dict, experiment_name: str, results_root=None) -> str:
    """Append one complete evaluation response per line, including repeat runs."""
    import fcntl
    import json
    import os
    import re
    from pathlib import Path

    root = Path(results_root) if results_root is not None else Path(__file__).resolve().parents[1] / "outputs" / "results"
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_name.strip()).strip("._-")[:120] or "experiment"
    path = root / f"{name}.jsonl"
    result["experiment_name"] = experiment_name
    result["output_files"]["results_jsonl_path"] = str(path)
    line = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
    root.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.write(line)
        output.flush()
        os.fsync(output.fileno())
    return str(path)


def compute_rouge_l_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-example ROUGE-L after generations have been persisted."""

    result = df.copy()
    result["rouge_l"] = [
        eval_rouge_recall(generated, truth)[1]
        for generated, truth in zip(
            result["gen_answer"].astype(str),
            result["answer"].astype(str),
        )
    ]
    return result


def compute_cosine_similarity_scores(
    df: pd.DataFrame,
    embedding_model: Any,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Add semantic similarity to a retain-set result table."""

    result = df.copy()
    result["cosine_similarity"] = eval_cosine_similarity_batched(
        result["gen_answer"].astype(str).tolist(),
        result["answer"].astype(str).tolist(),
        embedding_model,
        batch_size=batch_size,
    )
    return result


def compute_fq_scores(df: pd.DataFrame):
    """Aggregate forget quality from previously stored model and ROUGE data."""

    mean_probability = float(df["conditional_probability"].mean())
    mean_rouge_l = float(df["rouge_l"].mean())
    component_scores = np.array(
        [1.0 - mean_probability, 1.0 - mean_rouge_l]
    )
    forget_quality = float(hmean(component_scores))
    average_perplexity = float(df["perplexity"].mean())
    return (
        component_scores,
        forget_quality,
        average_perplexity,
        mean_probability,
        mean_rouge_l,
    )


def compute_mu_scores(df: pd.DataFrame):
    """Aggregate model utility from stored retain-set result columns."""

    mean_probability = float(df["conditional_probability"].mean())
    mean_rouge_l = float(df["rouge_l"].mean())
    mean_cosine_similarity = float(df["cosine_similarity"].mean())
    component_scores = np.array(
        [mean_probability, mean_rouge_l, mean_cosine_similarity]
    )
    model_utility = float(hmean(component_scores))
    average_perplexity = float(df["perplexity"].mean())
    return (
        component_scores,
        model_utility,
        average_perplexity,
        mean_probability,
        mean_rouge_l,
        mean_cosine_similarity,
    )
