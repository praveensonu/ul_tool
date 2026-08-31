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
def get_probs_ppl(
    question: str,
    answer: str,
    model,
    tokenizer: PreTrainedTokenizer,
    device,
):
    eos_token = tokenizer.eos_token or ""
    full_text = question + answer + eos_token
    question_encoded = tokenizer(
        question,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    question_length = question_encoded["input_ids"].size(1)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    input_ids = encoded["input_ids"]
    labels = input_ids.clone()
    labels[0, :question_length] = -100
    output = model(
        input_ids,
        attention_mask=encoded["attention_mask"],
        labels=labels,
    )
    loss = output.loss
    conditional_probability = torch.exp(-loss).item()
    perplexity = torch.exp(loss).item()
    return conditional_probability, perplexity


@torch.no_grad()
def generate_outputs(
    question: str,
    model,
    tokenizer,
    device,
    max_new_tokens: int = 50,
):
    inputs = tokenizer(
        question,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        return_dict_in_generate=False,
    )
    generated_ids = output[0][inputs["input_ids"].size(1) :]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def compute_model_outputs(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    dataset_name: str,
) -> pd.DataFrame:
    """Store language-model-only metrics and generations on a dataset copy."""

    result = df.copy()
    generated_answers = []
    probabilities = []
    perplexities = []

    for _, row in tqdm(
        result.iterrows(),
        total=len(result),
        desc=f"Scoring {dataset_name}",
    ):
        probability, perplexity = get_probs_ppl(
            row["question"],
            row["answer"],
            model,
            tokenizer,
            device,
        )
        generated_answer = generate_outputs(
            row["question"],
            model,
            tokenizer,
            device,
            max_new_tokens=int(row["num_tokens"]),
        )
        generated_answers.append(generated_answer)
        probabilities.append(probability)
        perplexities.append(perplexity)

    result["gen_answer"] = generated_answers
    result["conditional_probability"] = probabilities
    result["perplexity"] = perplexities
    return result


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
