"""Aggregate lm-eval benchmarks against the model already resident in memory."""
from __future__ import annotations

import math
import os
from contextlib import contextmanager


@contextmanager
def _dataset_token(token):
    previous = os.environ.get("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    try:
        yield
    finally:
        if token:
            if previous is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = previous


def _accuracy(result, task, *, group=False):
    if not isinstance(result, dict):
        raise ValueError(f"lm_eval did not return results for {task}.")
    metrics = result.get("groups", {}).get(task) if group else None
    if metrics is None:
        metrics = result.get("results", {}).get(task, {})
    value = metrics.get("acc,none")
    if value is None:
        raise ValueError(f"lm_eval did not return aggregate accuracy for {task}.")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"lm_eval returned invalid accuracy for {task}.")
    return score


def evaluate_benchmarks(model, tokenizer, batch_size=4, hf_key=None, progress_callback=None):
    """Use likelihood-based MMLU (5-shot) and GPQA Main (0-shot), on a 0–1 scale."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    current = {"name": None, "batch": 0}

    class BenchmarkLM(HFLM):
        def _model_call(self, *args, **kwargs):
            # The pinned HF backend calls this for each likelihood-scoring batch.
            # Reporting here also checks the worker's cancellation event.
            current["batch"] += 1
            if progress_callback and current["name"]:
                name = current["name"]
                progress_callback(f"benchmark_{name}_batch", f"Evaluating {name.upper()} batch {current['batch']}.")
            return super()._model_call(*args, **kwargs)

    padding_side = tokenizer.padding_side
    try:
        # Passing the object, rather than its path, avoids loading another model.
        lm = BenchmarkLM(
            pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
            backend="causal", parallelize=False,
        )
        scores = {}
        with _dataset_token(hf_key):
            for name, task, shots in (("mmlu", "mmlu", 5), ("gpqa", "gpqa_main_zeroshot", 0)):
                current.update(name=name, batch=0)
                if progress_callback:
                    progress_callback(f"benchmark_{name}", f"Evaluating {name.upper()} global accuracy.")
                result = simple_evaluate(
                    model=lm, tasks=[task], num_fewshot=shots,
                    log_samples=False, write_out=False, bootstrap_iters=0,
                    apply_chat_template=False, random_seed=0,
                    numpy_random_seed=1234, torch_random_seed=1234,
                    fewshot_random_seed=1234,
                )
                scores[name] = _accuracy(result, task, group=name == "mmlu")
                # Subject-level scores and individual responses are not retained.
                del result
        return scores
    finally:
        tokenizer.padding_side = padding_side
