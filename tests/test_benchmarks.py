import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from eval.benchmarks import _accuracy, evaluate_benchmarks
from eval_orchestrator import _collect_model_outputs, _collect_model_outputs_phase
from schemas import BenchmarkScores


class BenchmarkTests(unittest.TestCase):
    def test_only_global_accuracy_is_selected(self):
        result = {"groups": {"mmlu": {"acc,none": 0.63}},
                  "results": {"mmlu_subject": {"acc,none": 0.99}}}
        self.assertEqual(_accuracy(result, "mmlu", group=True), 0.63)
        self.assertEqual(_accuracy({"results": {"gpqa_main_zeroshot": {"acc,none": 0.3, "acc_norm,none": 0.5}}}, "gpqa_main_zeroshot"), 0.3)
        for bad in (None, {}, {"results": {"mmlu": {"acc,none": float("nan")}}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _accuracy(bad, "mmlu", group=True)

    def test_real_hflm_wraps_existing_model_and_returns_two_scores(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")),
            unk_token="<unk>", eos_token="<eos>", pad_token="<eos>",
        )
        model = GPT2LMHeadModel(GPT2Config(vocab_size=2, n_layer=1, n_head=1, n_embd=8, bos_token_id=1, eos_token_id=1))
        seen = []

        def evaluate(**kwargs):
            self.assertIs(kwargs["model"].model, model)
            self.assertIs(kwargs["model"].tokenizer, tokenizer)
            self.assertEqual(os.environ["HF_TOKEN"], "test-token")
            self.assertFalse(kwargs["log_samples"])
            self.assertFalse(kwargs["write_out"])
            self.assertEqual(kwargs["bootstrap_iters"], 0)
            import torch
            logits = kwargs["model"]._model_call(torch.tensor([[0, 1]]))
            self.assertEqual(tuple(logits.shape), (1, 2, 2))
            seen.append((kwargs["tasks"], kwargs["num_fewshot"]))
            if kwargs["tasks"] == ["mmlu"]:
                return {"groups": {"mmlu": {"acc,none": 0.6}}}
            return {"results": {"gpqa_main_zeroshot": {"acc,none": 0.4}}}

        with patch.dict(os.environ, {"HF_TOKEN": "original"}), patch("lm_eval.simple_evaluate", side_effect=evaluate):
            scores = evaluate_benchmarks(model, tokenizer, batch_size=2, hf_key="test-token")
            self.assertEqual(os.environ["HF_TOKEN"], "original")
        self.assertEqual(seen, [(["mmlu"], 5), (["gpqa_main_zeroshot"], 0)])
        self.assertEqual(BenchmarkScores.model_validate(scores).model_dump(), {"mmlu": 0.6, "gpqa": 0.4})

    def test_benchmarks_run_before_model_cleanup_only_in_scoring_phase(self):
        for phase, include_benchmarks in (("metrics", True), ("both", True), ("generation", True), ("metrics", False), ("both", False)):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                model = Mock()
                model.config = SimpleNamespace(use_cache=False)
                tokenizer = Mock()
                source = pd.DataFrame({"question": ["q"], "answer": ["a"], "gen_answer": ["generated"]})
                forget, retain = str(Path(directory) / "f.parquet"), str(Path(directory) / "r.parquet")
                source.to_parquet(forget, index=False)
                source.to_parquet(retain, index=False)
                events = []
                def benchmark(*args, **kwargs):
                    self.assertIs(args[0], model)
                    self.assertIs(args[1], tokenizer)
                    self.assertNotIn("released", events)
                    events.append("benchmarks")
                    return {"mmlu": 0.6, "gpqa": 0.4}
                with patch("eval_orchestrator._prepare_evaluation_dataframe", return_value=source), patch(
                    "eval.eval_utils.compute_model_outputs", return_value=source
                ), patch("eval.benchmarks.evaluate_benchmarks", side_effect=benchmark) as runner, patch(
                    "eval_orchestrator._release_cuda", side_effect=lambda _: events.append("released")
                ):
                    result = _collect_model_outputs_phase(
                        label="Pre-unlearning", loader=lambda: (model, tokenizer),
                        forget_source=source, retain_source=source, max_new_tokens=10,
                        forget_output_path=forget, retain_output_path=retain,
                        torch_module=Mock(), progress_callback=None, phase=phase, include_benchmarks=include_benchmarks,
                    )
                self.assertEqual(runner.call_count, int(include_benchmarks and phase != "generation"))
                self.assertEqual(events[-1], "released")
                self.assertEqual(result is None, not include_benchmarks or phase == "generation")

    def test_multi_gpu_retains_scoring_benchmarks(self):
        expected = {"mmlu": 0.6, "gpqa": 0.4}
        with patch("eval_orchestrator._collect_model_outputs_phase", side_effect=[expected, None]) as phase:
            self.assertEqual(_collect_model_outputs(gpu_ids=[2, 3]), expected)
        self.assertEqual([call.kwargs["phase"] for call in phase.call_args_list], ["metrics", "generation"])
