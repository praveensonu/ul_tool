import unittest
import time

import pandas as pd
from pydantic import ValidationError

from eval_orchestrator import _prepare_evaluation_dataframe
from eval.eval_utils import compute_fq_scores, compute_mu_scores
from evaluation_process import EvaluationProcessManager
from schemas import EvaluationRequest


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        del add_special_tokens
        return {
            "input_ids": [text.split() for text in texts],
        }


def return_evaluation_result(api_config):
    return {"received": api_config["value"]}


def return_evaluation_result_with_progress(api_config, progress_callback=None):
    if progress_callback:
        progress_callback("calculating", "Calculating test metrics.")
    return {"received": api_config["value"]}


def cancellable_evaluation_result(api_config, progress_callback=None):
    for index in range(100):
        if progress_callback:
            progress_callback("calculating", f"Calculating batch {index}.")
        time.sleep(0.02)
    return {"received": api_config["value"]}


class EvaluationSchemaTests(unittest.TestCase):
    def valid_payload(self):
        return {
            "orchestrator_config": {
                "model": {"model_name": "base-model"},
                "dataset": {
                    "forget_set_path": "forget.parquet",
                    "retain_set_path": "retain.parquet",
                },
                "gpu": {"gpu_id": 0},
            },
            "training_result": {"output_dir": "outputs/run/forget_retain"},
            "embedding_model_name": "local/embedding-model",
        }

    def test_evaluation_request_accepts_embedding_model_input(self):
        request = EvaluationRequest.model_validate(self.valid_payload())

        self.assertEqual(request.max_new_tokens, 256)
        self.assertEqual(request.embedding_model_name, "local/embedding-model")

    def test_evaluation_request_requires_embedding_model_input(self):
        payload = self.valid_payload()
        del payload["embedding_model_name"]

        with self.assertRaisesRegex(ValidationError, "embedding_model_name"):
            EvaluationRequest.model_validate(payload)

    def test_evaluation_request_requires_retain_set(self):
        payload = self.valid_payload()
        payload["orchestrator_config"]["dataset"]["retain_set_path"] = None

        with self.assertRaisesRegex(ValidationError, "retain_set_path"):
            EvaluationRequest.model_validate(payload)


class EvaluationOrchestratorTests(unittest.TestCase):
    def test_prepares_missing_and_invalid_generation_lengths(self):
        dataframe = pd.DataFrame(
            {
                "question": ["q1", "q2", "q3"],
                "answer": ["one two", "one two three four", "one"],
                "num_tokens": [None, -1, 500],
            }
        )

        result = _prepare_evaluation_dataframe(
            dataframe,
            FakeTokenizer(),
            "forget set",
            max_new_tokens=3,
        )

        self.assertEqual(result["num_tokens"].tolist(), [2, 3, 3])
        self.assertTrue(result["question"].map(type).eq(str).all())
        self.assertTrue(result["answer"].map(type).eq(str).all())

    def test_rejects_an_empty_dataset(self):
        dataframe = pd.DataFrame(columns=["question", "answer"])

        with self.assertRaisesRegex(ValueError, "at least one row"):
            _prepare_evaluation_dataframe(
                dataframe,
                FakeTokenizer(),
                "retain set",
                max_new_tokens=10,
            )

    def test_aggregates_scores_from_stored_columns(self):
        forget = pd.DataFrame(
            {
                "conditional_probability": [0.2, 0.4],
                "perplexity": [5.0, 7.0],
                "rouge_l": [0.1, 0.3],
            }
        )
        retain = forget.assign(cosine_similarity=[0.7, 0.9])

        _, forget_quality, forget_ppl, probability, rouge_l = compute_fq_scores(forget)
        _, utility, retain_ppl, _, _, cosine = compute_mu_scores(retain)

        self.assertGreater(forget_quality, 0)
        self.assertGreater(utility, 0)
        self.assertEqual(forget_ppl, 6.0)
        self.assertEqual(retain_ppl, 6.0)
        self.assertAlmostEqual(probability, 0.3)
        self.assertAlmostEqual(rouge_l, 0.2)
        self.assertAlmostEqual(cosine, 0.8)


class EvaluationProcessManagerTests(unittest.TestCase):
    def test_returns_child_process_result(self):
        manager = EvaluationProcessManager(runner=return_evaluation_result)

        result = manager.run({"value": 42})

        self.assertEqual(result, {"received": 42})
        self.assertFalse(manager.is_running)

    def test_exposes_background_progress(self):
        manager = EvaluationProcessManager(
            runner=return_evaluation_result_with_progress
        )
        started = manager.start({"value": 7})
        deadline = time.monotonic() + 15

        while time.monotonic() < deadline:
            status = manager.get_status(started["job_id"])
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(0.02)

        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["result"], {"received": 7})
        self.assertTrue(
            any(event["stage"] == "calculating" for event in status["progress"])
        )

    def test_cooperatively_cancels_a_background_evaluation(self):
        manager = EvaluationProcessManager(runner=cancellable_evaluation_result)
        started = manager.start({"value": 7})
        deadline = time.monotonic() + 15

        while time.monotonic() < deadline:
            status = manager.get_status(started["job_id"])
            if status["progress"]:
                break
            time.sleep(0.02)

        self.assertTrue(manager.cancel(started["job_id"]))
        while time.monotonic() < deadline:
            status = manager.get_status(started["job_id"])
            if status["status"] == "cancelled":
                break
            time.sleep(0.02)

        self.assertEqual(status["status"], "cancelled")
        self.assertFalse(manager.is_running)


class EvaluationRouteTests(unittest.TestCase):
    def test_canonical_evaluation_route_is_registered(self):
        from main import app

        self.assertIn("/api/evaluation/run", app.openapi()["paths"])
        self.assertIn("/api/evaluation/start", app.openapi()["paths"])
        self.assertIn(
            "/api/evaluation/status/{job_id}", app.openapi()["paths"]
        )
        self.assertIn(
            "/api/evaluation/cancel/{job_id}", app.openapi()["paths"]
        )


if __name__ == "__main__":
    unittest.main()
