import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from fastapi import UploadFile

from data_selection import caching
from dataset.dataset_loader import prepare_raslik_dataframe
from schemas import DatasetExtractionResponse


PROMPT_TEMPLATE = "Instruction\n\n{question}\nAnswer:"


class RaslikDatasetPreparationTests(unittest.TestCase):
    def test_adds_ids_prompt_and_generation_columns(self):
        dataframe = pd.DataFrame(
            {
                "question": ["First?", "Second?"],
                "answer": ["One", "Two"],
            }
        )

        prepared = prepare_raslik_dataframe(
            dataframe,
            prompt_template=PROMPT_TEMPLATE,
            dataset_name="full dataset",
        )

        self.assertEqual(prepared["id"].tolist(), ["0", "1"])
        self.assertEqual(
            prepared["prompt"].tolist(),
            [
                "Instruction\n\nFirst?\nAnswer:",
                "Instruction\n\nSecond?\nAnswer:",
            ],
        )
        self.assertEqual(prepared["prompt"].tolist(), prepared["question"].tolist())
        self.assertEqual(prepared["generation"].tolist(), ["One", "Two"])

    def test_preserves_already_normalized_jsonl_columns(self):
        dataframe = pd.DataFrame(
            {
                "id": ["sample-1"],
                "prompt": ["Existing prompt"],
                "generation": ["Existing generation"],
            }
        )

        prepared = prepare_raslik_dataframe(
            dataframe,
            prompt_template=PROMPT_TEMPLATE,
            dataset_name="poison set",
        )

        self.assertEqual(prepared.loc[0, "prompt"], "Existing prompt")
        self.assertEqual(prepared.loc[0, "generation"], "Existing generation")

    def test_rejects_duplicate_ids(self):
        dataframe = pd.DataFrame(
            {
                "id": [1, 1],
                "question": ["First?", "Second?"],
                "answer": ["One", "Two"],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            prepare_raslik_dataframe(
                dataframe,
                prompt_template=PROMPT_TEMPLATE,
                dataset_name="full dataset",
            )


class RaslikGradientCachingTests(unittest.TestCase):
    def test_builds_and_runs_training_then_poison_configs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_counter = 0
            calls = []

            def save_upload(upload):
                nonlocal raw_counter
                raw_counter += 1
                suffix = Path(upload.filename or "").suffix
                path = root / f"raw-{raw_counter}{suffix}"
                path.write_bytes(upload.file.read())
                return str(path)

            def runner(config_path, log_path):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                calls.append((config_path.name, config, log_path.name))

            full_dataset = UploadFile(
                filename="full.csv",
                file=io.BytesIO(b"question,answer\nFull question,Full answer\n"),
            )
            poison_set = UploadFile(
                filename="poison.jsonl",
                file=io.BytesIO(
                    b'{"id":"poison-1","question":"Poison question","answer":"Poison answer"}\n'
                ),
            )

            with (
                patch.object(caching, "PROJECT_ROOT", root),
                patch.object(caching, "GRADIENTS_ROOT", root / "outputs" / "gradients"),
                patch.object(
                    caching,
                    "RASLIK_UPLOAD_ROOT",
                    root / "uploaded_datasets" / "raslik",
                ),
                patch.object(caching, "save_upload_file", side_effect=save_upload),
            ):
                response = caching.cache_uploaded_gradients(
                    full_dataset=full_dataset,
                    poison_set=poison_set,
                    prompt_template=PROMPT_TEMPLATE,
                    experiment_name="My experiment",
                    model_name="local/model",
                    max_length=1024,
                    adaptor_path=None,
                    runner=runner,
                )

            self.assertEqual([call[0] for call in calls], ["training.json", "poison.json"])
            self.assertEqual(response["training_rows"], 1)
            self.assertEqual(response["poison_rows"], 1)
            training_record = json.loads(
                (root / response["training_data_path"])
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(training_record["id"], "0")
            self.assertEqual(training_record["generation"], "Full answer")
            self.assertEqual(
                training_record["prompt"],
                "Instruction\n\nFull question\nAnswer:",
            )
            self.assertTrue(
                response["training_grads_path"].startswith(
                    f"outputs/gradients/{response['experiment_name']}/training"
                )
            )
            self.assertEqual(calls[0][1]["model"]["model_path"], "local/model")
            self.assertEqual(calls[0][1]["model"]["max_length"], 1024)
            self.assertIsNone(calls[0][1]["model"]["lora_path"])
            self.assertTrue(calls[0][1]["influence"]["save_to_grads_path"])
            self.assertEqual(
                calls[1][1]["influence"]["grads_path"],
                response["poison_grads_path"],
            )

    def test_extraction_returns_parquet_paths_ready_for_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_counter = 0

            def save_upload(upload):
                nonlocal raw_counter
                raw_counter += 1
                path = root / f"raw-{raw_counter}{Path(upload.filename or '').suffix}"
                path.write_bytes(upload.file.read())
                return str(path)

            def runner(config_path, _log_path):
                config = json.loads(config_path.read_text(encoding="utf-8"))
                dataset_path = root / config["data"]["train_data_path"]
                grads_path = root / config["influence"]["grads_path"]
                records = [
                    json.loads(line)
                    for line in dataset_path.read_text(encoding="utf-8").splitlines()
                ]
                values = {"a": 2.0, "b": 1.0, "c": -1.0, "d": -2.0, "p": 1.0}
                for record in records:
                    torch.save(
                        torch.tensor([values[record["id"]], 0.0]),
                        grads_path / f"{record['id']}.pt",
                    )

            full_dataset = UploadFile(
                filename="full.csv",
                file=io.BytesIO(
                    b"id,question,answer\n"
                    b"a,A?,A\n"
                    b"b,B?,B\n"
                    b"c,C?,C\n"
                    b"d,D?,D\n"
                ),
            )
            poison_set = UploadFile(
                filename="poison.csv",
                file=io.BytesIO(b"id,question,answer\np,Poison?,Poison\n"),
            )

            with (
                patch.object(caching, "PROJECT_ROOT", root),
                patch.object(caching, "GRADIENTS_ROOT", root / "outputs" / "gradients"),
                patch.object(
                    caching,
                    "RASLIK_UPLOAD_ROOT",
                    root / "uploaded_datasets" / "raslik",
                ),
                patch.object(caching, "save_upload_file", side_effect=save_upload),
            ):
                response = caching.extract_uploaded_datasets(
                    full_dataset=full_dataset,
                    poison_set=poison_set,
                    prompt_template=PROMPT_TEMPLATE,
                    experiment_name="Extraction",
                    model_name="local/model",
                    max_length=512,
                    adaptor_path=None,
                    selection_method="raslik",
                    forget_size=1,
                    retain_size=1,
                    grace_top_n=None,
                    grace_num_clusters=None,
                    runner=runner,
                )

            self.assertEqual(response["forget_rows"], 1)
            self.assertEqual(response["retain_rows"], 1)
            DatasetExtractionResponse.model_validate(response)
            self.assertEqual(
                pd.read_parquet(root / response["forget_set_path"])["id"].tolist(),
                ["a"],
            )
            self.assertEqual(
                pd.read_parquet(root / response["retain_set_path"])["id"].tolist(),
                ["d"],
            )
            self.assertTrue(response["has_retain_set"])
            self.assertFalse(response["gradients_retained"])
            self.assertFalse((root / response["training_grads_path"]).exists())
            self.assertFalse((root / response["poison_grads_path"]).exists())
            self.assertFalse(
                (root / response["selection_metadata_path"]).parent.joinpath(
                    "average_poison_gradient.pt"
                ).exists()
            )

    def test_background_extraction_can_be_cancelled_and_cleans_gradients(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_counter = 0
            runner_started = threading.Event()
            release_runner = threading.Event()

            def save_upload(upload):
                nonlocal raw_counter
                raw_counter += 1
                path = root / f"raw-{raw_counter}{Path(upload.filename or '').suffix}"
                path.write_bytes(upload.file.read())
                return str(path)

            def runner(_config_path, _log_path):
                runner_started.set()
                release_runner.wait(timeout=5)

            full_dataset = UploadFile(
                filename="full.csv",
                file=io.BytesIO(b"id,question,answer\na,A?,A\nb,B?,B\n"),
            )
            poison_set = UploadFile(
                filename="poison.csv",
                file=io.BytesIO(b"id,question,answer\np,Poison?,Poison\n"),
            )
            manager = caching.GradientCacheManager(runner=runner)

            with (
                patch.object(caching, "PROJECT_ROOT", root),
                patch.object(caching, "GRADIENTS_ROOT", root / "outputs" / "gradients"),
                patch.object(
                    caching,
                    "RASLIK_UPLOAD_ROOT",
                    root / "uploaded_datasets" / "raslik",
                ),
                patch.object(caching, "save_upload_file", side_effect=save_upload),
            ):
                started = manager.start_extract(
                    full_dataset=full_dataset,
                    poison_set=poison_set,
                    prompt_template=PROMPT_TEMPLATE,
                    experiment_name="Cancellation",
                    model_name="local/model",
                    max_length=512,
                    adaptor_path=None,
                    selection_method="raslik",
                    forget_size=1,
                    retain_size=1,
                    grace_top_n=None,
                    grace_num_clusters=None,
                    keep_gradients=False,
                )
                self.assertTrue(runner_started.wait(timeout=5))
                self.assertTrue(manager.cancel(started["job_id"]))
                release_runner.set()

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    status = manager.get_status(started["job_id"])
                    if status["status"] == "cancelled":
                        break
                    time.sleep(0.02)

                self.assertEqual(status["status"], "cancelled")
                experiment_root = root / "outputs" / "gradients"
                self.assertFalse(any(experiment_root.glob("*/training")))
                self.assertFalse(any(experiment_root.glob("*/poison")))


class RaslikRouteTests(unittest.TestCase):
    def test_canonical_gradient_cache_route_is_registered(self):
        from main import app

        self.assertIn("/api/dataset/cache-gradients", app.openapi()["paths"])
        self.assertIn("/api/dataset/extract", app.openapi()["paths"])
        self.assertIn("/api/dataset/extract/start", app.openapi()["paths"])
        self.assertIn(
            "/api/dataset/extract/status/{job_id}", app.openapi()["paths"]
        )
        self.assertIn(
            "/api/dataset/extract/cancel/{job_id}", app.openapi()["paths"]
        )


if __name__ == "__main__":
    unittest.main()
