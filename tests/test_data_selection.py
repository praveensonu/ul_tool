import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_selection.selection import (
    select_and_export_datasets,
    select_grace_ids,
    select_raslik_ids,
)


class RaslikSelectionTests(unittest.TestCase):
    def test_selects_highest_and_lowest_average_inner_products(self):
        training_ids = ["a", "b", "c", "d"]
        training = np.asarray(
            [[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]],
            dtype=np.float32,
        )
        poison = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)

        forget, retain, metadata = select_raslik_ids(
            training_ids,
            training,
            poison,
            forget_size=1,
            retain_size=1,
        )

        self.assertEqual(forget, ["a"])
        self.assertEqual(retain, ["d"])
        self.assertEqual(metadata["average_inner_product_scores"]["a"], 4.0)
        self.assertEqual(metadata["average_inner_product_scores"]["d"], -4.0)

    def test_rejects_overlapping_requested_sizes(self):
        with self.assertRaisesRegex(ValueError, r"forget_size \+ retain_size"):
            select_raslik_ids(
                ["a", "b"],
                np.eye(2, dtype=np.float32),
                np.ones((1, 2), dtype=np.float32),
                forget_size=2,
                retain_size=1,
            )


class GraceSelectionTests(unittest.TestCase):
    def test_uses_nnomp_for_forget_and_balanced_cluster_omp_for_retain(self):
        training_ids = [f"t{index}" for index in range(8)]
        training = np.asarray(
            [
                [4.0, 0.0],
                [3.0, 1.0],
                [2.0, -1.0],
                [0.0, 4.0],
                [0.0, 3.0],
                [0.0, -3.0],
                [0.0, -4.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        poison = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)

        forget, retain, metadata = select_grace_ids(
            training_ids,
            training,
            poison,
            forget_size=2,
            retain_size=4,
            top_n=3,
            num_clusters=2,
        )

        self.assertEqual(len(forget), 2)
        self.assertEqual(len(retain), 4)
        self.assertTrue(set(forget).issubset(set(metadata["top_n_ids"])))
        self.assertTrue(set(retain).isdisjoint(set(metadata["top_n_ids"])))
        self.assertEqual(
            sorted(metadata["cluster_selection_counts"].values()),
            [2, 2],
        )

    def test_validates_grace_pool_and_cluster_sizes(self):
        training = np.eye(4, dtype=np.float32)
        poison = np.ones((1, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "top_n must be at least"):
            select_grace_ids(
                ["a", "b", "c", "d"],
                training,
                poison,
                forget_size=2,
                retain_size=1,
                top_n=1,
                num_clusters=1,
            )


class SelectionExportTests(unittest.TestCase):
    def test_exports_ranked_rows_as_parquet(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_grads = root / "training"
            poison_grads = root / "poison"
            training_grads.mkdir()
            poison_grads.mkdir()

            gradients = {
                "a": [2.0, 0.0],
                "b": [1.0, 0.0],
                "c": [-1.0, 0.0],
                "d": [-2.0, 0.0],
            }
            for sample_id, gradient in gradients.items():
                torch.save(torch.tensor(gradient), training_grads / f"{sample_id}.pt")
            torch.save(torch.tensor([1.0, 0.0]), poison_grads / "p0.pt")

            training_data_path = root / "training.jsonl"
            records = [
                {
                    "id": sample_id,
                    "question": f"Question {sample_id}",
                    "answer": f"Answer {sample_id}",
                    "prompt": f"Prompt {sample_id}",
                    "generation": f"Answer {sample_id}",
                }
                for sample_id in gradients
            ]
            training_data_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            result = select_and_export_datasets(
                method="raslik",
                training_grads_path=training_grads,
                poison_grads_path=poison_grads,
                training_data_path=training_data_path,
                output_dir=root / "selected" / "raslik",
                forget_size=1,
                retain_size=1,
            )

            forget = pd.read_parquet(result["forget_path"])
            retain = pd.read_parquet(result["retain_path"])
            self.assertEqual(forget["id"].tolist(), ["a"])
            self.assertEqual(retain["id"].tolist(), ["d"])
            self.assertTrue(result["metadata_path"].is_file())


if __name__ == "__main__":
    unittest.main()
