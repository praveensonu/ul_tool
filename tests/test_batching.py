import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

from eval.eval_utils import (
    compute_model_outputs, get_probs_ppl, get_probs_ppl_batch,
    generate_outputs_batch, save_evaluation_jsonl,
)
from data_selection.RASLIK import calc_inner
from data_selection.RASLIK.RapidGrad import RapidGrad


class Encoded(dict):
    def to(self, device):
        return Encoded({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    padding_side = "right"
    eos_token = "~"
    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[1 + ord(c) % 7 for c in text] for text in texts]
        if return_tensors is None:
            return {"input_ids": rows}
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            size = width - len(row)
            ids.append(([0] * size + row) if self.padding_side == "left" else (row + [0] * size))
            masks.append(([0] * size + [1] * len(row)) if self.padding_side == "left" else ([1] * len(row) + [0] * size))
        return Encoded(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))

    def decode(self, tokens, **kwargs):
        return ",".join(map(str, tokens.tolist()))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.embedding = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 8)
        self.batch_sizes = []

    def forward(self, input_ids, **kwargs):
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.linear(self.embedding(input_ids)))

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.generation_inputs = input_ids
        return torch.cat([input_ids, torch.full((len(input_ids), max_new_tokens), 3)], dim=1)


class EvaluationBatchTests(unittest.TestCase):
    def test_padding_and_per_row_metric_equivalence(self):
        model, tokenizer = TinyModel(), Tokenizer()
        questions, answers = ["ab", "abcde"], ["cdef", "a"]
        individual = [get_probs_ppl(q, a, model, tokenizer, "cpu") for q, a in zip(questions, answers)]
        probabilities, perplexities = get_probs_ppl_batch(questions, answers, model, tokenizer, "cpu")
        torch.testing.assert_close(torch.tensor(probabilities), torch.tensor([r[0] for r in individual]))
        torch.testing.assert_close(torch.tensor(perplexities), torch.tensor([r[1] for r in individual]))
        self.assertEqual(model.batch_sizes, [1, 1, 2])
        self.assertEqual(tokenizer.padding_side, "right")

    def test_left_padded_generation_keeps_individual_caps(self):
        model, tokenizer = TinyModel(), Tokenizer()
        outputs = generate_outputs_batch(["ab", "abcde"], model, tokenizer, "cpu", [1, 3])
        self.assertEqual(outputs, ["3", "3,3,3"])
        self.assertEqual(model.generation_inputs[0, :3].tolist(), [0, 0, 0])
        self.assertEqual(tokenizer.padding_side, "right")

    def test_final_partial_batch_and_row_order(self):
        model = TinyModel()
        data = pd.DataFrame({"question": ["ab", "cde", "f"], "answer": ["c", "f", "de"], "num_tokens": [1, 3, 2]}, index=[8, 2, 5])
        result = compute_model_outputs(data, model, Tokenizer(), "cpu", "test", batch_size=2)
        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual(result.index.tolist(), [8, 2, 5])
        self.assertEqual(result.gen_answer.tolist(), ["3", "3,3,3", "3,3"])

    def test_jsonl_append_and_safe_experiment_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            result = {"status": "success", "output_files": {}, "score": 0.7}
            path = save_evaluation_jsonl(result, "../../My experiment", directory)
            save_evaluation_jsonl({**result, "score": 0.8}, "../../My experiment", directory)
            self.assertEqual(Path(path).parent, Path(directory))
            self.assertEqual(Path(path).name, "My-experiment.jsonl")
            records = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual([row["score"] for row in records], [0.7, 0.8])
            self.assertEqual(records[0]["output_files"]["results_jsonl_path"], path)


class GradientBatchTests(unittest.TestCase):
    def test_per_example_gradients_match_individual_normalization(self):
        model = TinyModel()
        samples = [
            (torch.tensor([1, 2, 3]), torch.tensor([-100, 2, 3]), 1, "a"),
            (torch.tensor([4, 5]), torch.tensor([-100, 5]), 1, "b"),
        ]
        # Avoid allocating the production 2**24 padding block in this tiny test.
        with patch.object(calc_inner, "params", None), patch.object(calc_inner.F, "pad", side_effect=lambda value, *args, **kwargs: value):
            individual = torch.stack([calc_inner.grad_z(z, t, length, model, need_reshape=False) for z, t, length, _ in samples])
            batched = calc_inner.grad_z_batch(samples, model, "cpu", 0)
        torch.testing.assert_close(batched, individual, atol=1e-6, rtol=1e-5)
        self.assertEqual(model.batch_sizes, [1, 1, 2])

    def test_rapidgrad_projects_each_row_identically(self):
        compressor = RapidGrad.__new__(RapidGrad)
        compressor.is_init = True
        compressor.D = 8
        compressor.perm_dim_list = [2, 4]
        compressor.perm_mat_list = [[1, 0], [2, 0, 3, 1]]
        compressor.random_mat = torch.tensor([1, -1, 1, -1, 1, 1, -1, -1])
        vectors = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        batched = compressor(vectors, 2)
        # Reference the original scalar permutation/reduction algorithm.
        expected = []
        for vector in vectors:
            vector = vector.reshape(2, -1)[[1, 0], :]
            vector = vector.reshape(-1, 4)[:, [2, 0, 3, 1]]
            vector = vector.reshape(-1) * compressor.random_mat
            expected.append(vector.reshape(-1, 4).sum(dim=1))
        torch.testing.assert_close(batched, torch.stack(expected))
        torch.testing.assert_close(compressor(vectors[0], 2), expected[0])


if __name__ == "__main__":
    unittest.main()
