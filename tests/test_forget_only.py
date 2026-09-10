import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

from orchestrator import build_unlearning_run
from unlearning.data_helpers.data_module import IdkForgetOnlyDataset
from unlearning.methods import RETAIN_REQUIRED_METHODS


class Tokenizer:
    eos_token = '~'
    pad_token_id = 0

    def __call__(self, text, max_length=None, **kwargs):
        ids = [ord(char) % 15 + 1 for char in text]
        return {'input_ids': ids[:max_length] if max_length else ids}


class ForgetOnlyTests(unittest.TestCase):
    def test_routing_and_training_with_and_without_retain(self):
        forget = pd.DataFrame({'question': ['Q', 'R'], 'answer': ['aa', 'bb'], 'alternate': ['cc', 'dd']})
        retain = pd.DataFrame({'question': ['S'], 'answer': ['ee']})
        for method in ('npo', 'dpo'):
            for with_retain in (False, True):
                with self.subTest(method=method, retain=with_retain), tempfile.TemporaryDirectory() as output:
                    dataset, collator, trainer_cls, kwargs, run_type = build_unlearning_run(
                        method, forget, retain if with_retain else None, Tokenizer(), 8
                    )
                    suffix = 'forget_retain' if with_retain else 'forget_only'
                    self.assertEqual(run_type, f'{method}_{suffix}')
                    if not with_retain:
                        self.assertEqual(trainer_cls.__name__, f'{method.upper()}ForgetOnlyTrainer')
                        self.assertEqual(kwargs['alpha'], 1.0)
                        self.assertNotIn('retain_loss_type', kwargs)
                    batch = collator([dataset[0], dataset[1]])
                    if method == 'dpo':
                        self.assertEqual(len(batch), 3 if with_retain else 2)
                        self.assertFalse(torch.equal(batch[0][0], batch[1][0]))
                    model = GPT2LMHeadModel(GPT2Config(
                        vocab_size=16, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                        n_positions=8, n_embd=8, n_layer=1, n_head=1,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                    ))
                    trainer = trainer_cls(
                        model=model, args=TrainingArguments(output_dir=output, use_cpu=True, report_to=[]),
                        train_dataset=dataset, data_collator=collator, **kwargs,
                    )
                    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertIsNotNone(outputs)
                    loss.backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in trainer.ref_model.parameters()))

    def test_alternate_column_does_not_require_idk_file(self):
        data = pd.DataFrame({'question': ['Q'], 'answer': ['a'], 'alternate': ['b']})
        with tempfile.TemporaryDirectory() as directory:
            dataset = IdkForgetOnlyDataset(data, Tokenizer(), 8, idk_path=Path(directory) / 'missing')
            self.assertEqual(len(dataset[0]), 2)

    def test_idk_fallback_without_retain(self):
        data = pd.DataFrame({'question': ['Q'], 'answer': ['a']})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'idk.jsonl'
            path.write_text('unknown\n')
            dataset = IdkForgetOnlyDataset(data, Tokenizer(), 12, idk_path=path)
            self.assertEqual(dataset._alternate_answer(0), 'unknown')
            self.assertEqual(len(dataset[0]), 2)

    def test_only_gradient_difference_requires_retain(self):
        self.assertEqual(RETAIN_REQUIRED_METHODS, {'grad_diff'})
        with self.assertRaisesRegex(ValueError, 'requires a retain set'):
            build_unlearning_run('grad_diff', None, None, Tokenizer(), 8)
