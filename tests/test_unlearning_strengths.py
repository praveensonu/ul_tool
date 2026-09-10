import unittest
from unittest.mock import patch

import torch
from pydantic import ValidationError

from config.training_config import build_orchestrator_config
from schemas import FinalTrainingConfigRequest, GeneralHyperParams
from orchestrator import build_unlearning_run
from tests.test_forget_only import Tokenizer
import pandas as pd


class StrengthTests(unittest.TestCase):
    def test_api_config_preserves_strengths_and_defaults(self):
        payload = dict(
            model_name='test', method='full', gpu_ids=[0], forget_set_path='forget.parquet',
            hyperparams=dict(
                general=dict(max_steps=1, learning_rate=0.001, context_length=8),
                optimization=dict(batch_size=1, grad_accum=1, weight_decay=0),
                schedule=dict(save_steps=1), memory={},
            ),
        )
        for strengths in ({}, {'gamma': 2.5, 'alpha': 0.3}):
            payload['hyperparams']['general'].update(strengths)
            config = build_orchestrator_config(FinalTrainingConfigRequest(**payload))
            self.assertEqual(config['hyperparams']['general']['gamma'], strengths.get('gamma', 1.0))
            self.assertEqual(config['hyperparams']['general']['alpha'], strengths.get('alpha', 1.0))

    def test_strength_validation(self):
        for field in ('gamma', 'alpha'):
            for value in (-1, float('inf'), float('nan')):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    GeneralHyperParams(max_steps=1, learning_rate=0.001, context_length=8, **{field: value})

    def test_shared_strengths_scale_every_objective(self):
        data = pd.DataFrame({'question': ['Q'], 'answer': ['a'], 'alternate': ['b']})
        modules = {'grad_ascent': 'ga', 'grad_diff': 'gd', 'simnpo': 'snpo', 'npo': 'npo', 'dpo': 'dpo'}
        for method, module in modules.items():
            for retained in (False, True):
                if method == 'grad_diff' and not retained:
                    continue
                with self.subTest(method=method, retained=retained):
                    dataset, collator, cls, kwargs, run_type = build_unlearning_run(
                        method, data, data if retained else None, Tokenizer(), 8, gamma=2.5, alpha=0.3
                    )
                    self.assertEqual(kwargs['gamma'], 2.5)
                    self.assertEqual(kwargs['alpha'], 0.3)
                    trainer = object.__new__(cls)
                    trainer.gamma, trainer.alpha = kwargs['gamma'], kwargs['alpha']
                    trainer.beta, trainer.delta, trainer.ref_model = 0.1, 0, None
                    trainer.compute_retain_loss = lambda *args: torch.tensor(5.0)
                    with patch(f'unlearning.{module}.trainer.compute_forget_loss', return_value=(torch.tensor(3.0), 'outputs')):
                        loss, _ = trainer.compute_loss(None, collator([dataset[0]]), return_outputs=True)
                    expected = 2.5 * 3 + (0.3 * 5 if run_type.endswith('forget_retain') else 0)
                    self.assertAlmostEqual(loss.item(), expected)
