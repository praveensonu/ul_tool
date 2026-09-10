import unittest

import pandas as pd
from pydantic import ValidationError

from config.training_config import build_orchestrator_config
from orchestrator import build_unlearning_run
from schemas import FinalTrainingConfigRequest
from tests.test_forget_only import Tokenizer


class MethodHyperparameterTests(unittest.TestCase):
    def request(self, method, overrides):
        return FinalTrainingConfigRequest(
            model_name='test', method='full', unlearning_method=method,
            gpu_ids=[0], forget_set_path='forget.parquet', retain_set_path='retain.parquet',
            hyperparams=dict(
                general=dict(max_steps=1, learning_rate=0.001, context_length=8),
                optimization=dict(batch_size=1, grad_accum=1, weight_decay=0),
                schedule=dict(save_steps=1), memory={}, method=overrides,
            ),
        )

    def test_defaults_and_custom_values_reach_both_trainer_variants(self):
        data = pd.DataFrame({'question': ['Q'], 'answer': ['a'], 'alternate': ['b']})
        defaults = {'dpo': {'beta': 0.1}, 'npo': {'beta': 0.1}, 'simnpo': {'beta': 4.5, 'delta': 0.0}}
        for method, expected in defaults.items():
            for overrides in ({}, {'beta': 0.7}, {'delta': 1.2} if method == 'simnpo' else {}):
                config = build_orchestrator_config(self.request(method, overrides))
                params = config['hyperparams']['method']
                self.assertEqual(params, dict(expected, **overrides))
                for retain in (None, data):
                    _, _, _, args, _ = build_unlearning_run(
                        method, data, retain, Tokenizer(), 8, method_hyperparams=params
                    )
                    for key, value in params.items():
                        self.assertEqual(args[key], value)

    def test_gradient_methods_have_no_extra_parameters(self):
        for method in ('grad_ascent', 'grad_diff'):
            config = build_orchestrator_config(self.request(method, {}))
            self.assertEqual(config['hyperparams']['method'], {})
            with self.assertRaises(ValidationError):
                self.request(method, {'beta': 0.1})

    def test_invalid_values_and_wrong_method_fields(self):
        for method, overrides in (
            ('dpo', {'delta': 1}), ('npo', {'delta': 1}),
            ('dpo', {'beta': 0}), ('npo', {'beta': -1}),
            ('simnpo', {'beta': float('inf')}), ('simnpo', {'delta': float('nan')}),
        ):
            with self.subTest(method=method, overrides=overrides), self.assertRaises(ValidationError):
                self.request(method, overrides)
