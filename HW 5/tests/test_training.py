"""Проверяем математическую эквивалентность и потери данных в цикле."""

import copy
import unittest

import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.data import pad_batch
from src.loss import answer_loss
from src.runtime import set_seed
from src.train import evaluate, lora_config, train_loop

torch.set_num_threads(1)


def model():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, attention_dropout=0.0, pad_token_id=0,
    ))


def examples():
    return [
        {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5, "labels": [-100, -100, -100, 4, 5]},
        {"input_ids": [6, 7, 8], "attention_mask": [1] * 3, "labels": [-100, -100, 8]},
        {"input_ids": [2, 9, 1, 7], "attention_mask": [1] * 4, "labels": [-100, -100, 1, 7]},
        {"input_ids": [8, 4, 2], "attention_mask": [1] * 3, "labels": [-100, 4, 2]},
        {"input_ids": [1, 3, 9], "attention_mask": [1] * 3, "labels": [-100, -100, 9]},
    ]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_sparse_loss_and_gradients_equal_dense_causal_lm(self):
        dense = model().eval()
        sparse = copy.deepcopy(dense)
        batch = pad_batch(examples()[:2], 0)
        dense_loss = dense(**batch).loss
        total, count = answer_loss(sparse, batch)
        self.assertEqual(count, 3)
        torch.testing.assert_close(total / count, dense_loss)
        dense_loss.backward()
        (total / count).backward()
        for a, b in zip(dense.parameters(), sparse.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-7)

    def test_validation_is_token_weighted_and_restores_mode(self):
        m = model().train()
        loss = evaluate(m, examples(), 0, torch.device("cpu"), 2)
        self.assertTrue(m.training)
        reference = m.eval()(**pad_batch(examples(), 0)).loss.item()
        self.assertAlmostEqual(loss, reference, places=5)
        m.eval()
        evaluate(m, examples(), 0, torch.device("cpu"), 1)
        self.assertFalse(m.training)

    def test_incomplete_accumulation_is_flushed_each_epoch(self):
        cfg = dict(batch_size=2, grad_accum=2, epochs=2, lr=2e-4,
                   weight_decay=0.0, warmup_ratio=0.03, max_grad_norm=1.0,
                   eval_batch_size=2, eval_every=10, seed=42)
        result = train_loop(model(), examples(), examples(), 0, torch.device("cpu"), cfg, None)
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["seen_examples"], 10)
        self.assertEqual(result["seen_tokens"], 36)
        self.assertEqual([p[0] for p in result["curve_val"]], [0, 4])

    def test_accumulation_matches_physical_effective_batch(self):
        a = model()
        b = copy.deepcopy(a)
        cfg = dict(batch_size=1, grad_accum=2, epochs=1, lr=2e-4,
                   weight_decay=0.0, warmup_ratio=0.03, max_grad_norm=1.0,
                   eval_batch_size=2, eval_every=10, seed=42)
        train_loop(a, examples(), examples(), 0, torch.device("cpu"), cfg, None)
        train_loop(b, examples(), examples(), 0, torch.device("cpu"),
                   {**cfg, "batch_size": 2, "grad_accum": 1}, None)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=3e-5, atol=1e-6)

    def test_lora_freeze_reduces_params_and_preserves_base(self):
        cfg = {"lora": {"r": 2, "alpha": 4, "dropout": 0.0,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
               "model": {}}
        all_m = get_peft_model(model(), lora_config(cfg, 4, 0))
        half_m = get_peft_model(model(), lora_config(cfg, 4, 2))
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(count(all_m), 2 * count(half_m))
        self.assertTrue(all("lora_" in n for n, p in half_m.named_parameters() if p.requires_grad))
        self.assertEqual(half_m.peft_config["default"].layers_to_transform, [2, 3])
        with self.assertRaises(ValueError):
            lora_config(cfg, 4, 4)

    def test_no_supervision_fails_instead_of_silent_nan(self):
        batch = pad_batch(examples()[:1], 0)
        batch["labels"].fill_(-100)
        with self.assertRaises(ValueError):
            answer_loss(model(), batch)


if __name__ == "__main__":
    unittest.main()
