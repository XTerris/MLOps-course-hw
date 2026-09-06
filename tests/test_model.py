import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import BatchEncoding

from src.config import load_params
from src.model import build_prompt, generate, generate_tokens, load_model, set_seed


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.params = load_params()
        self.inputs = BatchEncoding({"input_ids": torch.tensor([[10, 11]])})
        self.model = SimpleNamespace(
            device=torch.device("cpu"),
            generate=Mock(return_value=torch.tensor([[10, 11, 20, 21, 22]])),
        )

    def test_loader_uses_config_and_sets_seed(self):
        self.params["model"].update(name="test/model", device="cpu", dtype="float32")
        with (
            patch("src.model.AutoTokenizer.from_pretrained") as tokenizer_loader,
            patch("src.model.AutoModelForCausalLM.from_pretrained") as model_loader,
            patch("src.model.set_seed") as seed,
        ):
            tokenizer, model = load_model(self.params)
        tokenizer_loader.assert_called_once_with("test/model")
        model_loader.assert_called_once_with(
            "test/model", dtype=torch.float32, device_map="cpu"
        )
        seed.assert_called_once_with(self.params["generate"]["seed"])
        model.eval.assert_called_once_with()
        self.assertIs(tokenizer, tokenizer_loader.return_value)

    def test_chat_template_receives_thinking_flag(self):
        tokenizer = Mock()
        build_prompt(tokenizer, self.params, "prompt")
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "prompt"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def test_thinking_flag_is_optional(self):
        del self.params["generate"]["enable_thinking"]
        tokenizer = Mock()
        build_prompt(tokenizer, self.params, "prompt")
        self.assertNotIn("enable_thinking", tokenizer.apply_chat_template.call_args.kwargs)

    def test_greedy_generation_crops_prompt_and_disables_sampling(self):
        tokens = generate_tokens(self.model, self.params, self.inputs)
        self.assertEqual(tokens.tolist(), [20, 21, 22])
        options = self.model.generate.call_args.kwargs
        self.assertFalse(options["do_sample"])
        self.assertIsNone(options["temperature"])
        self.assertEqual(options["max_new_tokens"], 200)

    def test_positive_temperature_enables_sampling(self):
        self.params["generate"]["temperature"] = 0.7
        generate_tokens(self.model, self.params, self.inputs)
        options = self.model.generate.call_args.kwargs
        self.assertTrue(options["do_sample"])
        self.assertEqual(options["temperature"], 0.7)

    def test_generate_decodes_only_new_tokens(self):
        tokenizer = Mock(return_value=self.inputs)
        tokenizer.apply_chat_template.return_value = "chat prompt"
        tokenizer.decode.return_value = "answer"
        self.assertEqual(generate(tokenizer, self.model, self.params, "prompt"), ("answer", 3))
        tokenizer.assert_called_once_with(
            "chat prompt", return_tensors="pt", add_special_tokens=False
        )
        self.assertEqual(tokenizer.decode.call_args.args[0].tolist(), [20, 21, 22])
        self.assertTrue(tokenizer.decode.call_args.kwargs["skip_special_tokens"])

    def test_seed_repeats_all_random_sources(self):
        set_seed(42)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        set_seed(42)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
