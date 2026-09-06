import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.bench import peak_rss_mb, run_benchmark, synchronize
from src.config import load_params


class BenchmarkTests(unittest.TestCase):
    def test_load_warmup_and_generation_are_separate(self):
        params = load_params()
        params["model"]["name"] = "test/model"
        model = SimpleNamespace(
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=SimpleNamespace(_commit_hash="revision"),
        )
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        with (
            patch("src.bench.load_model", return_value=("tokenizer", model)) as loader,
            patch("src.bench.prepare_inputs", return_value=inputs) as prepare,
            patch("src.bench.generate_tokens", side_effect=[range(1000), range(100), range(80), range(90)]) as generate,
            patch("src.bench.synchronize") as sync,
            patch("src.bench.time.perf_counter", side_effect=[0, 5, 10, 12, 20, 24, 30, 31]),
            patch("src.bench.peak_rss_mb", return_value=512),
        ):
            report = run_benchmark(params)
        loader.assert_called_once_with(params)
        prepare.assert_called_once_with("tokenizer", model, params, params["bench"]["prompt"])
        self.assertEqual(generate.call_count, 4)
        self.assertEqual(sync.call_count, 8)
        self.assertEqual(report["load_time_sec"], 5)
        self.assertEqual(report["generation_time_sec_all"], [2, 4, 1])
        self.assertEqual(report["generated_tokens_all"], [100, 80, 90])
        self.assertEqual(report["tokens_per_sec_all"], [50, 20, 90])
        self.assertEqual(report["tokens_per_sec"], 50)
        self.assertEqual(report["peak_rss_mb"], 512)
        self.assertEqual(report["model"], "test/model")
        self.assertEqual(report["dtype"], "float32")

    def test_empty_measurement_or_missing_warmup_is_rejected(self):
        for field in ("warmup_runs", "measure_runs"):
            with self.subTest(field=field):
                params = load_params()
                params["bench"][field] = 0
                with self.assertRaises(ValueError), patch("src.bench.load_model") as loader:
                    run_benchmark(params)
                loader.assert_not_called()

    def test_peak_rss_macos_bytes(self):
        with (
            patch("src.bench.sys.platform", "darwin"),
            patch("src.bench.resource.getrusage", return_value=SimpleNamespace(ru_maxrss=1048576)),
        ):
            self.assertEqual(peak_rss_mb(), 1)

    def test_peak_rss_linux_kibibytes(self):
        with (
            patch("src.bench.sys.platform", "linux"),
            patch("src.bench.resource.getrusage", return_value=SimpleNamespace(ru_maxrss=1024)),
        ):
            self.assertEqual(peak_rss_mb(), 1)

    def test_accelerators_are_synchronized_and_cpu_is_not(self):
        with patch("torch.mps.synchronize") as mps, patch("torch.cuda.synchronize") as cuda:
            synchronize(torch.device("cpu"))
            mps.assert_not_called()
            cuda.assert_not_called()
            synchronize(torch.device("mps"))
            mps.assert_called_once_with()
            synchronize(torch.device("cuda:0"))
            cuda.assert_called_once_with(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
