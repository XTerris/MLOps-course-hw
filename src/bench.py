"""Замер производительности машины на выбранной модели.

Три числа меряются РАЗДЕЛЬНО — смешивать их бессмысленно:
  * время загрузки модели  — разовая стоимость старта;
  * tokens/sec             — скорость генерации, только после прогрева;
  * пиковая RSS            — максимум за процесс, а не снимок в конце.
"""

import json
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import load_params
from src.model import generate_tokens, load_model, prepare_inputs


def peak_rss_mb() -> float:
    """Пиковая резидентная память процесса.

    ru_maxrss на macOS в байтах, на Linux в килобайтах.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def synchronize(device) -> None:
    """Дождаться завершения асинхронной работы ускорителя."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_benchmark(params: dict) -> dict:
    """Раздельно измерить загрузку, прогретую генерацию и пик RSS."""
    prompt = params["bench"]["prompt"]
    warmup_runs = params["bench"]["warmup_runs"]
    measure_runs = params["bench"]["measure_runs"]
    if warmup_runs < 1 or measure_runs < 1:
        raise ValueError("warmup_runs и measure_runs должны быть не меньше 1")

    load_started = time.perf_counter()
    tokenizer, model = load_model(params)
    synchronize(model.device)
    load_time = time.perf_counter() - load_started
    inputs = prepare_inputs(tokenizer, model, params, prompt)

    for _ in range(warmup_runs):
        generate_tokens(model, params, inputs)
        synchronize(model.device)

    speeds = []
    durations = []
    token_counts = []
    for _ in range(measure_runs):
        synchronize(model.device)
        started = time.perf_counter()
        new_tokens = generate_tokens(model, params, inputs)
        synchronize(model.device)
        elapsed = time.perf_counter() - started
        n_tokens = len(new_tokens)
        speeds.append(n_tokens / elapsed)
        durations.append(elapsed)
        token_counts.append(n_tokens)

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "model": params["model"]["name"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "dtype": str(model.dtype).removeprefix("torch."),
        "prompt": prompt,
        "generate": dict(params["generate"]),
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "input_tokens": inputs["input_ids"].shape[1],
        "load_time_sec": round(load_time, 4),
        "tokens_per_sec": round(statistics.median(speeds), 2),
        "tokens_per_sec_all": [round(speed, 2) for speed in speeds],
        "generation_time_sec_all": [round(duration, 4) for duration in durations],
        "generated_tokens_all": token_counts,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }


def main() -> None:
    report = run_benchmark(load_params())
    Path("docs").mkdir(exist_ok=True)
    Path("docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
