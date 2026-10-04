"""LoRA на собственных тензорах ДЗ 4; отдельные полные и smoke-прогоны."""

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import peft  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup  # noqa: E402

from src.config import load_params
from src.data import batches, load_split
from src.loss import answer_loss
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed, synchronize

TRAIN_CODE = ("src/train.py", "src/data.py", "src/loss.py", "src/runtime.py", "src/config.py")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs_fingerprint(params: dict) -> str:
    """Код, конфиг, версии зависимостей и фактические входные тензоры."""
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    h.update(Path("../uv.lock").read_bytes())
    for split in ("train", "val"):
        h.update(file_sha256(params["data"][split]).encode())
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    if not 0 <= freeze_first < n_layers:
        raise ValueError(f"freeze_first должен быть от 0 до {n_layers - 1}")
    cfg = params["lora"]
    return LoraConfig(
        r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        layers_to_transform=list(range(freeze_first, n_layers)),
        revision=params["model"].get("revision"), task_type="CAUSAL_LM",
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int, on_batch=None) -> float:
    """Средний loss на токен ответа после causal shift, без промпта и pad."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        ordered = sorted(examples, key=lambda e: len(e["input_ids"]))
        for batch in batches(ordered, batch_size, pad_id, shuffle=False, seed=0):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_sum, n = answer_loss(model, batch)
            total += loss_sum.item()
            count += n
            if on_batch:
                on_batch()
        if count == 0 or not math.isfinite(total):
            raise ValueError("валидация не содержит ответа либо loss не конечен")
        return total / count
    finally:
        model.train(was_training)
        if device.type == "mps":
            torch.mps.empty_cache()


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def train_loop(model, examples, val_examples, pad_id, device, tcfg, max_steps, on_eval=None) -> dict:
    """Накопление по токенам; хвост каждой эпохи даёт отдельный шаг."""
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    total_steps = math.ceil(micro_per_epoch / tcfg["grad_accum"]) * tcfg["epochs"]
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps должен быть положительным")
        total_steps = min(total_steps, max_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps,
    )
    curve_train, curve_val = [], []
    eval_seconds, peak = 0.0, allocated_bytes(device)
    started = time.perf_counter()

    def measure(step):
        nonlocal eval_seconds, peak
        synchronize(device)
        tick = time.perf_counter()
        def sample_memory():
            nonlocal peak
            peak = max(peak, allocated_bytes(device))

        value = evaluate(model, val_examples, pad_id, device, tcfg["eval_batch_size"], sample_memory)
        synchronize(device)
        eval_seconds += time.perf_counter() - tick
        peak = max(peak, allocated_bytes(device))
        curve_val.append([step, round(value, 6)])
        if on_eval:
            on_eval(step, value, curve_train)

    measure(0)
    optimizer.zero_grad(set_to_none=True)
    step, seen_tokens, seen_examples = 0, 0, 0
    model.train()
    for epoch in range(tcfg["epochs"]):
        accum_loss, accum_tokens = 0.0, 0
        for micro, batch in enumerate(
            batches(examples, tcfg["batch_size"], pad_id, shuffle=True, seed=tcfg["seed"] + epoch), 1,
        ):
            seen_tokens += int(batch["attention_mask"].sum())
            seen_examples += batch["input_ids"].shape[0]
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_sum, n = answer_loss(model, batch)
            if not torch.isfinite(loss_sum).item():
                raise FloatingPointError(f"не конечный loss перед шагом {step + 1}")
            loss_sum.backward()
            accum_loss += loss_sum.item()
            accum_tokens += n
            peak = max(peak, allocated_bytes(device))
            if micro % tcfg["grad_accum"] and micro != micro_per_epoch:
                continue
            # Нормируем сумму градиентов на число токенов всего эффективного батча.
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(accum_tokens)
            torch.nn.utils.clip_grad_norm_(trainable, tcfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss / accum_tokens, 4)])
            accum_loss, accum_tokens = 0.0, 0
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                measure(step)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    synchronize(device)
    seconds = time.perf_counter() - started - eval_seconds
    return {
        "steps": step, "expected_steps": total_steps,
        "seen_examples": seen_examples, "seen_tokens": seen_tokens,
        "base_val_loss": curve_val[0][1], "final_val_loss": curve_val[-1][1],
        "curve_train": curve_train, "curve_val": curve_val, "diverged": False,
        "seconds": round(seconds, 1), "eval_seconds": round(eval_seconds, 1),
        "seconds_per_step": round(seconds / step, 3),
        "train_tokens_per_sec": round(seen_tokens / seconds, 1),
        "peak_memory_mb": round(peak / 1048576, 1), "memory_metric": memory_metric(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="отдельная папка для smoke-прогона")
    ap.add_argument("--val-limit", type=int, default=None)
    args = ap.parse_args()
    params = load_params()
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}")
    variant, tcfg = variants[args.variant], params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")
    device, dtype = resolve_device(params["model"]["device"]), resolve_dtype(params["model"]["dtype"])
    set_seed(tcfg["seed"])  # до создания модели, LoRA и dropout
    fingerprint = inputs_fingerprint(params)
    train_blob, val_blob = load_split(params["data"]["train"]), load_split(params["data"]["val"])
    for blob in (train_blob, val_blob):
        if blob["model"] != params["model"]["name"] or blob["padding_side"] != "left":
            raise ValueError("модель или сторона padding не совпадает с тензорами ДЗ 4")
    if args.val_limit is not None:
        if args.val_limit < 1:
            raise ValueError("val_limit должен быть положительным")
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
    pad_id = train_blob["pad_token_id"]
    mcfg = params["model"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"], revision=mcfg.get("revision"), padding_side="left")
    if tokenizer.pad_token_id != pad_id or val_blob["pad_token_id"] != pad_id:
        raise ValueError("pad_token_id не совпадает")
    model = AutoModelForCausalLM.from_pretrained(mcfg["name"], revision=mcfg.get("revision"), dtype=dtype).to(device)
    if tcfg.get("gradient_checkpointing"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, model.config.num_hidden_layers, variant["freeze_first"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{args.variant}] {device}, обучаемых {trainable:,} из {total:,} ({trainable / total:.3%}); "
          f"train {len(train_blob['examples'])}, val {len(val_blob['examples'])}", flush=True)

    def on_eval(step, value, curve):
        train = f"train {curve[-1][1]:.4f}; " if curve else "база; "
        print(f"  шаг {step}: {train}val {value:.6f}", flush=True)

    results = train_loop(model, train_blob["examples"], val_blob["examples"], pad_id, device, tcfg, max_steps, on_eval)
    if inputs_fingerprint(params) != fingerprint:
        raise RuntimeError("код, конфиг или входные данные изменились во время обучения")
    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    inference = {
        "model": mcfg, "compare": params["compare"], "padding_side": tokenizer.padding_side,
        "do_sample": False, "inputs_fingerprint": fingerprint,
    }
    (adapter_dir / "inference_config.json").write_text(json.dumps(inference, ensure_ascii=False, indent=2) + "\n")
    metrics = {
        **results, "variant": args.variant, "freeze_first": variant["freeze_first"],
        "model": mcfg["name"], "model_revision": model.config._commit_hash,
        "device": device.type, "dtype": mcfg["dtype"], "seed": tcfg["seed"], "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "train_examples": len(train_blob["examples"]), "val_examples": len(val_blob["examples"]),
        "trainable_params": trainable, "total_params": total, "trainable_share": round(trainable / total, 6),
        "adapter_dir": str(adapter_dir), "adapter_size_mb": dir_size_mb(adapter_dir),
        "adapter_weights_mb": round((adapter_dir / "adapter_model.safetensors").stat().st_size / 1048576, 2),
        "inputs_fingerprint": fingerprint,
        "data_sha256": {k: file_sha256(params["data"][k]) for k in ("train", "val")},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(f"[{args.variant}] {metrics['steps']} шагов, {metrics['seen_examples']} примеров; "
          f"train {metrics['seconds']} с, val {metrics['eval_seconds']} с; "
          f"память {metrics['peak_memory_mb']} МиБ; адаптер {metrics['adapter_size_mb']} МиБ", flush=True)


if __name__ == "__main__":
    main()
