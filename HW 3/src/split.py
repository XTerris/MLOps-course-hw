"""Стадия split: детерминированное групповое разбиение train/val/test."""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group, normalize_text


def validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != {"train", "val", "test"}:
        raise ValueError("split.ratios должен содержать train, val и test")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("все split.ratios должны быть положительными")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("сумма split.ratios должна быть равна 1")


def group_split(
    examples: list[Example], ratios: dict[str, float], seed: int
) -> dict[str, list[Example]]:
    """Разложить целые репозитории, сохраняя размер и баланс классов.

    Сначала случайно (но воспроизводимо) разрешаем равенства, затем большие
    группы распределяем жадно в сплит с максимальным относительным дефицитом
    строк и меток. Ни одна группа не может оказаться в двух сплитах.
    """
    validate_ratios(ratios)
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[normalize_group(example.topic)].append(example)

    groups = list(grouped.items())
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    names = list(ratios)
    labels = sorted({normalize_text(ex.assistant) for ex in examples})
    total_labels = Counter(normalize_text(ex.assistant) for ex in examples)
    target_rows = {name: len(examples) * ratios[name] for name in names}
    target_labels = {
        name: {label: total_labels[label] * ratios[name] for label in labels}
        for name in names
    }
    buckets: dict[str, list[Example]] = {name: [] for name in names}
    row_counts = Counter()
    label_counts: dict[str, Counter[str]] = {name: Counter() for name in names}

    for _, rows in groups:
        group_labels = Counter(normalize_text(ex.assistant) for ex in rows)

        def score(name: str) -> tuple[float, float]:
            row_deficit = (target_rows[name] - row_counts[name]) / max(target_rows[name], 1)
            class_deficits = [
                (target_labels[name][label] - label_counts[name][label])
                / max(target_labels[name][label], 1)
                for label in labels
                if group_labels[label]
            ]
            class_score = sum(class_deficits) / max(len(class_deficits), 1)
            return (row_deficit + class_score, row_deficit)

        chosen = max(names, key=score)
        buckets[chosen].extend(rows)
        row_counts[chosen] += len(rows)
        label_counts[chosen].update(group_labels)

    for rows in buckets.values():
        rows.sort(key=lambda example: example.id)
    return buckets


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if not examples:
        raise SystemExit("split: очищенный датасет пуст")
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    buckets = group_split(examples, cfg["ratios"], cfg["seed"])
    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for example in rows:
                fh.write(dump(example) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    all_groups = {normalize_group(ex.topic) for ex in examples}
    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(all_groups),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows})
            for name, rows in buckets.items()
        },
        "class_counts": {
            name: dict(sorted(Counter(ex.assistant for ex in rows).items()))
            for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    metric_path = Path(paths["metrics_split"])
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not is_clean(rep):
        raise SystemExit(
            "split: обнаружена контаминация train/test: "
            + json.dumps(rep, ensure_ascii=False)
        )

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(all_groups)}, контаминация 0, {metrics['seconds']} с)"
    )


if __name__ == "__main__":
    main()
