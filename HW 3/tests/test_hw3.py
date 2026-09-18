from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.collect import GitHubClient, english_probability, label_classes, pick_prompt
from src.contamination import is_clean, report
from src.dedup import exact_duplicates, near_duplicates
from src.diversity import violations
from src.pii import scrub
from src.schema import Example, SchemaError, iter_examples
from src.split import group_split


def example(index: int, repo: str, label: str = "bug", user: str | None = None) -> Example:
    return Example.model_validate(
        {
            "id": f"github:{repo}#{index}",
            "topic": repo,
            "messages": [
                {"role": "system", "content": f"Prompt variant {index % 5}"},
                {"role": "user", "content": user or f"Unique issue text number {index} for {repo}"},
                {"role": "assistant", "content": label},
            ],
        }
    )


class CollectTests(unittest.TestCase):
    def test_label_mapping_rejects_ambiguous_issue(self) -> None:
        mapping = {
            "bug": ["bug"],
            "documentation": ["documentation", "docs"],
            "feature": ["enhancement"],
            "question": ["question"],
        }
        labels = [{"name": "Bug"}, {"name": "enhancement"}]
        self.assertEqual(label_classes(labels, mapping), {"bug", "feature"})

    def test_prompt_selection_is_stable(self) -> None:
        prompts = ["one", "two", "three"]
        chosen = pick_prompt("github:owner/repo#1", prompts)
        self.assertEqual(chosen, pick_prompt("github:owner/repo#1", prompts))
        self.assertIn(chosen, prompts)

    def test_english_probability(self) -> None:
        english = "The command fails when the configuration file contains an empty section."
        russian = "Команда завершается ошибкой при пустой секции конфигурации."
        self.assertGreater(english_probability(english), 0.8)
        self.assertLess(english_probability(russian), 0.2)

    def test_rate_limit_response_is_retried(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "1"}, request=request)
            return httpx.Response(200, json={"ok": True}, request=request)

        client = GitHubClient(
            "https://api.github.test",
            "2026-03-10",
            transport=httpx.MockTransport(handler),
        )
        try:
            with patch("src.collect.time.sleep") as sleep:
                self.assertEqual(client.get_json("/test"), {"ok": True})
            sleep.assert_called_once_with(1.0)
            self.assertEqual(client.request_count, 2)
        finally:
            client.close()


class CleaningTests(unittest.TestCase):
    def test_schema_error_contains_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            good = example(1, "owner/repo").model_dump()
            bad = example(2, "owner/repo").model_dump()
            bad["messages"][2]["role"] = "user"
            path.write_text(
                json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SchemaError, r"raw\.jsonl:2"):
                list(iter_examples(path))

    def test_pii_scrubbing(self) -> None:
        cleaned, hits = scrub(
            "Born: дата рождения 12.03.1975, phone +7 (999) 123-45-67, "
            "email ivanov@example.ru"
        )
        self.assertEqual(sum(hits.values()), 3)
        self.assertIn("[DATE]", cleaned)
        self.assertIn("[PHONE]", cleaned)
        self.assertIn("[EMAIL]", cleaned)

    def test_exact_and_near_duplicates(self) -> None:
        base = (
            "The application crashes after saving the configuration when the output "
            "directory contains spaces and the cache is enabled"
        )
        near = base.replace("configuration", "configuration   ").replace("cache", "cache,")
        other = "Please add a new command that exports reports in a machine readable format"
        self.assertEqual(exact_duplicates([base, base, other]), [1])
        self.assertEqual(near_duplicates([base, near, other], 3, 128, 0.85), [1])


class SplitAndGateTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self) -> None:
        labels = ["bug", "documentation", "feature", "question"]
        rows = [
            example(i, f"owner/repo-{i // 4:02d}", labels[i % 4])
            for i in range(240)
        ]
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        first = group_split(rows, ratios, 42)
        second = group_split(rows, ratios, 42)
        self.assertEqual(
            {name: [row.id for row in values] for name, values in first.items()},
            {name: [row.id for row in values] for name, values in second.items()},
        )
        groups = {name: {row.topic for row in values} for name, values in first.items()}
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertLess(abs(len(first["train"]) / len(rows) - 0.8), 0.05)

    def test_contamination_detects_group_overlap(self) -> None:
        left = [example(1, "owner/repo-a")]
        right = [example(2, "owner/repo-a", user="A completely different issue body")]
        result = report(left, right, shingle_words=3, num_perm=64, threshold=0.8)
        self.assertEqual(result["group_overlap"], 1)
        self.assertFalse(is_clean(result))

    def test_diversity_reports_classifier_specific_failures(self) -> None:
        stats = {
            "examples": 1200,
            "system_prompts": 1,
            "groups": 1,
            "largest_group": "owner/repo",
            "largest_group_share": 1.0,
            "label_count": 1,
            "largest_label": "bug",
            "largest_label_share": 1.0,
            "answer_len": {"ratio_p90_p10": 1.0},
            "same_length_share": 1.0,
            "duplicate_answer_share": 0.999,
        }
        cfg = {
            "min_examples": 1000,
            "min_system_prompts": 3,
            "min_groups": 50,
            "max_group_share": 0.1,
            "min_labels": 4,
            "max_label_share": 0.35,
            "min_answer_len_ratio": 1.6,
            "max_same_length_share": 0.35,
            "max_duplicate_answer_share": 1.0,
        }
        found = violations(stats, cfg)
        self.assertTrue(any("системных промптов" in message for message in found))
        self.assertTrue(any("классов ответов" in message for message in found))
        self.assertTrue(any("крупнейший класс" in message for message in found))


if __name__ == "__main__":
    unittest.main()
