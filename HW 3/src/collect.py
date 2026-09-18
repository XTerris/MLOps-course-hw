"""Стадия collect: GitHub Issues API → data/raw.jsonl.

Источник — публичные issues англоязычных Python-проектов. Стандартные labels
используются как слабая разметка четырёх классов. Сбор детерминирован:
фиксированы диапазоны дат, порядок запросов и выбор системного промпта.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from langdetect import DetectorFactory, LangDetectException, detect_langs

from src.config import load_params, version_config

DetectorFactory.seed = 0


class GitHubApiError(RuntimeError):
    """GitHub API вернул неисправимую или исчерпавшую повторы ошибку."""


class GitHubClient:
    """Минимальный REST-клиент с pagination и уважением rate limit."""

    def __init__(
        self,
        base_url: str,
        api_version: str,
        token: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": api_version,
            "User-Agent": "bmstu-mlops-hw3-dataset",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self.request_count = 0
        self.cache_hits = 0
        self._license_cache: dict[str, str] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
        if response.headers.get("X-RateLimit-Remaining") == "0":
            try:
                reset = float(response.headers["X-RateLimit-Reset"])
                return max(1.0, reset - time.time() + 1.0)
            except (KeyError, ValueError):
                pass
        return min(60.0, float(2**attempt))

    def get_json(
        self, path_or_url: str, *, params: dict[str, Any] | None = None, retries: int = 6
    ) -> dict[str, Any]:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        cache_path: Path | None = None
        if self.cache_dir is not None:
            cache_key = hashlib.sha256(
                json.dumps([url, params or {}], sort_keys=True).encode("utf-8")
            ).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                self.cache_hits += 1
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        last: httpx.Response | None = None
        for attempt in range(retries + 1):
            try:
                response = self.client.get(url, params=params)
            except httpx.RequestError as exc:
                if attempt == retries:
                    raise GitHubApiError(f"GitHub API недоступен: {exc}") from exc
                time.sleep(min(60.0, float(2**attempt)))
                continue
            self.request_count += 1
            last = response
            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise GitHubApiError(f"GitHub API вернул не объект: {url}")
                if cache_path is not None:
                    cache_path.write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )
                return payload
            if response.status_code in {403, 429, 500, 502, 503, 504} and attempt < retries:
                delay = self._retry_delay(response, attempt)
                if response.status_code in {403, 429}:
                    print(
                        f"collect: GitHub rate limit, ожидание {delay:.0f} с",
                        flush=True,
                    )
                time.sleep(delay)
                continue
            message = response.text[:300].replace("\n", " ")
            raise GitHubApiError(
                f"GitHub API: HTTP {response.status_code} для {url}: {message}"
            )
        assert last is not None
        raise GitHubApiError(f"GitHub API: исчерпаны повторы, HTTP {last.status_code}")

    def licensed_repositories(self, cfg: dict[str, Any]) -> dict[str, str]:
        """Одним search-запросом на лицензию построить проверенный пул репозиториев."""
        license_queries = {
            "MIT": "mit",
            "Apache-2.0": "apache-2.0",
            "BSD-2-Clause": "bsd-2-clause",
            "BSD-3-Clause": "bsd-3-clause",
            "ISC": "isc",
        }
        def fetch(per_license: int) -> dict[str, str]:
            repositories: dict[str, str] = {}
            for spdx in cfg["allowed_licenses"]:
                query_license = license_queries.get(spdx)
                if query_license is None:
                    raise ValueError(f"для SPDX {spdx!r} не задан GitHub license qualifier")
                payload = self.get_json(
                    "/search/repositories",
                    params={
                        "q": (
                            f'language:{cfg["repository_language"]} '
                            f"license:{query_license} archived:false"
                        ),
                        "sort": "stars",
                        "order": "desc",
                        "per_page": per_license,
                        "page": 1,
                    },
                )
                for item in payload.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("full_name") or "")
                    license_info = item.get("license") or {}
                    item_spdx = (
                        license_info.get("spdx_id")
                        if isinstance(license_info, dict)
                        else None
                    )
                    if name and item_spdx == spdx:
                        repositories[name] = spdx
            return repositories

        # Базовый пул остаётся стабильным при расширении, поэтому старые query
        # chunks совпадают побайтно и извлекаются из временного HTTP-кэша.
        base = fetch(int(cfg["repository_base_per_license"]))
        expanded = fetch(int(cfg["repositories_per_license"]))
        ordered: dict[str, str] = dict(sorted(base.items()))
        for name, spdx in sorted(expanded.items()):
            ordered.setdefault(name, spdx)
        return ordered

    def search_issues(
        self, label: str, cfg: dict[str, Any], repositories: list[str]
    ) -> Iterator[dict[str, Any]]:
        """Выдать issues из заранее лицензированного пула репозиториев."""
        chunk_size = int(cfg["repositories_per_query"])
        for offset in range(0, len(repositories), chunk_size):
            chunk = repositories[offset : offset + chunk_size]
            repo_scope = " ".join(f"repo:{name}" for name in chunk)
            for date_window in cfg["date_windows"]:
                query = (
                    f'is:issue {repo_scope} label:"{label}" created:{date_window}'
                )
                for page in range(1, cfg["max_pages_per_window"] + 1):
                    payload = self.get_json(
                        "/search/issues",
                        params={
                            "q": query,
                            "sort": "created",
                            "order": "asc",
                            "per_page": cfg["per_page"],
                            "page": page,
                        },
                    )
                    items = payload.get("items", [])
                    if not isinstance(items, list):
                        raise GitHubApiError("GitHub search response: items не является списком")
                    for item in items:
                        if isinstance(item, dict):
                            yield item
                    available = min(int(payload.get("total_count", 0)), 1000)
                    if len(items) < cfg["per_page"] or page * cfg["per_page"] >= available:
                        break

    def repository_license(self, repository_url: str) -> str:
        if repository_url not in self._license_cache:
            payload = self.get_json(repository_url)
            license_info = payload.get("license") or {}
            spdx = license_info.get("spdx_id") if isinstance(license_info, dict) else None
            self._license_cache[repository_url] = str(spdx or "NOASSERTION")
        return self._license_cache[repository_url]


def english_probability(text: str) -> float:
    """Детерминированная вероятность английского от langdetect."""
    try:
        probabilities = detect_langs(text[:12000])
    except LangDetectException:
        return 0.0
    for probability in probabilities:
        if probability.lang == "en":
            return float(probability.prob)
    return 0.0


def label_classes(labels: list[Any], mapping: dict[str, list[str]]) -> set[str]:
    """Канонические классы, найденные среди labels issue."""
    names = {
        str(label.get("name", "")).strip().casefold()
        for label in labels
        if isinstance(label, dict)
    }
    return {
        canonical
        for canonical, aliases in mapping.items()
        if names & {alias.casefold() for alias in aliases}
    }


def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Стабильный вариант инструкции; встроенный hash() для этого непригоден."""
    digest = hashlib.sha256(example_id.encode("utf-8")).digest()
    return variants[int.from_bytes(digest[:8], "big") % len(variants)]


def repository_name(repository_url: str) -> str:
    marker = "/repos/"
    if marker not in repository_url:
        raise ValueError(f"неожиданный repository_url: {repository_url}")
    return repository_url.split(marker, 1)[1].strip("/")


def issue_record(item: dict[str, Any], canonical: str, prompts: list[str]) -> dict[str, Any]:
    repo = repository_name(str(item["repository_url"]))
    example_id = f"github:{repo}#{item['number']}"
    title = str(item.get("title") or "").strip().replace("\x00", "")
    body = str(item.get("body") or "").strip().replace("\x00", "")
    user = f"Title: {title}\n\nBody:\n{body}"
    return {
        "id": example_id,
        "topic": repo,
        "messages": [
            {"role": "system", "content": pick_prompt(example_id, prompts)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": canonical},
        ],
    }


def _write_metrics(path: str, metrics: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    current = version_config(params)
    paths = params["paths"]
    target = int(current["examples_per_class"])
    prompts = list(cfg["system_prompts"])
    if not prompts:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")

    mapping = {name: list(aliases) for name, aliases in cfg["label_mapping"].items()}
    query_labels = {name: aliases[0] for name, aliases in mapping.items()}
    allowed_licenses = set(cfg["allowed_licenses"])
    token = os.environ.get("GITHUB_TOKEN")
    client = GitHubClient(
        cfg["api_url"],
        cfg["api_version"],
        token,
        cache_dir=os.environ.get("GITHUB_API_CACHE"),
    )
    started = time.perf_counter()
    scanned = Counter()
    dropped = Counter()
    selected_counts = Counter()
    repository_counts = Counter()
    license_counts = Counter()
    prompts_used: set[str] = set()
    seen_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    licensed_repositories = client.licensed_repositories(cfg)
    if len(licensed_repositories) * cfg["max_per_repository"] < target * len(mapping):
        raise SystemExit(
            "collect: лицензированный пул репозиториев заведомо мал для заданного объёма"
        )
    streams = {
        canonical: client.search_issues(
            query_label, cfg, list(licensed_repositories)
        )
        for canonical, query_label in query_labels.items()
    }
    exhausted: set[str] = set()

    try:
        while any(selected_counts[name] < target for name in mapping):
            progress = False
            for canonical in mapping:
                if selected_counts[canonical] >= target or canonical in exhausted:
                    continue
                stream = streams[canonical]
                while selected_counts[canonical] < target:
                    try:
                        item = next(stream)
                    except StopIteration:
                        exhausted.add(canonical)
                        break
                    scanned[canonical] += 1
                    if "pull_request" in item:
                        dropped["pull_request"] += 1
                        continue
                    found_classes = label_classes(item.get("labels") or [], mapping)
                    if found_classes != {canonical}:
                        dropped["ambiguous_or_mismatched_label"] += 1
                        continue
                    title = str(item.get("title") or "").strip()
                    body = str(item.get("body") or "").strip()
                    if not title or not body:
                        dropped["empty_title_or_body"] += 1
                        continue
                    if english_probability(f"{title}\n{body}") < cfg["english_min_probability"]:
                        dropped["not_english"] += 1
                        continue
                    repository_url = str(item.get("repository_url") or "")
                    try:
                        repo = repository_name(repository_url)
                    except ValueError:
                        dropped["invalid_repository_url"] += 1
                        continue
                    example_id = f"github:{repo}#{item.get('number')}"
                    if example_id in seen_ids:
                        dropped["duplicate_issue"] += 1
                        continue
                    if repository_counts[repo] >= cfg["max_per_repository"]:
                        dropped["repository_cap"] += 1
                        continue
                    license_id = licensed_repositories.get(repo, "NOASSERTION")
                    if license_id not in allowed_licenses:
                        dropped["license_not_allowed"] += 1
                        continue

                    record = issue_record(item, canonical, prompts)
                    selected.append(record)
                    selected_counts[canonical] += 1
                    repository_counts[repo] += 1
                    license_counts[license_id] += 1
                    prompts_used.add(record["messages"][0]["content"])
                    seen_ids.add(example_id)
                    if len(selected) % 100 == 0:
                        print(
                            f"collect: отобрано {len(selected)} / {target * len(mapping)}",
                            flush=True,
                        )
                    progress = True
                    break
            if not progress and all(
                selected_counts[name] >= target or name in exhausted for name in mapping
            ):
                break
    finally:
        client.close()

    complete = all(selected_counts[name] == target for name in mapping)
    largest_repo = repository_counts.most_common(1)[0] if repository_counts else ("", 0)
    metrics = {
        "version": cfg["version"],
        "complete": complete,
        "authenticated": bool(token),
        "date_windows": list(cfg["date_windows"]),
        "target_per_class": target,
        "rows_scanned": sum(scanned.values()),
        "rows_written": len(selected),
        "class_counts": dict(sorted(selected_counts.items())),
        "repositories": len(repository_counts),
        "largest_repository": largest_repo[0],
        "largest_repository_count": largest_repo[1],
        "largest_repository_share": round(largest_repo[1] / max(len(selected), 1), 4),
        "license_counts": dict(sorted(license_counts.items())),
        "system_prompt_variants": len(prompts_used),
        "drop_reasons": dict(sorted(dropped.items())),
        "scanned_by_class": dict(sorted(scanned.items())),
        "api_requests": client.request_count,
        "api_cache_hits": client.cache_hits,
        "seconds": round(time.perf_counter() - started, 2),
    }
    _write_metrics(paths["metrics_collect"], metrics)
    if not complete:
        shortages = {
            name: target - selected_counts[name]
            for name in mapping
            if selected_counts[name] < target
        }
        raise SystemExit(
            "collect: не удалось набрать целевой объём; нехватка по классам: "
            + json.dumps(shortages, ensure_ascii=False)
        )

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in sorted(selected, key=lambda row: row["id"]):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"collect: {cfg['version']}, {len(selected)} строк, "
        f"классы {dict(sorted(selected_counts.items()))}, "
        f"репозиториев {len(repository_counts)}, API-запросов {client.request_count}, "
        f"{metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()
