"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def version_config(params: dict) -> dict:
    """Параметры текущей версии датасета.

    Версия живёт в params, а не в аргументах командной строки: иначе DVC не
    увидит переключение v1/v2 и оставит артефакты от предыдущего запуска.
    """
    version = params["collect"]["version"]
    versions = params["collect"]["versions"]
    if version not in versions:
        raise SystemExit(
            f"collect.version = {version!r}, но в collect.versions "
            f"есть только {sorted(versions)}"
        )
    return versions[version]
