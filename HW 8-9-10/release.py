"""Контракт выпуска; только стандартная библиотека и без загрузки весов."""
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return sha256(canonical({'data_sha256': sha256(Path(data_path).read_bytes()),
                             'config': config}))


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_release(manifest: dict, report: dict) -> None:
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise ValueError('Ожидались manifest и report')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Неизвестная версия схемы')
    if not all(_nonempty(manifest.get(k)) for k in ('model_id', 'artifact_uri')):
        raise ValueError('Не указан идентификатор или URI')
    keys = ('model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256')
    if not all(_digest(manifest.get(k)) for k in keys):
        raise ValueError('Некорректный SHA-256')
    runtime = manifest.get('runtime')
    try:
        valid_runtime = isinstance(runtime, dict) and bool(runtime) and sha256(canonical(runtime)) == manifest['runtime_sha256']
    except (TypeError, ValueError):
        valid_runtime = False
    if not valid_runtime:
        raise ValueError('Runtime и его SHA-256 не согласованы')
    if report.get('status') != 'complete' or report.get('passed') is not True or not _nonempty(report.get('evaluation_run_id')):
        raise ValueError('Оценка не завершена или выпуск отклонён')
    if any(report.get(k) != manifest[k] for k in ('model_id', *keys)):
        raise ValueError('Отчёт и manifest не согласованы')
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('Нет метрик')
    for name in ('accuracy', 'format_rate'):
        value = metrics.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Недопустимая метрика {name}')


def verify_files(manifest: dict, root: Path) -> None:
    """Пересчитать все заданные файлы; сам validate_release проверяет метаданные."""
    hashes = manifest.get('file_hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Не задан состав файлов артефакта')
    root = Path(root).resolve()
    uri = manifest.get('artifact_uri')
    if not _nonempty(uri) or '://' in uri or uri.startswith('models:/'):
        raise ValueError('Нужен локальный путь к сохранённому артефакту')
    artifact = Path(uri)
    if not artifact.is_absolute():
        artifact = root / artifact
    artifact = artifact.resolve()
    if not artifact.exists() or (artifact.is_file() and artifact.name not in hashes):
        raise ValueError('URI не указывает на проверяемый артефакт')
    # Фикстура указывает на один файл, реальный артефакт обычно на каталог.
    root = artifact if artifact.is_dir() else artifact.parent
    for name, expected in hashes.items():
        if not _nonempty(name) or not _digest(expected):
            raise ValueError('Некорректная запись файла')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Имя файла должно быть относительным путём без ..')
        # Hugging Face snapshot состоит из ссылок на blobs; содержимое проверяется хешем.
        path = root / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f'Файл артефакта отсутствует или изменён: {name}')
    if sha256(canonical(hashes)) != manifest.get('model_sha256'):
        raise ValueError('Отпечаток набора файлов не совпадает')


def validate_stub(manifest: dict, report: dict, workspace: Path) -> None:
    """Синтетическое решение не допускает публикацию настоящей модели."""
    if report.get('synthetic') is not True or not str(manifest.get('model_id','')).startswith('fixture-'):
        raise ValueError('Этот запуск допускает только синтетические STUB-фикстуры')
    verify_files(manifest, workspace)
    uri = Path(manifest['artifact_uri'])
    artifact = (uri if uri.is_absolute() else Path(workspace)/uri).resolve()
    base = artifact if artifact.is_dir() else artifact.parent
    if any(b'STUB ONLY' not in (base/name).read_bytes() for name in manifest['file_hashes']):
        raise ValueError('Тестовый артефакт должен содержать маркер STUB ONLY')


def atomic_write(target: Path, data: bytes) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name + '.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    atomic_write(target, json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
