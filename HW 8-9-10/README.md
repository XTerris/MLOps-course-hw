# ДЗ 8–10: выпуск модели и локальный API

Исправленный проект объединяет расследование семи дефектов, Prefect-flow, quality gate ДЗ 6–7 и FastAPI. Итог, результаты и ограничения приведены в [отчёте](docs/report.md); [HTML-презентация](docs/presentation.html) содержит десять разделов отчёта, стрелки переключают слайды, кнопка включает показ по пять секунд. Видеодемонстрация `HW8910-demo.mp4` передаётся отдельно.

Кандидат из ДЗ 6–7 отклонён. Принятой версии baseline в имеющихся материалах не подтверждено: реальный сервис ниже запускается в диагностическом режиме. Учебные версии в `lab.py` помечены `STUB ONLY` и не подтверждают качество настоящей модели.

## Окружение

Нужен Python 3.12 и ранее подготовленное окружение проекта MLOps с PyTorch, Transformers, PEFT и MLflow из корневого `uv.lock`. Из каталога `HW 8-9-10` можно использовать его, установив Prefect и FastAPI отдельно, не меняя зависимости ДЗ 6–7:

```sh
PY="$(cd .. && pwd)/.venv/bin/python"
uv pip install --python "$PY" --target .deps -r requirements-release.lock
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
```

Файл `requirements-release.lock` фиксирует все зависимости процесса выпуска. Для работы с моделью и MLflow дополнительно нужны зависимости корневого окружения; версия Prefect 3.4.25 проверена с FastAPI 0.115.14. Не включайте `.deps`, `.prefect`, веса и кеши в Git.

## Быстрые проверки

```sh
make check PYTHON="$PY"
make check-hw67 PYTHON="$PY"
```

Первая команда выполняет 13 исходных и девять дополнительных тестов без загрузки весов. Три теста поднимают локальный HTTP-сервер и требуют доступ к loopback. Проверка исходного сломанного прототипа с теми же семью основными тестами сохранена в `evidence/before-integration.log`, успешный результат — в `evidence/check-all.log`. Каталог исходных тестов ДЗ 6–7 не менялся.

## Prefect и восстановление

В `../HW 6-7/data/eval` должен быть доступен golden set. Эти данные находятся в DVC: если их нет в клоне, восстановите их из доступного DVC remote или приложения к ДЗ 6–7.

```sh
TASK_DIR="$(mktemp -d "${TMPDIR:-/tmp/}hw8910.XXXXXX")"
"$PY" flow_experiments.py --hw67 '../HW 6-7' \
  --workspace "$TASK_DIR/lab" --prefect-home "$TASK_DIR/prefect" \
  --logs "$TASK_DIR/logs"
```

Сценарий копирует golden, создаёт учебные версии A/B и запускает отдельные процессы со сбоями и повторами. Логи, состояния задач и коды возврата пишутся в указанный каталог. Примеры выполненного прогона — `evidence/prefect-*.log` и `evidence/prefect-summary.json`. В его публикациях используются только синтетические фикстуры. `PREFECT_HOME` должен быть новым для чистого опыта.

Для восстановления артефакта кандидата нужен работающий локальный MLflow из ДЗ 6–7 на `127.0.0.1:5006`:

```sh
"$PY" make_candidate_manifest.py --hw67 '../HW 6-7' \
  --artifact-output "$TASK_DIR/candidate-artifacts" --output "$TASK_DIR/candidate-manifest.json"
"$PY" gate_report.py --hw67 '../HW 6-7' --manifest "$TASK_DIR/candidate-manifest.json" \
  --output "$TASK_DIR/release-report.json"
```

Команда gate возвращает код 2 при ожидаемом отказе. Затем `verify_real_refusal.py` запускает этот кандидат через Prefect и проверяет неизменность указателя учебного стенда:

```sh
"$PY" verify_real_refusal.py --workspace "$TASK_DIR/lab" \
  --manifest "$TASK_DIR/candidate-manifest.json" --report "$TASK_DIR/release-report.json" \
  --prefect-home "$TASK_DIR/prefect" --log "$TASK_DIR/refusal.log"
```

Отчёт сохраняет отказ и ссылки на исходные отчёты и ответы. Старый eval не фиксировал хеши model/runtime; их вычисление после оценки само по себе не подтверждает положительный допуск. `lab.py run` ограничен учебными фикстурами, реальный путь идёт через gate в Prefect.

## Диагностический API и замер

Нужен локальный снимок `Qwen/Qwen3-0.6B` ревизии, указанной в ДЗ 6–7. Следующая команда пересчитывает хеши настоящих файлов и создаёт manifest, не публикуя модель:

```sh
"$PY" make_baseline_manifest.py --hw67 '../HW 6-7' --output "$TASK_DIR/baseline.json"
"$PY" service.py --backend transformers --diagnostic-manifest "$TASK_DIR/baseline.json" \
  --workspace "$TASK_DIR/lab" --port 18918
```

После ответа `/ready` со статусом 200 в другом терминале выполните:

```sh
"$PY" benchmark.py --url http://127.0.0.1:18918 --count 8 --output "$TASK_DIR/benchmark.json"
```

Сохранённые результаты предыдущего запуска находятся в `results/real-smoke.json` и `results/benchmark.json`. Manifest содержит абсолютный путь к локальному снимку, поэтому на другой машине его нужно пересоздать. Для учебного стенда используйте `service.py --workspace "$TASK_DIR/lab"` без `--backend transformers`.

## Доступ к данным

Файлы модели и адаптера не включены в Git. Для воспроизведения нужны доступный локальный снимок базовой модели, MLflow-артефакт training run и наборы DVC из ДЗ 6–7. Их нужно отдельно предоставить преподавателю либо указать доступное ему хранилище; локальные абсолютные пути в сохранённых manifest и журналах не работают на чужой машине.
