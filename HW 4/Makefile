.PHONY: install sample tokenize repro check clean

install:
	uv sync --project .. --locked

# Стадия 0, только для преподавателя: parquet курса -> data/train.jsonl, data/val.jsonl.
# У студента вход другой — его датасет из ДЗ 3, положенный в те же файлы.
sample:
	uv run --project .. --locked python -m scripts.make_sample

# Стадия tokenize: JSONL -> токенизированный датасет + метрики + отчёт.
# Та же команда объявлена стадией в dvc.yaml — оттуда её и запускает dvc repro.
tokenize:
	uv run --project .. --locked python -m src.tokenize_data

repro:
	uv run --project .. --locked dvc repro

check:
	bash tests/check.sh

clean:
	rm -rf data metrics/tokenize.json docs/tokenize_report.md
