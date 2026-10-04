.PHONY: install train train-all train-freeze compare plot repro test check clean

install:
	uv sync --project .. --locked

# Два прогона: адаптеры на всех слоях и с замороженными первыми 14.
train: train-all train-freeze plot

train-all:
	uv run --project .. --locked python -m src.train --variant all_layers

train-freeze:
	uv run --project .. --locked python -m src.train --variant freeze14

plot:
	uv run --project .. --locked python -m src.plot

# Базовая модель против адаптера на пяти фиксированных промптах.
compare:
	uv run --project .. --locked python -m src.compare --variant all_layers

repro:
	uv run --project .. --locked dvc repro

test:
	uv run --project .. --locked python -m unittest discover -s tests -v

check:
	bash tests/check.sh

# clean — артефакты обучения (адаптеры, метрики, графики); всё пересоздаётся make train.
clean:
	rm -rf models metrics/*.json docs/curves.png docs/compare.md .check_*.log
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
