.PHONY: install repro v1 v2 diff dag diversity contamination test acceptance check clean

install:
	uv sync --project .. --locked

repro:
	uv run --project .. --locked dvc repro

v1:
	uv run --project .. --locked python scripts/set_version.py v1
	uv run --project .. --locked dvc repro

v2:
	uv run --project .. --locked python scripts/set_version.py v2
	uv run --project .. --locked dvc repro

diff:
	uv run --project .. --locked dvc metrics diff

dag:
	uv run --project .. --locked dvc dag

diversity:
	uv run --project .. --locked python -m src.diversity

contamination:
	uv run --project .. --locked python scripts/check_contamination.py

test:
	uv run --project .. --locked python -m unittest discover -s tests -p 'test_*.py' -v

acceptance:
	bash tests/check.sh

check: test acceptance

clean:
	rm -rf data metrics/*.json params.yaml.bak params.yaml.orig
