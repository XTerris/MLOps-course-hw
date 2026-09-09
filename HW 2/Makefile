.PHONY: install inspect check clean

install:
	uv sync --project .. --locked

inspect:
	uv run --project .. --locked python -m src.inspect_model

check:
	bash tests/check.sh

clean:
	rm -rf docs/report.json src/__pycache__ tests/__pycache__
