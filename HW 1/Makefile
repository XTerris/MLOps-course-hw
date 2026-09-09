.PHONY: install generate bench check test clean

install:
	uv sync --project .. --locked

generate:
	uv run --project .. --locked python -m src.generate

bench:
	uv run --project .. --locked python -m src.bench

check:
	bash tests/check.sh

test:
	uv run --project .. --locked python -m unittest discover -s tests -v

clean:
	rm -rf docs/bench.json out1.txt out2.txt params.yaml.bak
