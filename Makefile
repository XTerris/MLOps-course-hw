.PHONY: install generate bench check clean

install:
	uv sync --locked

generate:
	uv run --locked python -m src.generate

bench:
	uv run --locked python -m src.bench

check:
	bash tests/check.sh

clean:
	rm -rf docs/bench.json out1.txt out2.txt params.yaml.bak
