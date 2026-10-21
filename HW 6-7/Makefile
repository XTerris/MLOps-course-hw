.PHONY: install up down sweep promote predict check check-storage
install:
	uv sync --project .. --frozen
up:
	docker compose -p mlops-hw6 up --build -d --wait
down:
	docker compose -p mlops-hw6 down
sweep:
	uv run --project .. --frozen python -m src.sweep
promote:
	uv run --project .. --frozen python -m evaluation.cli promote
predict:
	uv run --project .. --frozen python -m src.predict --uri models:/my-qwen-adapter@champion --prompt "Какой показатель отражает запасы железа?"
check:
	bash tests/check.sh
check-storage:
	uv run --project .. --frozen python tests/check_storage.py

.PHONY: report candidate baseline-init evaluate eval check-eval check-negative check-all baseline-update predict-candidate
report:
	uv run --project .. --frozen python -m src.report
candidate:
	uv run --project .. --frozen python -m evaluation.cli candidate
baseline-init:
	uv run --project .. --frozen python -m evaluation.cli baseline-init
evaluate:
	uv run --project .. --frozen python -m evaluation.cli evaluate
eval:
	uv run --project .. --frozen python -m evaluation.cli gate
check-eval:
	uv run --project .. --frozen python -m unittest discover -s tests -p 'test_*.py' -v
check-negative:
	uv run --project .. --frozen python -m evaluation.cli negative
check-all: check check-eval
baseline-update:
	uv run --project .. --frozen python -m evaluation.cli baseline-update
predict-candidate:
	uv run --project .. --frozen python -m src.predict --uri models:/my-qwen-adapter@candidate --prompt "Выберите правильный вариант."
