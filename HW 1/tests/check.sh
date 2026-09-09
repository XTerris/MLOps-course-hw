#!/usr/bin/env bash
# Самопроверка домашней работы 1.
# Зелёный check.sh необходим для сдачи, но не достаточен: код читается глазами.
set -euo pipefail
cd "$(dirname "$0")/.."

work_dir=$(mktemp -d)
cp params.yaml "$work_dir/params.yaml"
cleanup() {
  cp "$work_dir/params.yaml" params.yaml
  rm -rf "$work_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }

echo
echo "1. Имя модели не захардкожено в коде"
if grep -rqE '"(Qwen|HuggingFaceTB|meta-llama)/' src/ 2>/dev/null; then
  fail "в src/ найдено имя модели строкой — оно должно жить только в params.yaml"
  grep -rnE '"(Qwen|HuggingFaceTB|meta-llama)/' src/ | sed 's/^/      /'
else
  ok "в src/ имён моделей нет"
fi

echo
echo "2. Смена модели в params.yaml не требует правок в коде"
sed 's|^  name: .*|  name: "HuggingFaceTB/SmolLM2-135M-Instruct"|' "$work_dir/params.yaml" > params.yaml
# Мало убедиться, что не упало: проверяем, что загрузилась именно та модель,
# которая указана в конфиге. Иначе захардкоженное имя проходит незамеченным.
if uv run --project .. --locked python -c "
from src.config import load_params
from src.model import load_model
p = load_params()
_, m = load_model(p)
got = str(getattr(m.config, 'name_or_path', ''))
assert p['model']['name'] in got, f'конфиг просит {p[\"model\"][\"name\"]}, а загружено {got}'
" > "$work_dir/model.log" 2>&1; then
  ok "загружается именно та модель, что указана в params.yaml"
else
  fail "загружена не та модель — имя берётся не из конфига"
  cat "$work_dir/model.log"
fi
cp "$work_dir/params.yaml" params.yaml

echo
echo "3. Генерация воспроизводима при temperature = 0"
if ! uv run --project .. --locked python -c "from src.config import load_params; assert load_params()['generate']['temperature'] == 0"; then
  fail "для этой проверки требуется temperature = 0"
elif ! make --no-print-directory generate > "$work_dir/out1.txt" 2> "$work_dir/generate1.log"; then
  fail "первый запуск генерации завершился с ошибкой"
  cat "$work_dir/generate1.log"
elif ! make --no-print-directory generate > "$work_dir/out2.txt" 2> "$work_dir/generate2.log"; then
  fail "второй запуск генерации завершился с ошибкой"
  cat "$work_dir/generate2.log"
elif [ ! -s "$work_dir/out1.txt" ]; then
  fail "генерация не вывела ответ"
elif diff -q "$work_dir/out1.txt" "$work_dir/out2.txt" > /dev/null 2>&1; then
  ok "два прогона дали идентичный вывод"
else
  fail "прогоны отличаются — не зафиксирован seed либо включён сэмплинг"
  diff "$work_dir/out1.txt" "$work_dir/out2.txt" | head -5 | sed 's/^/      /' || true
fi

echo
echo "4. Зависимости зафиксированы"
if [ -f ../uv.lock ] && uv lock --project .. --check; then
  ok "uv.lock на месте и соответствует pyproject.toml"
else
  fail "нет uv.lock — выполните uv sync и закоммитьте файл"
fi
if grep -qE '^ *"(torch|transformers|accelerate|numpy|pyyaml|psutil)" *,' ../pyproject.toml 2>/dev/null; then
  fail "зависимости без версий в pyproject.toml"
else
  ok "версии зависимостей указаны"
fi

echo
echo "5. Замеры разделены и есть прогрев"
if uv run --project .. --locked python -m unittest discover -s tests -v; then
  ok "таймеры, прогрев, медиана, RSS и генерация проверены тестами"
else
  fail "unit-тесты обнаружили ошибку"
fi

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Не забудьте docs/hardware.md с разбором дефектов.\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
