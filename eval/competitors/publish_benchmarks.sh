#!/usr/bin/env bash
# Собирает артефакт с сырыми прогонами сравнения и сводками калибровки.
# Usage: eval/competitors/publish_benchmarks.sh <tag> [data-dir] [packs-root]
#
# Кладёт в dist/<tag>-benchmarks.tar.gz:
#   benchmarks/*.json   сырые отчёты `eval.competitors.run` (не коммитятся в git)
#   calibration/*.json  temperature.json паков: T, ECE/Brier до и после
#   manifest.json       список файлов с размером и контрольной суммой
#   README.md           окружение и команды воспроизведения
#
# Сам релиз не создаёт — артефакт прикладывается к GitHub Release вручную.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <tag> [data-dir] [packs-root]" >&2
  exit 2
fi

tag="$1"
data_dir="${2:-data/competitors_real}"
packs_root="${3:-checkpoints/packs}"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi
python_bin="${python_bin:-python}"

[[ -d "$data_dir" ]] || {
  echo "нет каталога с прогонами: $data_dir" >&2
  exit 1
}

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
root="$stage/${tag}-benchmarks"
mkdir -p "$root/benchmarks" "$root/calibration"

count=0
for f in "$data_dir"/*.json; do
  [[ -f "$f" ]] || continue
  cp "$f" "$root/benchmarks/"
  count=$((count + 1))
done
[[ "$count" -gt 0 ]] || {
  echo "в $data_dir нет json-отчётов" >&2
  exit 1
}

if [[ -d "$packs_root" ]]; then
  for d in "$packs_root"/*/; do
    [[ -f "$d/temperature.json" ]] || continue
    cp "$d/temperature.json" "$root/calibration/$(basename "$d").json"
  done
fi
if [[ -f checkpoints/reference/temperature.json ]]; then
  cp checkpoints/reference/temperature.json "$root/calibration/reference.json"
fi

cat > "$root/README.md" <<'EOF'
# Прогоны сравнения с Clef-Flash и Laya

Сырые отчёты `python -m eval.competitors.run` и сводки температурной
калибровки паков. Числа и выводы — в `docs/competitor-benchmark.md`
репозитория (тег, к которому приложен этот архив).

## Что внутри

- `benchmarks/<spec>_pack_<participant>.json` — один прогон на пару
  «набор × участник»: `reports` по каждому участнику (accuracy, latency
  p50/p95, escalation_rate) и `questions` с разбивкой по вопросам.
- `benchmarks/<spec>_pack_<participant>_sweep.json` — кривая
  «качество против доли эскалаций» (`cascade_sweep.rows`).
- `calibration/<pack>.json` — `temperature.json`: подобранные `T`,
  метрики до/после на val и test, список голов, отклонённых гейтом
  «не навреди».
- `manifest.json` — размеры и sha256 файлов.

## Окружение

- GPU: одна карта 16 ГБ (Clef-Flash целиком в bf16 не влезает).
- Clef-Flash 9.4B — GGUF Q4_K_M в `llama-server` (llama.cpp b11430),
  поверх него `eval.competitors.release --model clef-flash --base-url`.
- Laya 0.4.15 (ModernBERT-large, 421M) — `eval.competitors.release
  --model laya --device cuda`, поставлена `uv pip install --target`.
- Наши головы: gpt2 (124M), `DECISION_MAX_LENGTH=256`, `DECISION_DEVICE=cuda`.

## Воспроизведение

```bash
CLEF_GGUF=/path/clef-q4km.gguf \
LLAMA_BIN=/path/llama-server \
LAYA_PYTHONPATH=/path/laya_pkg \
  eval/competitors/reproduce_real.sh --specs support,public,routerbench --n 200
```

Скрипт обучает и калибрует паки (если их нет), поднимает участников,
гоняет наборы и `--sweep`, складывает JSON в `OUT` (по умолчанию
`data/competitors_real`) и гасит процессы по `trap`.

Оговорки:

- `--n` — размер test; размер датасета скрипт пересчитывает сам.
- Паки обучаются на train-сплите (`--seed 0`), температура — на val;
  метрики в отчётах — на test.
- Латентность конкурентов зависит от машины и квантования; accuracy —
  нет (кроме редких расхождений argmax в Q4_K_M).
EOF

"$python_bin" - "$root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "manifest.json":
        continue
    files.append(
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
(root / "manifest.json").write_text(
    json.dumps({"files": files}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"файлов: {len(files)}")
PY

mkdir -p dist
tar -czf "dist/${tag}-benchmarks.tar.gz" -C "$stage" "${tag}-benchmarks"
echo "собрано: dist/${tag}-benchmarks.tar.gz"
