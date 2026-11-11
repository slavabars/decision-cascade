#!/usr/bin/env bash
# Запись демо-партии в лабиринте: asciinema -> agg.
#
# Нужны бинарники asciinema и agg (ссылки — в README примера), а также
# обученный пак: checkpoints/packs/maze.
#
# Запуск:
#   bash examples/games/maze/record.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-.venv/bin/python}"
PACK="${PACK:-checkpoints/packs/maze}"
OUT="${OUT:-examples/games/maze/demo.gif}"
CAST="${CAST:-data/maze/demo.cast}"
SEED="${SEED:-3}"
STEPS="${STEPS:-220}"
DELAY="${DELAY:-0.07}"

ASCIINEMA="${ASCIINEMA:-$(command -v asciinema || echo .tools/asciinema)}"
AGG="${AGG:-$(command -v agg || echo .tools/agg)}"

for tool in "$ASCIINEMA" "$AGG"; do
  [[ -x "$tool" ]] || {
    echo "нет $tool — скачайте бинарник (см. шапку скрипта)" >&2
    exit 1
  }
done
[[ -f "$PACK/metadata.json" ]] || {
  echo "нет пака $PACK — сначала examples/games/maze/train.py" >&2
  exit 1
}

mkdir -p "$(dirname "$CAST")" "$(dirname "$OUT")"

"$ASCIINEMA" rec --overwrite --quiet \
  --title "decision-cascade: лабиринт" \
  --idle-time-limit 1 \
  --command "$PYTHON -m examples.games.maze.play --pack '$PACK' \
    --seed $SEED --demo --max-steps $STEPS --delay $DELAY" \
  "$CAST"

"$AGG" --theme github-dark --font-family "DejaVu Sans Mono" --font-size 15 \
  --rows 21 --cols 56 --speed 1.6 --idle-time-limit 1 --last-frame-duration 4 \
  "$CAST" "$OUT"

echo "GIF  -> $OUT"
echo "cast -> $CAST"
