#!/usr/bin/env bash
# Запись демо-партии в GIF: asciinema -> agg.
#
# Оба инструмента нужны как бинарники (в apt их может не быть):
#   https://github.com/asciinema/asciinema/releases
#   https://github.com/asciinema/agg/releases
# Stockfish — тоже бинарник, ставится в .tools/ скриптом fetch_engine.sh.
#
# Запуск:
#   ENGINE=.tools/stockfish PACK=checkpoints/packs/chess \
#     bash examples/games/chess/record.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON="${PYTHON:-.venv/bin/python}"
ENGINE="${ENGINE:-.tools/stockfish}"
PACK="${PACK:-checkpoints/packs/chess}"
OUT="${OUT:-examples/games/chess/demo.gif}"
CAST="${CAST:-data/chess/demo.cast}"
DELAY="${DELAY:-0.45}"
PLIES="${PLIES:-64}"

ASCIINEMA="${ASCIINEMA:-$(command -v asciinema || echo .tools/asciinema)}"
AGG="${AGG:-$(command -v agg || echo .tools/agg)}"

for tool in "$ASCIINEMA" "$AGG"; do
  [[ -x "$tool" ]] || {
    echo "нет $tool — скачайте бинарник (см. шапку скрипта)" >&2
    exit 1
  }
done
[[ -x "$ENGINE" ]] || {
  echo "нет движка $ENGINE — запустите examples/games/chess/fetch_engine.sh" >&2
  exit 1
}

mkdir -p "$(dirname "$CAST")" "$(dirname "$OUT")"

"$ASCIINEMA" rec --overwrite --quiet \
  --title "decision-cascade: шахматы, need_deep решает, когда думать" \
  --idle-time-limit 1 \
  --command "$PYTHON -m examples.games.chess.play \
    --engine '$ENGINE' --pack '$PACK' --demo \
    --max-plies $PLIES --delay $DELAY \
    --pgn data/chess/demo-game.pgn" \
  "$CAST"

# github-dark + DejaVu: ближняя к терминалу пара, есть в системе.
"$AGG" --theme github-dark --font-family "DejaVu Sans Mono" --font-size 15 \
  --rows 18 --cols 74 --speed 1.6 --idle-time-limit 1 --last-frame-duration 4 \
  "$CAST" "$OUT"

echo "GIF  -> $OUT"
echo "cast -> $CAST"
