#!/usr/bin/env bash
# Быстрый старт: поднять сервер и прогнать запрос к /decisions.
#
# Запускает uvicorn в фоне, ждёт готовности, делает запрос и останавливает сервер.
# По умолчанию модель не загружается (hash-энкодер) — старт мгновенный.
#
# Запуск:
#   bash examples/quickstart/quickstart.sh
#
# Переменные окружения:
#   PYTHON   — интерпретатор (по умолчанию .venv/bin/python)
#   PORT     — порт сервера (по умолчанию 8000)
#   MODEL_ID — HF-модель энкодера (по умолчанию не задана)
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp -t decision-quickstart.XXXXXX.log)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "== Запуск сервера на ${BASE} =="
if [[ -n "${MODEL_ID:-}" ]]; then
  DECISION_MODEL_ID="${MODEL_ID}" \
    "${PYTHON}" -m uvicorn api.server.main:app --port "${PORT}" >"${LOG}" 2>&1 &
else
  "${PYTHON}" -m uvicorn api.server.main:app --port "${PORT}" >"${LOG}" 2>&1 &
fi
SERVER_PID=$!

echo "== Ожидание готовности =="
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null -X POST "${BASE}/decisions" \
      -H 'content-type: application/json' \
      -d '{"state":"ping","questions":{"ok":{"type":"noul","instructions":"ping"}}}' 2>/dev/null; then
    break
  fi
  sleep 0.5
done

echo
echo "== POST /decisions =="
curl -sS -X POST "${BASE}/decisions" \
  -H 'content-type: application/json' \
  -d '{
    "state": "Не приходит выплата Stripe третьи сутки, теряем выручку!",
    "questions": {
      "срочность": {"type": "noul", "instructions": "Клиент сигнализирует о срочности"},
      "команда": {"type": "choice", "instructions": "Какая команда возьмёт обращение",
                  "options": ["биллинг", "интеграция", "продажи", "общая поддержка"]},
      "накал": {"type": "score", "instructions": "Насколько клиент раздражён",
                "levels": ["спокоен", "умеренно", "сильно", "на грани"]}
    }
  }' | "${PYTHON}" -m json.tool

echo
echo "Готово. Лог сервера: ${LOG}"
