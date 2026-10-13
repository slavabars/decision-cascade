#!/usr/bin/env bash
# Протокол Jev/SystemOne: POST /v1/systemone (drop-in для Clef и др.).
#
# Поднимает сервер и делает два запроса: корректный (вопросы только через
# criteria) и с неподдерживаемой локальным бэкендом модальностью (images) —
# чтобы увидеть типизированную ошибку MODALITY_NOT_SUPPORTED.
#
# Запуск:
#   bash examples/protocol/systemone.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp -t decision-systemone.XXXXXX.log)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

"${PYTHON}" -m uvicorn api.server.main:app --port "${PORT}" >"${LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null -X POST "${BASE}/v1/systemone" \
      -H 'content-type: application/json' \
      -d '{"state":"ping","questions":{"ok":{"type":"noul","criteria":["no","yes"]}}}' 2>/dev/null; then
    break
  fi
  sleep 0.5
done

echo "== Успешный запрос (опции выводятся из criteria) =="
curl -sS -X POST "${BASE}/v1/systemone" \
  -H 'content-type: application/json' \
  -d '{
    "state": "Не приходит выплата Stripe третьи сутки, теряем выручку!",
    "questions": {
      "team": {"type": "choice",
               "criteria": {"billing": "оплата", "integration": "API",
                            "sales": "продажи", "support": "интерфейс"}},
      "urgent": {"type": "noul",
                 "criteria": ["нет спешки", "срочно, теряем деньги"]},
      "severity": {"type": "score",
                   "criteria": ["спокоен", "недоволен", "в ярости"]}
    }
  }' | "${PYTHON}" -m json.tool

echo
echo "== Неподдерживаемая модальность (локальный бэкенд — только текст) =="
curl -sS -X POST "${BASE}/v1/systemone" \
  -H 'content-type: application/json' \
  -d '{
    "state": "Скриншот ошибки во вложении",
    "images": ["error.png"],
    "questions": {"ok": {"type": "noul", "criteria": ["no", "yes"]}}
  }' | "${PYTHON}" -m json.tool
echo "(это HTTP 422 и код MODALITY_NOT_SUPPORTED — см. README)"

echo
echo "Готово. Лог сервера: ${LOG}"
