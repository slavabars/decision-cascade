#!/usr/bin/env bash
# Проверка живости сервера: /openapi.json + smoke-запрос к /decisions.
#
# Запускает сервер, ждёт готовности, проверяет схемe и делает один запрос.
# Возвращает ненулевой код, если сервер не ответил.
#
# Запуск:
#   bash examples/ops/health_check.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
PORT="${PORT:-8000}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp -t decision-health.XXXXXX.log)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

"${PYTHON}" -m uvicorn api.server.main:app --port "${PORT}" >"${LOG}" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null "${BASE}/openapi.json" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "FAIL: сервер не поднялся за 30 c; лог: ${LOG}"
  exit 1
fi
echo "OK: /openapi.json отвечает"

if curl -fsS -X POST "${BASE}/decisions" \
    -H 'content-type: application/json' \
    -d '{"state":"ping","questions":{"ok":{"type":"noul","instructions":"ping"}}}' \
    -o /dev/null; then
  echo "OK: /decisions отвечает"
else
  echo "FAIL: /decisions не ответил; лог: ${LOG}"
  exit 1
fi

echo "Всё в порядке."
