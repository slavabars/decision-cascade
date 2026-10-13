#!/usr/bin/env bash
# Серверный каскад: local-first, низкоуверенное эскалируется на внешний сервис.
#
# Поднимает «внешний» сервер (порт 8001) и каскадный (порт 8000). Каскадный
# сначала отвечает локальными головами, а вопросы с уверенностью ниже порога
# переспрашивает у внешнего. Ответ содержит usage.escalated — число эскалаций.
#
# Запуск:
#   bash examples/cascade/server_cascade.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8001}"
CASCADE_PORT="${CASCADE_PORT:-8000}"
UPSTREAM="http://127.0.0.1:${UPSTREAM_PORT}"
CASCADE="http://127.0.0.1:${CASCADE_PORT}"
THRESHOLD="${THRESHOLD:-0.9}"
LOG_UP="$(mktemp -t decision-upstream.XXXXXX.log)"
LOG_CA="$(mktemp -t decision-cascade.XXXXXX.log)"

cleanup() {
  if [[ -n "${UP_PID:-}" ]]; then kill "${UP_PID}" 2>/dev/null || true; fi
  if [[ -n "${CA_PID:-}" ]]; then kill "${CA_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

"${PYTHON}" -m uvicorn api.server.main:app --port "${UPSTREAM_PORT}" >"${LOG_UP}" 2>&1 &
UP_PID=$!

DECISION_BACKEND=cascade \
DECISION_CASCADE_FALLBACK_URL="${UPSTREAM}" \
DECISION_CASCADE_FALLBACK_MODEL=upstream-local \
DECISION_CASCADE_THRESHOLD="${THRESHOLD}" \
  "${PYTHON}" -m uvicorn api.server.main:app --port "${CASCADE_PORT}" >"${LOG_CA}" 2>&1 &
CA_PID=$!

PAYLOAD='{"state":"Не приходит выплата Stripe, теряем выручку!","questions":{"team":{"type":"choice","criteria":{"billing":"оплата","support":"интерфейс"}},"urgent":{"type":"noul","criteria":["нет спешки","срочно"]}}}'
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null -X POST "${CASCADE}/v1/systemone" \
      -H 'content-type: application/json' -d "${PAYLOAD}"; then
    break
  fi
  sleep 0.5
done

echo "== Каскадный ответ (порог уверенности ${THRESHOLD}) =="
curl -sS -X POST "${CASCADE}/v1/systemone" \
  -H 'content-type: application/json' -d "${PAYLOAD}" | "${PYTHON}" -m json.tool

echo
echo "Готово. Логи: ${LOG_UP}, ${LOG_CA}"
