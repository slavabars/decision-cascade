#!/usr/bin/env bash
# Шлюз: наш сервер проксирует к внешнему SystemOne-сервису (DECISION_BACKEND=http).
#
# Поднимает два сервера: «внешний» (порт 8001) и шлюз (порт 8000), который
# проксирует к нему. Так один и тот же протокол подключает любую внешнюю модель
# (Clef/Clef-Flash, vLLM, SGLang) — достаточно поменять DECISION_BACKEND_URL.
#
# Запуск:
#   bash examples/protocol/gateway.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
UPSTREAM_PORT="${UPSTREAM_PORT:-8001}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
UPSTREAM="http://127.0.0.1:${UPSTREAM_PORT}"
GATEWAY="http://127.0.0.1:${GATEWAY_PORT}"
LOG_UP="$(mktemp -t decision-upstream.XXXXXX.log)"
LOG_GW="$(mktemp -t decision-gateway.XXXXXX.log)"

cleanup() {
  if [[ -n "${UP_PID:-}" ]]; then kill "${UP_PID}" 2>/dev/null || true; fi
  if [[ -n "${GW_PID:-}" ]]; then kill "${GW_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT

echo "== Апстрим на ${UPSTREAM} =="
"${PYTHON}" -m uvicorn api.server.main:app --port "${UPSTREAM_PORT}" >"${LOG_UP}" 2>&1 &
UP_PID=$!

echo "== Шлюз на ${GATEWAY} -> ${UPSTREAM} =="
DECISION_BACKEND=http \
DECISION_BACKEND_URL="${UPSTREAM}" \
DECISION_BACKEND_MODEL=upstream-local \
DECISION_BACKEND_MODALITIES=text \
  "${PYTHON}" -m uvicorn api.server.main:app --port "${GATEWAY_PORT}" >"${LOG_GW}" 2>&1 &
GW_PID=$!

PAYLOAD='{"state":"Сервис недоступен, нужна помощь","questions":{"team":{"type":"choice","criteria":{"billing":"оплата","support":"интерфейс"}}}}'
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null -X POST "${GATEWAY}/v1/systemone" \
      -H 'content-type: application/json' -d "${PAYLOAD}" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

echo
echo "== POST ${GATEWAY}/v1/systemone (проксируется на апстрим) =="
curl -sS -X POST "${GATEWAY}/v1/systemone" \
  -H 'content-type: application/json' -d "${PAYLOAD}" | "${PYTHON}" -m json.tool

echo
echo "Готово. Логи: ${LOG_UP}, ${LOG_GW}"
