#!/usr/bin/env bash
# Smoke-тест установленного wheel: чистый venv, старт сервера, /health, /ready
# и один POST /decisions. Запускается из временного каталога вне репозитория,
# чтобы проверить именно пакет, а не исходники.
#
# Usage: scripts/smoke_wheel.sh [path/to/decision_cascade-*.whl]
#        PYTHON=python3.11 — интерпретатор для venv
#        (по умолчанию .venv/bin/python, если он есть, иначе python3)
#
# Venv создаётся с --system-site-packages: зависимости (torch, fastapi) уже
# стоят в окружении CI, а проверяем мы упаковку и запуск, а не их установку.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
fi
python_bin="${python_bin:-python3}"
wheel="${1:-}"
if [[ -z "$wheel" ]]; then
  wheel="$(find "$repo_root/dist" -maxdepth 1 -name 'decision_cascade-*.whl' 2>/dev/null | sort | tail -1)"
fi
[[ -f "$wheel" ]] || { echo "wheel not found: $wheel" >&2; exit 2; }
wheel="$(cd "$(dirname "$wheel")" && pwd)/$(basename "$wheel")"

tmp="$(mktemp -d "${TMPDIR:-/tmp}/dc-smoke.XXXXXX")"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

"$python_bin" -m venv --system-site-packages "$tmp/venv"
# Если PYTHON указывает на проектный .venv, его пакеты не видны новому venv
# через --system-site-packages (тот берёт пакеты базового интерпретатора).
# Добавляем их .pth-файлом; wheel всё равно стоит раньше в sys.path.
base_site="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
smoke_site="$("$tmp/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ "$base_site" != "$smoke_site" ]]; then
  printf '%s\n' "$base_site" >"$smoke_site/deps.pth"
fi
# Editable-install (setuptools) ставит meta-path finder, который перехватывает
# `api`/`src` раньше wheel. Для smoke он вреден: убираем такие finder'ы.
cat >"$smoke_site/sitecustomize.py" <<'PY'
import sys

sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if "editable" not in type(finder).__module__.lower()
]
PY
"$tmp/venv/bin/python" -m pip install --quiet --no-deps --force-reinstall "$wheel"

# Дальше работаем вне репозитория: иначе cwd попадёт в sys.path и smoke
# проверит исходники вместо wheel.
cd "$tmp"

# Защита от подмены: если из venv импортируется исходное дерево (editable
# install в системном окружении), smoke проверял бы не wheel.
PYTHONPATH='' "$tmp/venv/bin/python" - "$tmp" <<'PY'
import pathlib
import sys

import api.server.main as main

path = pathlib.Path(main.__file__).resolve()
root = pathlib.Path(sys.argv[1]).resolve()
if root not in path.parents:
    raise SystemExit(f"imported source tree instead of wheel: {path}")
print(f"wheel import: {path}")
PY

port="$("$tmp/venv/bin/python" - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"

PYTHONPATH='' \
DECISION_DEVICE=cpu \
DECISION_UNTRAINED=allow \
DECISION_CHECKPOINT_DIR="$tmp/no-checkpoints" \
  "$tmp/venv/bin/python" -m uvicorn api.server.main:app \
    --host 127.0.0.1 --port "$port" >"$tmp/server.log" 2>&1 &
server_pid=$!

started=0
for _ in $(seq 1 120); do
  if curl -sf --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
    started=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "$started" != "1" ]]; then
  echo "server did not start; log:" >&2
  cat "$tmp/server.log" >&2
  exit 1
fi

curl -sf --max-time 5 "http://127.0.0.1:$port/health" | grep -q '"status":"ok"'
curl -sf --max-time 5 "http://127.0.0.1:$port/ready" | grep -q '"ready":true'

response="$(curl -sf --max-time 30 -X POST "http://127.0.0.1:$port/decisions" \
  -H 'Content-Type: application/json' \
  -d '{"state":"smoke test","questions":{"urgent":{"type":"noul","instructions":"urgent?"}}}')"
printf '%s' "$response" | grep -q '"schema_version":"1"'
printf '%s' "$response" | grep -q '"urgent"'

echo "smoke ok: $wheel"
