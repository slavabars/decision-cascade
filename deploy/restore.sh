#!/usr/bin/env bash
## Восстановление состояния decision-cascade из бэкапа и перезапуск сервиса.
## Запуск: sudo bash deploy/restore.sh <archive> [project-dir]
## Восстанавливает .env, паки голов и риск-модель; код не трогает.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <archive> [project-dir]" >&2
    exit 2
fi

ARCHIVE="$1"
PROJECT_DIR="${2:-$(cd "$SCRIPT_DIR/.." && pwd)}"
INSTANCE="${DECISION_INSTANCE:-api}"
READY_URL="${DECISION_READY_URL:-http://127.0.0.1:8080/ready}"
READY_TIMEOUT="${DECISION_READY_TIMEOUT:-60}"

require_root
restore_state "$PROJECT_DIR" "$ARCHIVE"
restart_service "$INSTANCE"
wait_ready "$READY_URL" "$READY_TIMEOUT" \
    || die "service is not ready after restore (systemctl status decision-api@${INSTANCE}.service)"
