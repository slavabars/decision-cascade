#!/usr/bin/env bash
## Обновление decision-cascade: бэкап -> git pull -> venv -> restart -> /ready.
## Запуск: sudo bash deploy/upgrade.sh [project-dir]
##
## Работает одинаково на Ubuntu и CentOS/RHEL: пакеты не ставит (это делает
## install.sh), состояние берёт из общего deploy/lib/common.sh. Если сервис не
## стал готовым, откатывает и данные, и код на предыдущий коммит.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

PROJECT_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
INSTANCE="${DECISION_INSTANCE:-api}"
READY_URL="${DECISION_READY_URL:-http://127.0.0.1:8080/ready}"
READY_TIMEOUT="${DECISION_READY_TIMEOUT:-60}"

require_root
[ -d "$PROJECT_DIR" ] || die "no such project dir: $PROJECT_DIR"

old_rev=""
if [ -d "$PROJECT_DIR/.git" ]; then
    git -C "$PROJECT_DIR" diff --quiet \
        || die "working tree is dirty; commit or stash changes before upgrade"
    old_rev="$(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
fi
old_version="$(installed_version "$PROJECT_DIR")"

archive="$(backup_state "$PROJECT_DIR")"
echo "backup: $archive"
echo "version before: $old_version${old_rev:+ ($old_rev)}"

if [ -n "$old_rev" ]; then
    git -C "$PROJECT_DIR" fetch --tags --prune || die "git fetch failed"
    git -C "$PROJECT_DIR" pull --ff-only || die "git pull --ff-only failed; resolve manually"
else
    echo "no git repo at $PROJECT_DIR; skipping code update"
fi

ensure_venv "$PROJECT_DIR"
restart_service "$INSTANCE"

if wait_ready "$READY_URL" "$READY_TIMEOUT"; then
    echo "upgraded: $old_version -> $(installed_version "$PROJECT_DIR")"
    exit 0
fi

echo "upgrade failed: restoring data and code from $archive" >&2
restore_state "$PROJECT_DIR" "$archive"
if [ -n "$old_rev" ]; then
    git -C "$PROJECT_DIR" checkout --detach "$old_rev" \
        || die "cannot checkout $old_rev; restore code manually"
    ensure_venv "$PROJECT_DIR"
fi
restart_service "$INSTANCE"
wait_ready "$READY_URL" "$READY_TIMEOUT" \
    || die "rollback finished but service is still not ready"
echo "rolled back to $old_version${old_rev:+ ($old_rev)}"
exit 1
