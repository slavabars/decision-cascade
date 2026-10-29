#!/usr/bin/env bash
## Бэкап состояния decision-cascade: .env, паки голов и риск-модель.
## Запуск: sudo bash deploy/backup.sh [project-dir]
## Архив кладётся в /var/backups/decision-cascade (DECISION_BACKUP_ROOT).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

PROJECT_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"

require_root
archive="$(backup_state "$PROJECT_DIR")"
echo "backup: $archive"
echo "restore: sudo bash deploy/restore.sh $archive $PROJECT_DIR"
