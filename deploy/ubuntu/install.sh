#!/usr/bin/env bash
## Установка decision-cascade на Ubuntu (apt).
## Запуск: sudo bash deploy/ubuntu/install.sh
##
## По умолчанию разворачивает закоммиченное состояние в /opt/decision-model и
## запускает сервис под системным пользователем decision. Переопределяется:
##   DECISION_INSTALL_DIR   каталог установки (по умолчанию /opt/decision-model)
##   DECISION_SERVICE_USER  пользователь сервиса (по умолчанию decision)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../lib/common.sh"

PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_DIR="${DECISION_INSTALL_DIR:-/opt/decision-model}"
if [ "$PROJECT_DIR" = "$INSTALL_DIR" ]; then
    # Ставим из рабочего дерева: сервис работает под владельцем чекаута, чтобы
    # не менять права на исходники.
    SERVICE_USER="${DECISION_SERVICE_USER:-${SUDO_USER:-root}}"
else
    SERVICE_USER="${DECISION_SERVICE_USER:-decision}"
fi

distro_pkg_install python3 python3-venv python3-pip curl git

ensure_project_dir "$PROJECT_DIR" "$INSTALL_DIR"
ensure_venv "$INSTALL_DIR"
ensure_service_user "$SERVICE_USER" "$SERVICE_USER" "$INSTALL_DIR"
install_systemd_unit \
    "$SCRIPT_DIR/../systemd/decision-api@.service" "api" \
    "$INSTALL_DIR" "$SERVICE_USER" "$SERVICE_USER"

echo "Ubuntu deploy done. App dir: $INSTALL_DIR"
echo "Liveness: curl -f http://127.0.0.1:8080/health"
echo "Readiness: curl -f http://127.0.0.1:8080/ready"
echo "Decision: curl -X POST http://127.0.0.1:8080/decisions"
