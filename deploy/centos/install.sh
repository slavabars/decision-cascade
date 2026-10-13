#!/usr/bin/env bash
## Установка decision-cascade на CentOS/RHEL-семейство (dnf).
## Запуск: sudo bash deploy/centos/install.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../lib/common.sh"

PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

distro_pkg_install python3 python3-pip curl git

ensure_venv "$PROJECT_DIR"
install_systemd_unit "$SCRIPT_DIR/../systemd/decision-api@.service" "api"

echo "CentOS deploy done. App dir: $PROJECT_DIR"
echo "Health: curl http://127.0.0.1:8080/decisions (POST)"