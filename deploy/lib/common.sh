#!/usr/bin/env bash
## Общий код деплоя для Ubuntu (apt) и CentOS/RHEL-семейства (dnf).
## Shellcheck-clean, паритет обеих дистрибутивов, гейт /etc/os-release.

die() {
    echo "ERROR: $*" >&2
    exit 1
}

detect_distro() {
    [ -r /etc/os-release ] || die "cannot read /etc/os-release"
    # shellcheck source=/dev/null
    . /etc/os-release
    local label="${ID:-} ${ID_LIKE:-}"
    case "$label" in
        *ubuntu* | *debian*)
            echo "ubuntu"
            ;;
        *rhel* | *centos* | *rocky* | *almalinux* | *fedora*)
            echo "centos"
            ;;
        *)
            die "unsupported distro: ID=${ID:-} ID_LIKE=${ID_LIKE:-} (only Ubuntu/CentOS-family)"
            ;;
    esac
}

require_root() {
    [ "$(id -u)" -eq 0 ] || die "must run as root (sudo bash $0)"
}

## distro_pkg_install <pkgs...>
distro_pkg_install() {
    local distro
    distro="$(detect_distro)"
    case "$distro" in
        ubuntu)
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq
            apt-get install -y -qq --no-install-recommends "$@"
            ;;
        centos)
            dnf install -y "$@"
            ;;
    esac
}

## install_systemd_unit <path-to-service-template> <instance>
## Копирует юнит в /etc/systemd/system и, при наличии systemd, включает сервис.
install_systemd_unit() {
    local template="$1"
    local instance="$2"
    local unit="/etc/systemd/system/decision-api@.service"
    require_root
    cp "$template" "$unit" || die "cannot copy unit to $unit (need root)"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl daemon-reload
        systemctl enable --now "decision-api@${instance}.service" \
            || die "failed to start decision-api@${instance}.service"
        echo "service started: decision-api@${instance}.service"
    else
        echo "systemd not available; unit installed at $unit"
        echo "start manually: .venv/bin/uvicorn api.server.main:app --port 8080"
    fi
}

## ensure_venv <project-dir>  — ставит/обновляет .venv из requirements.txt
ensure_venv() {
    local project_dir="$1"
    local venv_bin="$project_dir/.venv/bin"
    if [ ! -x "$venv_bin/python" ]; then
        python3 -m venv "$project_dir/.venv" || die "failed to create venv"
    fi
    "$venv_bin/python" -m pip install --quiet --upgrade pip
    "$venv_bin/python" -m pip install --quiet -r "$project_dir/requirements.txt" \
        || die "pip install failed"
    echo "venv ready: $venv_bin"
}