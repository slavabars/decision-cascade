#!/usr/bin/env bash
## Общий код деплоя для Ubuntu (apt) и CentOS/RHEL-семейства (dnf).
## Shellcheck-clean, паритет обеих дистрибутивов, гейт /etc/os-release.

die() {
    echo "ERROR: $*" >&2
    exit 1
}

## distro_from_release <os-release> — печатает ubuntu|centos, иначе падает.
## Вынесено из detect_distro, чтобы паритет дистрибутивов проверялся тестом
## на временных файлах, а не только на машине разработчика.
distro_from_release() {
    local file="$1"
    local id id_like label
    [ -r "$file" ] || die "cannot read $file"
    # shellcheck disable=SC1090
    id="$(. "$file" && printf '%s' "${ID:-}")"
    # shellcheck disable=SC1090
    id_like="$(. "$file" && printf '%s' "${ID_LIKE:-}")"
    label="$id $id_like"
    case "$label" in
        *ubuntu* | *debian*)
            echo "ubuntu"
            ;;
        *rhel* | *centos* | *rocky* | *almalinux* | *fedora*)
            echo "centos"
            ;;
        *)
            die "unsupported distro: ID=${id:-} ID_LIKE=${id_like:-} (only Ubuntu/CentOS-family)"
            ;;
    esac
}

detect_distro() {
    distro_from_release "${DECISION_OS_RELEASE:-/etc/os-release}"
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

## ---------------------------------------------------------------------------
## Upgrade / rollback: состояние сервиса — .env, паки голов и риск-модель.
## Код восстанавливается через git, веса энкодера — из релиза (см. docs/deploy.md).
## ---------------------------------------------------------------------------

#: Куда складывать резервные копии; переопределяется для тестов и нестандартных схем.
BACKUP_ROOT="${DECISION_BACKUP_ROOT:-/var/backups/decision-cascade}"
#: Что входит в бэкап. Полные чекпоинты и fine-tune-веса не берём: они большие
#: и воспроизводятся из релиза/рецепта.
BACKUP_ITEMS=(".env" "checkpoints/packs" "checkpoints/risk.json")

## installed_version <project-dir> — версия из .venv или "unknown".
installed_version() {
    local project_dir="$1"
    local python="$project_dir/.venv/bin/python"
    if [ -x "$python" ]; then
        "$python" -c 'import src; print(src.__version__)' 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

## _stage_backup <staging> <project-dir> — копирует состояние и версию в staging.
_stage_backup() {
    local staging="$1"
    local project_dir="$2"
    local item version
    for item in "${BACKUP_ITEMS[@]}"; do
        [ -e "$project_dir/$item" ] || continue
        mkdir -p "$staging/$(dirname "$item")" || return 1
        cp -a "$project_dir/$item" "$staging/$item" || return 1
    done
    version="$(installed_version "$project_dir")"
    printf '%s\n' "$version" > "$staging/.installed-version" || return 1
}

## backup_state <project-dir> — архивирует .env и паки, печатает путь к архиву.
backup_state() {
    local project_dir="$1"
    local stamp archive staging
    [ -d "$project_dir" ] || die "no such project dir: $project_dir"
    mkdir -p "$BACKUP_ROOT" || die "cannot create backup root: $BACKUP_ROOT"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive="$BACKUP_ROOT/decision-cascade-$stamp.tar.gz"
    staging="$(mktemp -d)"
    if ! _stage_backup "$staging" "$project_dir"; then
        rm -rf "$staging"
        die "cannot stage backup from $project_dir"
    fi
    if ! tar -czf "$archive" -C "$staging" .; then
        rm -rf "$staging"
        die "cannot write backup: $archive"
    fi
    rm -rf "$staging"
    echo "$archive"
}

## restore_state <project-dir> <archive> — возвращает .env и паки из бэкапа.
restore_state() {
    local project_dir="$1"
    local archive="$2"
    [ -f "$archive" ] || die "no such backup: $archive"
    [ -d "$project_dir" ] || die "no such project dir: $project_dir"
    tar -tzf "$archive" >/dev/null 2>&1 || die "backup is not readable: $archive"
    tar -xzf "$archive" -C "$project_dir" --strip-components=1 \
        || die "cannot restore from $archive"
    echo "restored: $archive -> $project_dir"
    if [ -f "$project_dir/.installed-version" ]; then
        echo "version in backup: $(cat "$project_dir/.installed-version")"
    fi
}

## restart_service [instance] — перезапускает systemd-юнит, если он есть.
restart_service() {
    local instance="${1:-api}"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl restart "decision-api@${instance}.service" \
            || die "failed to restart decision-api@${instance}.service"
        echo "service restarted: decision-api@${instance}.service"
    else
        echo "systemd not available; restart the API process manually"
    fi
}

## wait_ready [url] [seconds] — ждёт 200 от /ready; 0 — дождались, 1 — нет.
wait_ready() {
    local url="${1:-http://127.0.0.1:8080/ready}"
    local timeout="${2:-60}"
    local waited=0
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl not installed; skipping readiness wait"
        return 0
    fi
    while [ "$waited" -lt "$timeout" ]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "ready: $url"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "not ready after ${timeout}s: $url" >&2
    return 1
}
