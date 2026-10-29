"""Тесты deploy-модуля: синтаксис bash, гейт дистрибутива, юнит-шаблон."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
SH_FILES = sorted(
    p for p in DEPLOY.rglob("*.sh") if "vendor" not in p.parts
)


@pytest.mark.parametrize("script", SH_FILES, ids=lambda p: p.name)
def test_bash_syntax(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SH_FILES, ids=lambda p: p.name)
def test_shellcheck_clean(script: Path):
    if not shutil.which("shellcheck"):
        pytest.skip("shellcheck not installed")
    subprocess.run(["shellcheck", "-x", str(script)], check=True)


def test_common_drops_to_unsupported_distro():
    common = (DEPLOY / "lib/common.sh").read_text()
    assert "detect_distro" in common
    assert "/etc/os-release" in common
    assert "die" in common


def test_distro_scripts_gate_on_distro():
    for name in ("ubuntu", "centos"):
        script = (DEPLOY / name / "install.sh").read_text()
        assert "distro_pkg_install" in script
        assert "common.sh" in script


def test_systemd_unit_template():
    unit = (DEPLOY / "systemd" / "decision-api@.service").read_text()
    assert "[Unit]" in unit and "[Service]" in unit and "[Install]" in unit
    assert "uvicorn" in unit and "api.server.main:app" in unit
    assert "8080" in unit
    assert "%i" in unit
    # Каталог и пользователь подставляет установщик: статический /opt ломал
    # установку из любого другого каталога.
    assert "/opt/decision-model" not in unit
    assert "__PROJECT_DIR__" in unit
    assert "__SERVICE_USER__" in unit and "__SERVICE_GROUP__" in unit


def test_systemd_unit_shuts_down_gracefully():
    unit = (DEPLOY / "systemd" / "decision-api@.service").read_text()
    # SIGTERM + запас времени: uvicorn доводит запросы, которые уже принял.
    assert "KillSignal=SIGTERM" in unit
    assert "TimeoutStopSec=30" in unit
    assert "--timeout-graceful-shutdown 30" in unit
    assert "Restart=on-failure" in unit


def test_requirements_have_uvicorn():
    req = (ROOT / "requirements.txt").read_text()
    assert "uvicorn" in req


# --- паритет дистрибутивов и upgrade/rollback ---------------------------------

COMMON = DEPLOY / "lib" / "common.sh"


def _bash(script: str, *args: str, env: dict[str, str] | None = None):
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", "-c", script, "bash", *args],
        check=False,
        capture_output=True,
        text=True,
        env=full_env,
    )


@pytest.mark.parametrize(
    ("os_release", "expected"),
    [
        ('ID=ubuntu\nID_LIKE=debian\n', "ubuntu"),
        ('ID=debian\nID_LIKE=debian\n', "ubuntu"),
        ('ID=centos\nID_LIKE="rhel fedora"\n', "centos"),
        ('ID=rocky\nID_LIKE="rhel centos fedora"\n', "centos"),
        ('ID=almalinux\nID_LIKE="rhel centos"\n', "centos"),
    ],
)
def test_distro_parity_from_release_file(tmp_path, os_release, expected):
    release = tmp_path / "os-release"
    release.write_text(os_release, encoding="utf-8")
    result = _bash(
        'source "$1"; distro_from_release "$2"', str(COMMON), str(release)
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_distro_parity_rejects_unknown_release(tmp_path):
    release = tmp_path / "os-release"
    release.write_text("ID=arch\n", encoding="utf-8")
    result = _bash(
        'source "$1"; distro_from_release "$2"', str(COMMON), str(release)
    )
    assert result.returncode != 0
    assert "unsupported distro" in result.stderr


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "checkpoints" / "packs" / "public-gpt2").mkdir(parents=True)
    (project / ".env").write_text("DECISION_BACKEND=local\n", encoding="utf-8")
    (project / "checkpoints" / "packs" / "public-gpt2" / "heads.pt").write_bytes(b"pt")
    (project / "checkpoints" / "risk.json").write_text("{}\n", encoding="utf-8")
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\necho 0.13.0\n", encoding="utf-8")
    python.chmod(0o755)
    return project


def test_backup_and_restore_roundtrip(tmp_path):
    project = _make_project(tmp_path)
    backups = tmp_path / "backups"
    env = {"DECISION_BACKUP_ROOT": str(backups)}
    backup = _bash(
        'source "$1"; backup_state "$2"', str(COMMON), str(project), env=env
    )
    assert backup.returncode == 0, backup.stderr
    archive = Path(backup.stdout.strip().splitlines()[-1])
    assert archive.is_file() and archive.parent == backups

    # портим состояние: .env, паки и риск-модель
    (project / ".env").write_text("BROKEN=1\n", encoding="utf-8")
    shutil.rmtree(project / "checkpoints" / "packs")
    (project / "checkpoints" / "risk.json").unlink()

    restored = _bash(
        'source "$1"; restore_state "$2" "$3"',
        str(COMMON),
        str(project),
        str(archive),
    )
    assert restored.returncode == 0, restored.stderr
    assert (project / ".env").read_text(encoding="utf-8") == "DECISION_BACKEND=local\n"
    assert (project / "checkpoints" / "packs" / "public-gpt2" / "heads.pt").is_file()
    assert (project / "checkpoints" / "risk.json").is_file()
    assert (project / ".installed-version").read_text().strip() == "0.13.0"
    assert "version in backup: 0.13.0" in restored.stdout


def test_backup_skips_missing_items(tmp_path):
    project = tmp_path / "bare"
    project.mkdir()
    env = {"DECISION_BACKUP_ROOT": str(tmp_path / "backups")}
    result = _bash(
        'source "$1"; backup_state "$2"', str(COMMON), str(project), env=env
    )
    assert result.returncode == 0, result.stderr
    archive = Path(result.stdout.strip().splitlines()[-1])
    assert archive.is_file()
    listing = subprocess.run(
        ["tar", "-tzf", str(archive)], capture_output=True, text=True, check=True
    )
    assert ".installed-version" in listing.stdout
    assert ".env" not in listing.stdout


def test_restore_rejects_missing_archive(tmp_path):
    project = _make_project(tmp_path)
    result = _bash(
        'source "$1"; restore_state "$2" "$3"',
        str(COMMON),
        str(project),
        str(tmp_path / "nope.tar.gz"),
    )
    assert result.returncode != 0
    assert "no such backup" in result.stderr


def test_upgrade_and_restore_use_shared_helpers():
    upgrade = (DEPLOY / "upgrade.sh").read_text()
    restore = (DEPLOY / "restore.sh").read_text()
    backup = (DEPLOY / "backup.sh").read_text()
    for name, script in (
        ("upgrade", upgrade),
        ("restore", restore),
        ("backup", backup),
    ):
        assert "lib/common.sh" in script, name
    assert "backup_state" in upgrade and "restore_state" in upgrade
    assert "wait_ready" in upgrade and "ensure_venv" in upgrade
    assert "restore_state" in restore and "wait_ready" in restore
    assert "backup_state" in backup

# --- установка: юнит под фактический каталог и пользователя -------------------


def test_install_scripts_prepare_project_dir_and_user():
    for name in ("ubuntu", "centos"):
        script = (DEPLOY / name / "install.sh").read_text()
        assert "DECISION_INSTALL_DIR" in script
        assert "ensure_project_dir" in script
        assert "ensure_service_user" in script
        assert 'install_systemd_unit' in script
        # юнит получает каталог и пользователя, а не /opt по умолчанию
        assert '"$INSTALL_DIR" "$SERVICE_USER" "$SERVICE_USER"' in script


def test_render_unit_substitutes_paths_and_user(tmp_path):
    dest = tmp_path / "decision-api@.service"
    project = tmp_path / "opt" / "decision-model"
    result = _bash(
        'source "$1"; render_unit "$2" "$3" "svc" "svcgrp" "$4"',
        str(COMMON),
        str(DEPLOY / "systemd" / "decision-api@.service"),
        str(project),
        str(dest),
    )
    assert result.returncode == 0, result.stderr
    unit = dest.read_text()
    assert f"WorkingDirectory={project}" in unit
    assert f"ExecStart={project}/.venv/bin/uvicorn" in unit
    assert "User=svc" in unit and "Group=svcgrp" in unit
    assert "__" not in unit
    # шаблонные подстановки не съедают спецификаторы systemd
    assert "%i" in unit and "8080" in unit


def test_render_unit_rejects_pipe_in_path(tmp_path):
    dest = tmp_path / "unit"
    result = _bash(
        'source "$1"; render_unit "$2" "/opt/a|b" "u" "g" "$3"',
        str(COMMON),
        str(DEPLOY / "systemd" / "decision-api@.service"),
        str(dest),
    )
    assert result.returncode != 0
    assert "must not contain" in result.stderr


def _git_init(path: Path) -> None:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "init"],
        check=True,
        env={**os.environ, **env},
    )


def test_ensure_project_dir_clones_committed_state(tmp_path):
    src = tmp_path / "checkout"
    src.mkdir()
    (src / "README.md").write_text("hello\n", encoding="utf-8")
    _git_init(src)
    # незакоммиченный файл не должен уехать на сервер
    (src / "local-only.txt").write_text("x\n", encoding="utf-8")

    dest = tmp_path / "opt" / "decision-model"
    result = _bash(
        'source "$1"; ensure_project_dir "$2" "$3"',
        str(COMMON),
        str(src),
        str(dest),
    )
    assert result.returncode == 0, result.stderr
    assert (dest / "README.md").is_file()
    assert (dest / ".git").is_dir()
    assert not (dest / "local-only.txt").exists()
    assert "installed commit:" in result.stdout


def test_ensure_project_dir_refuses_dirty_checkout(tmp_path):
    src = tmp_path / "checkout"
    src.mkdir()
    (src / "README.md").write_text("hello\n", encoding="utf-8")
    _git_init(src)
    (src / "README.md").write_text("changed\n", encoding="utf-8")

    result = _bash(
        'source "$1"; ensure_project_dir "$2" "$3"',
        str(COMMON),
        str(src),
        str(tmp_path / "dest"),
    )
    assert result.returncode != 0
    assert "dirty" in result.stderr


def test_ensure_project_dir_copies_tree_without_git(tmp_path):
    src = tmp_path / "tarball"
    (src / "checkpoints").mkdir(parents=True)
    (src / ".venv" / "bin").mkdir(parents=True)
    (src / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (src / "checkpoints" / "heads.pt").write_bytes(b"pt")
    (src / ".venv" / "bin" / "python").write_text("x", encoding="utf-8")

    dest = tmp_path / "dest"
    result = _bash(
        'source "$1"; ensure_project_dir "$2" "$3"',
        str(COMMON),
        str(src),
        str(dest),
    )
    assert result.returncode == 0, result.stderr
    assert (dest / "requirements.txt").is_file()
    # venv и локальное состояние не копируем — они пересобираются/приезжают бэкапом
    assert not (dest / ".venv").exists()
    assert not (dest / "checkpoints").exists()


def test_ensure_project_dir_refuses_non_empty_destination(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("x", encoding="utf-8")

    result = _bash(
        'source "$1"; ensure_project_dir "$2" "$3"',
        str(COMMON),
        str(src),
        str(dest),
    )
    assert result.returncode != 0
    assert "not empty" in result.stderr


def test_ensure_project_dir_same_dir_is_noop(tmp_path):
    result = _bash(
        'source "$1"; ensure_project_dir "$2" "$2"',
        str(COMMON),
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "same as checkout" in result.stdout
