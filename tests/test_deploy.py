"""Тесты deploy-модуля: синтаксис bash, гейт дистрибутива, юнит-шаблон."""
from __future__ import annotations

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


def test_requirements_have_uvicorn():
    req = (ROOT / "requirements.txt").read_text()
    assert "uvicorn" in req