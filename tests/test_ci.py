"""Гейты CI не должны тихо исчезнуть из workflow-файлов."""
from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def test_ci_keeps_quality_gates():
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    for gate in (
        "ruff check",
        "shellcheck",
        "uv lock --check",
        "pip-audit",
        "pytest",
        "--cov=src",
        "smoke_wheel.sh",
    ):
        assert gate in ci, f"в CI пропал гейт: {gate}"


def test_nightly_workflow_runs_quality_regression():
    nightly = (WORKFLOWS / "nightly.yml").read_text(encoding="utf-8")
    assert "schedule:" in nightly
    assert "eval.nightly" in nightly
    assert "*-weights.tar.gz" in nightly
    assert "upload-artifact" in nightly


def test_workflows_do_not_use_sudo():
    for path in WORKFLOWS.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert "sudo " not in text, f"{path.name} требует root"
