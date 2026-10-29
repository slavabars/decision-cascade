"""Документация не отстаёт от кода: коды ошибок API описаны в troubleshooting."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "api"
DOC = ROOT / "docs" / "troubleshooting.md"


def _api_error_codes() -> set[str]:
    codes: set[str] = set()
    for path in API.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        codes.update(re.findall(r'"code": "([A-Z][A-Z_]+)"', path.read_text()))
    return codes


def test_every_api_error_code_is_documented():
    codes = _api_error_codes()
    assert codes, "в api/ не найдено ни одного кода ошибки — сломался поиск"
    text = DOC.read_text(encoding="utf-8")
    missing = sorted(code for code in codes if f"`{code}`" not in text)
    assert not missing, f"коды ошибок не описаны в docs/troubleshooting.md: {missing}"


def test_troubleshooting_covers_known_failure_modes():
    text = DOC.read_text(encoding="utf-8")
    for needle in (
        "UNTRAINED_HEADS",
        "MODALITY_NOT_SUPPORTED",
        "BACKEND_UNAVAILABLE",
        "DECISION_CASCADE_REMOTE_ACCURACY",
        "choice_head",
        "DECISION_FEATURE_DIM",
        "/ready",
    ):
        assert needle in text, f"нет раздела про {needle}"


def test_deploy_doc_links_back():
    deploy = (ROOT / "docs" / "deploy.md").read_text(encoding="utf-8")
    troubleshooting = DOC.read_text(encoding="utf-8")
    assert "troubleshooting.md" in deploy
    assert "deploy.md" in troubleshooting
