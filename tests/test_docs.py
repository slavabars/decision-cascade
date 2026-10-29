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


def test_observability_doc_covers_every_metric():
    from api.server import metrics

    text = (ROOT / "docs" / "observability.md").read_text(encoding="utf-8")
    for metric in (
        metrics.REQUESTS,
        metrics.LATENCY,
        metrics.DECISIONS,
        metrics.UNTRAINED_REFUSALS,
        metrics.READY,
        metrics.HEADS_LOADED,
    ):
        assert metric._name in text, f"метрика {metric._name} не описана"


def test_weak_labeling_doc_and_module_exist():
    doc = (ROOT / "docs" / "weak-labeling.md").read_text(encoding="utf-8")
    module = (ROOT / "training" / "labeling" / "run.py").read_text(encoding="utf-8")
    assert "training.labeling.run" in doc
    assert "train_heads" in doc
    assert "def label_examples" in module


def test_readme_links_new_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/observability.md" in readme
    assert "docs/weak-labeling.md" in readme
    assert "/metrics" in readme


def test_drift_doc_describes_tools_and_gate():
    text = (ROOT / "docs" / "drift.md").read_text(encoding="utf-8")
    assert "training.feedback.run" in text
    assert "eval.drift.run" in text
    assert "--control" in text
    assert "--replay" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/drift.md" in readme
    assert "training.feedback.run" in readme
