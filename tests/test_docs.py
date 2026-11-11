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


def test_observability_alert_metrics_exist_in_registry():
    """Метрики из примеров алертов должны реально существовать в реестре."""
    import re

    from prometheus_client import REGISTRY

    text = (ROOT / "docs" / "observability.md").read_text(encoding="utf-8")
    known: set[str] = set()
    for family in REGISTRY.collect():
        known.add(family.name)
        for suffix in ("_total", "_bucket", "_count", "_sum"):
            known.add(family.name + suffix)
    referenced = set(re.findall(r"\bdecision_[a-z0-9_]+\b", text))
    missing = sorted(name for name in referenced if name not in known)
    assert not missing, f"в алертах упомянуты неизвестные метрики: {missing}"


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


def test_latency_and_russian_docs_linked():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/latency.md" in readme
    assert "docs/ru-benchmark.md" in readme
    assert "eval.latency.sweep" in readme
    assert "ru_sentiment" in readme
    latency = (ROOT / "docs" / "latency.md").read_text(encoding="utf-8")
    assert "eval.latency.sweep" in latency
    russian = (ROOT / "docs" / "ru-benchmark.md").read_text(encoding="utf-8")
    assert "ru_sentiment" in russian
    assert "cointegrated/rubert-tiny2" in russian


BADGES = (
    "actions/workflows/ci.yml/badge.svg",
    "img.shields.io/pypi/v/decision-cascade",
    "img.shields.io/pypi/pyversions/decision-cascade",
    "img.shields.io/pypi/l/decision-cascade",
    "coverage_gate",
)


def test_readme_has_badges_and_language_switch():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for badge in BADGES:
        assert badge in readme, f"нет бейджа {badge}"
    assert "README.en.md" in readme


def test_english_readme_is_parity():
    en = (ROOT / "README.en.md").read_text(encoding="utf-8")
    assert "README.md" in en, "английский README должен ссылаться на русский"
    assert "pip install decision-cascade" in en
    assert "uvicorn api.server.main:app" in en
    for badge in BADGES:
        assert badge in en
    # те же числа, что и в русской таблице: расхождение — это ложь в одном из файлов
    for number in ("0.873", "0.855", "0.662", "0.542", "0.442", "0.312"):
        assert number in en


def test_quickstart_stays_near_the_top():
    """Установка не должна снова уехать за сотню строк позиционирования."""
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    head = "\n".join(lines[:120])
    assert "## Быстрый старт" in head
    assert "pip install decision-cascade" in head
    assert "curl -sS -X POST http://127.0.0.1:8000/decisions" in head
    # пример ответа в README — реальный вывод сервиса, а не выдумка
    assert '"schema_version": "1"' in head
    assert '"trained": false' in head


def test_readme_benchmark_table_matches_benchmarks_doc():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    for number in ("0.873", "0.855", "0.662", "0.542", "0.442", "0.312"):
        assert number in readme
        assert number in benchmarks, f"{number} нет в docs/benchmarks.md"
    # синтетический домен не должен попадать в таблицу публичных наборов
    top = readme.split("## Бенчмарки: публичные наборы", 1)[1].split("## Что это", 1)[0]
    table = [line for line in top.splitlines() if line.startswith("|")]
    assert table, "верхняя таблица бенчмарков не найдена"
    assert not any("0.915" in line for line in table)
    assert any("AG News" in line for line in table)
    assert any("RouterBench" in line for line in table)


def test_readme_toc_anchors_exist():
    import re

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    toc = readme.split("<summary><b>Оглавление</b></summary>", 1)[1].split("</details>", 1)[0]
    anchors = set(re.findall(r"\]\(#([^)]+)\)", toc))
    assert anchors, "оглавление не содержит ссылок"

    def slug(heading: str) -> str:
        value = heading.strip().lower()
        value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
        return value.replace(" ", "-")

    known = {
        slug(line[3:]) for line in readme.splitlines() if line.startswith("## ")
    }
    missing = sorted(anchor for anchor in anchors if anchor not in known)
    assert not missing, f"битые якоря в оглавлении: {missing}"


def test_env_example_documents_core_vars_and_has_no_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in (
        "DECISION_MODEL_ID",
        "DECISION_DEVICE",
        "DECISION_DTYPE",
        "DECISION_MAX_LENGTH",
        "DECISION_CHECKPOINT_DIR",
    ):
        assert f"{name}=" in text, f"нет {name} в .env.example"
    for secret in ("TOKEN", "PASSWORD", "SECRET", "API_KEY"):
        assert secret not in text.upper(), f"в .env.example похоже на секрет: {secret}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ".env.example" in readme


def test_info_config_is_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "GET /info" in readme
    assert "блок `config`" in readme
