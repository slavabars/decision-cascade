"""Согласованность релиза: версия, README, CHANGELOG и классификатор пакета.

Тест держит статус «stable» честным: если версия поднята, а README или CHANGELOG
забыли, релиз не пройдёт CI. Это дешевле, чем ловить расхождение в описании
пакета после публикации.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import src

ROOT = Path(__file__).resolve().parents[1]
VERSION = src.__version__


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), VERSION


def test_readme_status_matches_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    major = int(VERSION.split(".")[0])
    expected = "stable" if major >= 1 else "beta"
    assert f"**Статус: {expected} ({VERSION}).**" in readme


def test_changelog_has_release_section_and_link():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{VERSION}]" in changelog
    assert f"[{VERSION}]: https://github.com/slavabars/decision-cascade/" in changelog
    # Unreleased сравнивается с текущим релизом, а не с прошлым
    assert f"compare/v{VERSION}...HEAD" in changelog


def test_pyproject_classifier_matches_status():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = config["project"]["classifiers"]
    major = int(VERSION.split(".")[0])
    expected = (
        "Development Status :: 5 - Production/Stable"
        if major >= 1
        else "Development Status :: 4 - Beta"
    )
    assert expected in classifiers
    assert not any(
        item.startswith("Development Status") and item != expected
        for item in classifiers
    )


def test_stability_doc_covers_guarantees():
    text = (ROOT / "docs" / "stability.md").read_text(encoding="utf-8")
    assert "Политика депрекации" in text
    assert "schema_version" in text
    assert "Критерии выхода" in text
