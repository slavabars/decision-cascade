"""Конверсии протокола в адаптере релизов конкурентов (Laya, Clef-Flash).

Сам адаптер требует весов и внешних пакетов, поэтому тестируются только чистые
преобразования: ``options``/``levels`` -> ``criteria`` и обратная упаковка
ответов (``score`` по названиям уровней, ``noul`` -> ``probability``).
"""
from __future__ import annotations

import pytest

from eval.competitors.release import _criteria, _to_protocol, create_app


def test_criteria_maps_protocol_questions():
    noul = _criteria(
        {"type": "noul", "instructions": "Срочно?", "criteria": ["да", "нет"]}
    )
    assert noul == {
        "type": "noul",
        "instructions": "Срочно?",
        "criteria": {"true": "да", "false": "нет"},
    }

    choice = _criteria(
        {"type": "choice", "instructions": "Кто?", "options": ["a", "b"]}
    )
    assert choice["criteria"] == {"a": "a", "b": "b"}

    score = _criteria(
        {"type": "score", "instructions": "Насколько?", "levels": ["low", "high"]}
    )
    assert score["criteria"] == ["low", "high"]


def test_score_answer_is_keyed_by_levels():
    questions = {
        "tech": {"type": "score", "levels": ["low", "medium", "high"]},
    }
    answers = {
        "tech": {
            "type": "score",
            "score": 1.25,
            "confidence": 0.7,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35},
        }
    }
    converted = _to_protocol(answers, questions)["tech"]
    assert converted["legend"] == ["low", "medium", "high"]
    assert converted["probabilities"] == {"low": 0.1, "medium": 0.55, "high": 0.35}


def test_noul_answer_normalises_key_and_confidence():
    converted = _to_protocol(
        {"risk": {"type": "noul", "noul": 0.8}},
        {"risk": {"type": "noul"}},
    )["risk"]
    assert converted["probability"] == 0.8
    assert converted["confidence"] == pytest.approx(0.6)


def test_app_wraps_engine_into_systemone_response():
    class Engine:
        model_id = "fake"
        device = "cpu"

        def answer(self, state, questions):
            return (
                _to_protocol(
                    {"q": {"type": "noul", "noul": 0.25, "confidence": 0.5}},
                    questions,
                ),
                {"input_tokens": len(state)},
            )

    from fastapi.testclient import TestClient

    client = TestClient(create_app(Engine()))
    response = client.post(
        "/v1/systemone",
        json={"state": "state", "questions": {"q": {"type": "noul"}}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake"
    assert body["answers"]["q"]["probability"] == 0.25
    assert body["usage"] == {"input_tokens": 5}
