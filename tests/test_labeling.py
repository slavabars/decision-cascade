"""Слабая разметка: метки учителя -> датасет, который принимает train_heads."""
from __future__ import annotations

import json

import pytest

from src.backends.base import BackendResult
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec
from src.schemas import ChoiceResult, NoulResult, ScoreResult
from training.labeling import run


def _specs() -> dict[str, NoulSpec | ChoiceSpec | ScoreSpec]:
    return {
        "urgent": NoulSpec(instructions="urgent?"),
        "team": ChoiceSpec(
            instructions="which team?",
            options=("billing", "technical", "sales"),
        ),
        "severity": ScoreSpec(
            instructions="how severe?",
            levels=("low", "medium", "high"),
        ),
    }


def _teacher(
    *,
    urgent: float = 0.9,
    team: str = "billing",
    confidence: float = 0.9,
):
    """Учитель, который отвечает по типам вопросов и задаёт уверенность."""

    def decide(state: str, specs) -> BackendResult:
        decisions = {}
        for name, spec in specs.items():
            if isinstance(spec, NoulSpec):
                decisions[name] = NoulResult(noul=urgent, confidence=confidence)
            elif isinstance(spec, ChoiceSpec):
                choice = team if team in spec.options else spec.options[0]
                rest = (1.0 - 0.7) / max(len(spec.options) - 1, 1)
                probabilities = {
                    option: (0.7 if option == choice else rest)
                    for option in spec.options
                }
                decisions[name] = ChoiceResult(
                    choice=choice,
                    probabilities=probabilities,
                    confidence=confidence,
                )
            else:
                # наиболее вероятный — последний уровень; метка = его индекс
                rest = (1.0 - 0.7) / max(len(spec.levels) - 1, 1)
                probabilities = {
                    level: (0.7 if index == len(spec.levels) - 1 else rest)
                    for index, level in enumerate(spec.levels)
                }
                decisions[name] = ScoreResult(
                    score=float(len(spec.levels) - 1),
                    probabilities=probabilities,
                    confidence=confidence,
                    legend=spec.levels,
                )
        return BackendResult(decisions=decisions, usage={"trained": True})

    return decide


def test_label_conversion_for_each_type():
    specs = _specs()
    decisions = _teacher()("x", specs).decisions
    assert run._label(specs["urgent"], decisions["urgent"], "urgent") == 1
    assert run._label(specs["team"], decisions["team"], "team") == "billing"
    assert run._label(specs["severity"], decisions["severity"], "severity") == 2


def test_noul_threshold_is_half():
    specs = _specs()
    assert run._label(specs["urgent"], NoulResult(noul=0.49), "urgent") == 0
    assert run._label(specs["urgent"], NoulResult(noul=0.5), "urgent") == 1


def test_score_falls_back_to_rounded_value_without_probabilities():
    specs = _specs()
    decision = ScoreResult(score=1.4, probabilities={})
    assert run._label(specs["severity"], decision, "severity") == 1
    decision = ScoreResult(score=9.0, probabilities={})
    assert run._label(specs["severity"], decision, "severity") == 2


def test_unknown_choice_is_an_error():
    specs = _specs()
    with pytest.raises(ValueError, match="expected one of"):
        run._label(specs["team"], ChoiceResult(choice="nope"), "team")


def test_label_examples_builds_training_rows():
    rows = [
        {"id": "a", "state": "one"},
        {"id": "b", "state": "two"},
    ]
    examples, report = run.label_examples(rows, _specs(), _teacher(), teacher="clef")
    assert [example["id"] for example in examples] == ["a", "b"]
    assert examples[0]["labels"] == {"urgent": 1, "team": "billing", "severity": 2}
    assert examples[0]["teacher"] == "clef"
    assert examples[0]["teacher_confidence"]["urgent"] == 0.9
    assert report["input"] == 2
    assert report["kept"] == 2
    assert report["dropped"] == 0
    assert report["mean_confidence"]["team"] == pytest.approx(0.9)
    assert report["label_distribution"]["team"] == {"billing": 2}


def test_min_confidence_drops_low_confidence_rows():
    rows = [{"id": "a", "state": "one"}]
    examples, report = run.label_examples(
        rows,
        _specs(),
        _teacher(confidence=0.4),
        min_confidence=0.5,
    )
    assert examples == []
    assert report["dropped"] == 1


def test_missing_answer_is_an_error():
    def decide(state, specs):
        return BackendResult(decisions={"urgent": NoulResult(noul=1.0)})

    with pytest.raises(ValueError, match="did not answer"):
        run.label_examples([{"id": "a", "state": "x"}], _specs(), decide)


def test_read_states_supports_jsonl_text_and_plain_lines(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        '{"id": "1", "state": "json state"}\n'
        '{"text": "text field"}\n'
        '"a json string"\n'
        "plain line\n"
        "\n",
        encoding="utf-8",
    )
    rows = run.read_states(path)
    assert [row["state"] for row in rows] == [
        "json state",
        "text field",
        "a json string",
        "plain line",
    ]
    assert rows[0]["id"] == "1"


def test_read_states_rejects_object_without_state(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no non-empty"):
        run.read_states(path)


def test_cli_writes_dataset_and_report(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"id": "a", "state": "one"}\n', encoding="utf-8")
    out = tmp_path / "weak.jsonl"
    report_path = tmp_path / "weak.report.json"

    report = run.main(
        [
            "--input",
            str(source),
            "--out",
            str(out),
            "--specs",
            "support",
            "--min-confidence",
            "0.5",
        ],
        decide=_teacher(),
        teacher="clef",
    )

    assert report["kept"] == 1
    example = json.loads(out.read_text(encoding="utf-8").strip())
    assert example["state"] == "one"
    assert example["teacher"] == "clef"
    assert set(example["labels"]) == set(run.load_specs("support"))
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["questions"]
    assert saved["specs"] == "support"


def test_cli_requires_teacher_without_injected_decide(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"state": "one"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="--teacher-url"):
        run.main(
            ["--input", str(source), "--out", str(tmp_path / "out.jsonl")],
        )


def test_cli_rejects_empty_input(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no states"):
        run.main(
            ["--input", str(source), "--out", str(tmp_path / "out.jsonl")],
            decide=_teacher(),
        )


def test_labels_match_specs_for_train_heads(tmp_path):
    """Формат меток — ровно тот, что понимает train_benchmark."""
    source = tmp_path / "input.txt"
    source.write_text("one\ntwo\n", encoding="utf-8")
    out = tmp_path / "weak.jsonl"
    run.main(
        ["--input", str(source), "--out", str(out), "--specs", "support"],
        decide=_teacher(),
        teacher="clef",
    )
    specs = run.load_specs("support")
    for line in out.read_text(encoding="utf-8").splitlines():
        example = json.loads(line)
        assert set(example["labels"]) == set(specs)
        for name, spec in specs.items():
            value = example["labels"][name]
            if isinstance(spec, NoulSpec):
                assert value in (0, 1)
            elif isinstance(spec, ChoiceSpec):
                assert value in spec.options
            else:
                assert isinstance(value, int)
                assert 0 <= value < len(spec.levels)
