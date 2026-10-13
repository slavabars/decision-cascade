"""Тесты спецификаций вопросов: criteria, промпт, подпись схемы, (де)сериализация."""
import torch  # noqa: F401  (держит окружение torch импортированным как в остальных тестах)

from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    ScoreSpec,
    build_prompt,
    questions_signature,
    spec_from_dict,
    spec_to_dict,
)


def test_signature_order_independent():
    a = {"u": NoulSpec("a"), "c": ChoiceSpec("b", ("x", "y"))}
    b = {"c": ChoiceSpec("b", ("x", "y")), "u": NoulSpec("a")}
    assert questions_signature(a) == questions_signature(b)


def test_signature_sensitive_to_option_order():
    a = {"c": ChoiceSpec("b", ("x", "y"))}
    b = {"c": ChoiceSpec("b", ("y", "x"))}
    assert questions_signature(a) != questions_signature(b)


def test_signature_sensitive_to_instructions():
    assert questions_signature({"u": NoulSpec("a")}) != questions_signature(
        {"u": NoulSpec("b")}
    )


def test_spec_roundtrip_all_types():
    specs = [
        NoulSpec("q", ("yes", "no")),
        ChoiceSpec("q", ("a", "b"), {"a": "A", "b": "B"}),
        ScoreSpec("q", ("low", "high"), ("L", "H")),
    ]
    for spec in specs:
        assert spec_from_dict(spec_to_dict(spec)) == spec


def test_spec_roundtrip_without_criteria():
    for spec in [NoulSpec("q"), ChoiceSpec("q", ("a", "b")), ScoreSpec("q", ("l", "h"))]:
        assert spec_from_dict(spec_to_dict(spec)) == spec


def test_build_prompt_contains_state_instruction_and_criteria():
    prompt = build_prompt(
        "state X", ChoiceSpec("Who?", ("a", "b"), {"a": "alpha", "b": "beta"})
    )
    assert "state X" in prompt
    assert "Who?" in prompt
    assert "alpha" in prompt and "beta" in prompt


def test_build_prompt_score_lists_levels():
    prompt = build_prompt("s", ScoreSpec("How bad?", ("calm", "extreme")))
    assert "calm" in prompt and "extreme" in prompt


def test_build_prompt_noul_criteria():
    prompt = build_prompt("s", NoulSpec("Urgent?", ("yes it is", "no it is not")))
    assert "yes it is" in prompt and "no it is not" in prompt
