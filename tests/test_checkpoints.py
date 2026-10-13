"""Тесты чекпоинтов голов: сохранение/загрузка весов и спецификаций вопросов."""
import torch

from src.decision_head.heads import DecisionHeads
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec

QUESTIONS = {
    "u": NoulSpec("urgent?", ("yes", "no")),
    "c": ChoiceSpec("who?", ("billing", "sales"), {"billing": "money", "sales": "deals"}),
    "s": ScoreSpec("how bad?", ("low", "mid", "high"), ("calm", "tense", "critical")),
}


def test_save_checkpoint_creates_file(tmp_path):
    heads = DecisionHeads(dim=3, questions=QUESTIONS, seed=1)
    path = tmp_path / "heads.pt"
    heads.save_checkpoint(path)
    assert path.exists()


def test_load_checkpoint_restores_specs_and_weights(tmp_path):
    heads = DecisionHeads(dim=3, questions=QUESTIONS, seed=5)
    for head in heads.heads.values():
        with torch.no_grad():
            head.linear.weight.zero_()
            head.linear.weight[0] = torch.tensor([1.0] + [0.0] * (head.linear.in_features - 1))
            head.linear.bias.zero_()

    path = tmp_path / "heads.pt"
    heads.save_checkpoint(path)
    restored = DecisionHeads.load_checkpoint(path)

    assert restored.questions == heads.questions

    features = torch.randn(len(QUESTIONS), 3)
    original = heads.evaluate(features)
    reloaded = restored.evaluate(features)
    assert set(original) == set(reloaded)
    for name in original:
        assert original[name] == reloaded[name]
