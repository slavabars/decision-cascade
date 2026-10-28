"""Тесты температурной калибровки голов (temperature scaling)."""
import torch

from src.decision_head.heads import ChoiceHead, DecisionHeads, NoulHead, ScoreHead
from src.sampler.questions import ChoiceSpec, NoulSpec, ScoreSpec
from training.recipe.calibrate_temperature import fit_temperature

QUESTIONS = {
    "u": NoulSpec("urgent?", ("yes", "no")),
    "c": ChoiceSpec("who?", ("billing", "sales"), {"billing": "money", "sales": "deals"}),
    "s": ScoreSpec("how bad?", ("low", "mid", "high"), ("calm", "tense", "critical")),
}


def _heads(dim: int = 3, seed: int = 0) -> DecisionHeads:
    return DecisionHeads(dim=dim, questions=QUESTIONS, seed=seed)


def test_default_temperature_is_one_and_forward_unchanged():
    heads = _heads()
    features = torch.randn(len(QUESTIONS), 3)

    assert heads.temperatures() == {"u": 1.0, "c": 1.0, "s": 1.0}
    for head in heads.heads.values():
        raw = head.logits(features[0].unsqueeze(0))
        assert torch.allclose(head.scaled_logits(features[0].unsqueeze(0)), raw)

    before = heads.evaluate(features)
    heads.set_temperatures({"u": 1.0, "c": 1.0, "s": 1.0})
    assert heads.evaluate(features) == before


def test_temperature_softens_probabilities_but_keeps_argmax():
    torch.manual_seed(0)
    head = ChoiceHead(3, ("a", "b", "c"))
    with torch.no_grad():
        head.linear.weight.mul_(20.0)
    features = torch.randn(1, 3)

    sharp = head(features)
    head.set_temperature(10.0)
    soft = head(features)

    assert soft.choice == sharp.choice
    assert soft.confidence < sharp.confidence
    assert abs(sum(soft.probabilities.values()) - 1.0) < 1e-6


def test_noul_and_score_heads_apply_temperature():
    features = torch.randn(1, 3)

    noul = NoulHead(3)
    with torch.no_grad():
        noul.linear.weight.mul_(10.0)
    sharp_p = noul(features).noul
    noul.set_temperature(5.0)
    soft_p = noul(features).noul
    assert abs(soft_p - 0.5) < abs(sharp_p - 0.5)

    score = ScoreHead(3, ("low", "mid", "high"))
    with torch.no_grad():
        score.linear.weight.mul_(10.0)
    sharp = score(features)
    score.set_temperature(5.0)
    soft = score(features)
    assert soft.confidence < sharp.confidence
    assert abs(sum(soft.probabilities.values()) - 1.0) < 1e-6


def test_temperature_roundtrips_through_checkpoint(tmp_path):
    heads = _heads(seed=3)
    heads.set_temperatures({"u": 2.5, "c": 4.0, "s": 0.5})
    path = tmp_path / "heads.pt"
    heads.save_checkpoint(path)

    restored = DecisionHeads.load_checkpoint(path)
    assert restored.temperatures() == {"u": 2.5, "c": 4.0, "s": 0.5}

    features = torch.randn(len(QUESTIONS), 3)
    assert restored.evaluate(features) == heads.evaluate(features)


def test_old_checkpoint_without_temperature_loads_as_one(tmp_path):
    heads = _heads(seed=7)
    path = tmp_path / "heads.pt"
    heads.save_checkpoint(path)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.pop("temperature")
    torch.save(payload, path)

    restored = DecisionHeads.load_checkpoint(path)
    assert restored.temperatures() == {"u": 1.0, "c": 1.0, "s": 1.0}


def test_set_temperature_rejects_non_positive_and_unknown_question():
    import pytest

    heads = _heads()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature"):
            heads.set_temperatures({"u": bad})
    with pytest.raises(ValueError, match="unknown questions"):
        heads.set_temperatures({"nope": 2.0})


def test_fit_temperature_softens_overconfident_noul_logits():
    # 78% положительных, но голова всегда говорит «да» с логитом +20:
    # честная вероятность — доля класса, а не 1.0.
    targets = torch.tensor([1.0] * 780 + [0.0] * 220)
    logits = torch.full((1000,), 20.0)

    t = fit_temperature(logits, targets, noul=True)

    assert t > 1.0
    softened = float(torch.sigmoid(torch.tensor(20.0) / t))
    assert 0.7 < softened < 0.85


def test_fit_temperature_is_one_when_already_calibrated():
    # Логиты уже дают ровно долю класса — оптимальная T около 1.
    targets = torch.tensor([1.0] * 800 + [0.0] * 200)
    logit = float(torch.logit(torch.tensor(0.8)))
    logits = torch.full((1000,), logit)

    assert abs(fit_temperature(logits, targets, noul=True) - 1.0) < 0.2


def test_fit_temperature_never_sharpens():
    # Идеальное разделение на val: NLL уходит в T->0, но уверенность выше
    # исходной не поднимаем — иначе одно случайное совпадение даёт 1.0.
    targets = torch.tensor([1.0] * 100 + [0.0] * 100)
    logits = torch.where(targets > 0.5, 6.0, -6.0)

    assert fit_temperature(logits, targets, noul=True) == 1.0


def test_fit_temperature_for_choice_keeps_argmax():
    # Голова всегда выбирает класс 0 (логит 8), а верна лишь в 40% случаев.
    targets = torch.cat(
        [
            torch.zeros(240, dtype=torch.long),
            torch.ones(180, dtype=torch.long),
            torch.full((180,), 2, dtype=torch.long),
        ]
    )
    logits = torch.zeros(600, 3)
    logits[:, 0] = 8.0

    t = fit_temperature(logits, targets, noul=False)

    assert t > 1.0
    assert torch.equal((logits / t).argmax(-1), logits.argmax(-1))


def test_refresh_metadata_describes_calibrated_checkpoint(tmp_path):
    import json

    from training.recipe.calibrate_temperature import _refresh_metadata

    meta = {"metrics": {"train": {"acc": 1}, "val": {"old": 1}, "test": {"old": 1}}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    _refresh_metadata(
        tmp_path,
        {"q": 2.0},
        {"overall_accuracy": 0.5},
        {"overall_accuracy": 0.4},
        {"before": {}, "after": {}},
    )

    out = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert out["temperature"] == {"q": 2.0}
    assert out["metrics"]["val"] == {"overall_accuracy": 0.5}
    assert out["metrics"]["test"] == {"overall_accuracy": 0.4}
    assert out["metrics"]["train"] == {"acc": 1}
    assert out["calibration"]["fit"] == "val"


def test_refresh_metadata_is_noop_without_metadata_file(tmp_path):
    from training.recipe.calibrate_temperature import _refresh_metadata

    _refresh_metadata(tmp_path, {"q": 2.0}, {}, None, {"before": {}, "after": {}})
    assert not (tmp_path / "metadata.json").exists()
