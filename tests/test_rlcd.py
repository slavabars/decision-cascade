"""Тесты RLCD — написаны до реализации (TDD RED)."""
import torch

from training.rlcd.reward import group_advantages, reward_calibration


def test_reward_calibration_aligned_is_zero():
    probs = torch.tensor([1.0, 0.0, 0.8, 0.2])
    outc = torch.tensor([1.0, 0.0, 1.0, 0.0])
    r = reward_calibration(probs, outc)
    assert abs(r[0].item()) < 1e-6 and abs(r[1].item()) < 1e-6
    assert r.shape == probs.shape


def test_reward_calibration_penalizes_mismatch():
    probs = torch.tensor([1.0, 0.0])
    outc = torch.tensor([0.0, 1.0])
    r = reward_calibration(probs, outc)
    assert (r < 0).all() and abs(r.sum().item()) == pytest.approx(2.0, abs=1e-6)


def test_group_advantages_center_and_scale():
    r = torch.tensor([1.0, 1.0, -1.0, -1.0])
    a = group_advantages(r)
    assert abs(a.mean().item()) < 1e-6 and float(a.std().item()) > 0.5


import pytest


def test_rlcd_improves_calibration():
    from training.rlcd.train import RLCDNoulTrainer

    torch.manual_seed(0)
    n = 160
    features = torch.randn(n, 2)
    outc = (features[:, 0] > 0).float()
    trainer = RLCDNoulTrainer(dim=2, lr=0.2)
    before = float((trainer.predict_probs(features) - outc).abs().mean())
    trainer.train(features, outc, steps=300)
    after = float((trainer.predict_probs(features) - outc).abs().mean())
    assert after < before
    assert after < 0.35


def test_rlcd_improves_calibration_across_seeds():
    """Регресс на вырожденный градиент: награда обязана зависеть от сэмпла.

    Если награда считается по вероятности политики, а не по взятому действию,
    то E[∇ log π] = 0 и «обучение» идёт только конечновыборочным шумом — на одном
    seed это может случайно выглядеть как улучшение. Здесь сходимость проверяется
    на нескольких seeds сразу (при вырожденном градиенте тест падает).
    """
    from training.rlcd.train import RLCDNoulTrainer

    for seed in range(5):
        torch.manual_seed(seed)
        n = 160
        features = torch.randn(n, 2)
        outc = (features[:, 0] > 0).float()
        trainer = RLCDNoulTrainer(dim=2, lr=0.2)
        before = float((trainer.predict_probs(features) - outc).abs().mean())
        trainer.train(features, outc, steps=300)
        after = float((trainer.predict_probs(features) - outc).abs().mean())
        assert after < before, f"seed {seed}: {before=} {after=}"
        assert after < 0.35, f"seed {seed}: {after=}"


def test_calibrate_noul_heads_copies_weights_into_heads():
    from src.decision_head.heads import DecisionHeads
    from src.sampler.questions import ChoiceSpec, NoulSpec
    from training.rlcd.train import calibrate_noul_heads

    torch.manual_seed(0)
    features_u = torch.randn(160, 2)
    outcome = (features_u[:, 0] > 0).float()
    specs = {"u": NoulSpec("urgent?"), "team": ChoiceSpec("team?", ("a", "b"))}
    heads = DecisionHeads(2, specs, seed=0)
    features = {"u": features_u, "team": torch.randn(160, 2)}
    targets = {"u": outcome, "team": torch.zeros(160, dtype=torch.long)}

    def mae() -> float:
        with torch.no_grad():
            p = torch.sigmoid(heads.heads["u"].logits(features_u))
            return float((p - outcome).abs().mean())

    before = mae()
    history = calibrate_noul_heads(heads, features, targets, steps=300, lr=0.2)

    assert set(history) == {"u"}
    assert mae() <= before + 1e-6


def test_calibrate_noul_heads_reverts_when_already_calibrated():
    """RLCD не должен портить уже (хорошо) откалиброванную голову."""
    from src.decision_head.heads import DecisionHeads
    from src.sampler.questions import NoulSpec
    from training.rlcd.train import calibrate_noul_heads

    torch.manual_seed(0)
    heads = DecisionHeads(2, {"u": NoulSpec("urgent?")}, seed=0)
    with torch.no_grad():
        heads.heads["u"].linear.weight.copy_(torch.tensor([[50.0, 0.0]]))
        heads.heads["u"].linear.bias.zero_()
    x = torch.randn(200, 2)
    targets = {"u": (x[:, 0] > 0).float()}
    before = heads.heads["u"].linear.weight.detach().clone()

    calibrate_noul_heads(heads, {"u": x}, targets, steps=200, lr=0.05)

    assert torch.allclose(before, heads.heads["u"].linear.weight.detach())


class _FakePolicy(torch.nn.Module):
    """Политика с нулевыми логитами (p = 0.5) и копируемыми weight/bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dim))
        self.bias = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class _FakeTrainer:
    def __init__(self, dim, lr=1e-2, g=8):
        self.policy = _FakePolicy(dim)

    def train(self, features, outcomes, steps=200, batch_size=32):
        return [0.0]


def test_calibrate_gate_uses_held_out_split(monkeypatch):
    """Гейт считается на переданном (отложенном) наборе, а не на train."""
    from src.decision_head.heads import DecisionHeads
    from src.sampler.questions import NoulSpec
    from training.rlcd import train as rlcd

    monkeypatch.setattr(rlcd, "RLCDNoulTrainer", _FakeTrainer)

    heads = DecisionHeads(2, {"u": NoulSpec("urgent?")}, seed=0)
    with torch.no_grad():
        heads.heads["u"].linear.weight.zero_()
        heads.heads["u"].linear.bias.fill_(-10.0)  # p ~ 0

    x = torch.zeros(8, 2)
    train_targets = {"u": torch.zeros(8)}  # на train p~0 идеально
    gate_features = {"u": torch.zeros(8, 2)}
    gate_targets = {"u": torch.ones(8)}  # на gate p~0 полностью неверно

    gated = rlcd.calibrate_noul_heads(
        heads,
        {"u": x},
        train_targets,
        steps=1,
        gate_features=gate_features,
        gate_targets=gate_targets,
    )
    # gate: before=|0-1|=1, after=|0.5-1|=0.5 -> принимаем -> 0.5
    assert gated["u"] == pytest.approx(0.5, abs=1e-6)

    # Свежая голова: без gate (на train) before=|0-0|=0, after=0.5 -> откат -> 0
    ungated_heads = DecisionHeads(2, {"u": NoulSpec("urgent?")}, seed=0)
    with torch.no_grad():
        ungated_heads.heads["u"].linear.weight.zero_()
        ungated_heads.heads["u"].linear.bias.fill_(-10.0)
    ungated = rlcd.calibrate_noul_heads(
        ungated_heads, {"u": x}, train_targets, steps=1
    )
    assert ungated["u"] == pytest.approx(0.0, abs=1e-3)


def test_train_benchmark_passes_gate_examples(monkeypatch):
    from api.server.service import HashStateEncoder
    from eval.benchmark.core import train_benchmark
    from eval.datasets.text_dataset import SPECS, generate_examples
    from training.rlcd import train as rlcd

    captured: dict = {}

    def fake_calibrate(
        heads,
        features,
        targets,
        *,
        steps=0,
        lr=0.01,
        g=8,
        gate_features=None,
        gate_targets=None,
    ):
        captured["gate_features"] = gate_features
        captured["gate_targets"] = gate_targets
        return {}

    monkeypatch.setattr(rlcd, "calibrate_noul_heads", fake_calibrate)

    examples = generate_examples(n=40, seed=0)
    gate = generate_examples(n=10, seed=1)
    train_benchmark(
        SPECS,
        examples,
        HashStateEncoder(16),
        train_steps=2,
        rlcd_steps=1,
        rlcd_gate_examples=gate,
    )

    assert captured["gate_features"] is not None
    assert captured["gate_targets"] is not None
    assert next(iter(captured["gate_features"].values())).shape[0] == len(gate)


def test_train_benchmark_without_gate_passes_none(monkeypatch):
    from api.server.service import HashStateEncoder
    from eval.benchmark.core import train_benchmark
    from eval.datasets.text_dataset import SPECS, generate_examples
    from training.rlcd import train as rlcd

    captured: dict = {}

    def fake_calibrate(heads, features, targets, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(rlcd, "calibrate_noul_heads", fake_calibrate)

    train_benchmark(
        SPECS,
        generate_examples(n=30, seed=0),
        HashStateEncoder(16),
        train_steps=2,
        rlcd_steps=1,
    )
    assert captured.get("gate_features") is None
    assert captured.get("gate_targets") is None


def test_calibrate_rlcd_script_runs(tmp_path):
    from training.recipe.calibrate_rlcd import main

    result = main(
        [
            "--specs",
            "support",
            "--limit",
            "60",
            "--encoder",
            "hash",
            "--dim",
            "32",
            "--steps",
            "10",
            "--out",
            str(tmp_path),
        ]
    )

    assert (tmp_path / "heads.pt").exists()
    assert (tmp_path / "calibration.json").exists()
    assert {"before", "after", "noul"} <= set(result)
    assert result["noul"]  # есть хотя бы одна noul-голова
    assert result["after"]["overall_accuracy"] >= 0.0