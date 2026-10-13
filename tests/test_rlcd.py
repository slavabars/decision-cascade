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