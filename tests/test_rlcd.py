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