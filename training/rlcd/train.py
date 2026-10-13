"""RLCD: reinforcement learning for calibrated decisions.

Прототип для Noul-головы: политика p = sigmoid(linear(features)). Внутри
группы берутся G выборок Y ~ Bernoulli(p) (GRPO-стиль), reward =
-|p - o|, преимущества центрируются по среднему батча, policy-gradient
обновление через лог-вероятность выборок.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from training.rlcd.reward import reward_calibration


class RLCDNoulTrainer:
    def __init__(self, dim: int, lr: float = 1e-2, g: int = 8):
        self.policy = nn.Linear(dim, 1)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.g = g

    def predict_probs(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.policy(features).squeeze(-1))

    def train(
        self,
        features: Tensor,
        outcomes: Tensor,
        steps: int = 200,
        batch_size: int = 32,
    ) -> list[float]:
        n = features.shape[0]
        history: list[float] = []
        for _ in range(steps):
            idx = torch.randint(0, n, (batch_size,))
            x = features[idx]
            o = outcomes[idx]
            probs = torch.sigmoid(self.policy(x).squeeze(-1))
            samples = torch.stack(
                [torch.bernoulli(probs) for _ in range(self.g)]
            )
            log_probs = [
                s * torch.log(probs + 1e-9)
                + (1 - s) * torch.log(1 - probs + 1e-9)
                for s in samples
            ]
            rewards = reward_calibration(probs, o)
            advantages = rewards - rewards.mean()
            loss = -torch.stack(log_probs).mul(advantages.detach()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            history.append(float(rewards.mean().detach().item()))
        return history