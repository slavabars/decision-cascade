"""RLCD: reinforcement learning for calibrated decisions.

Прототип для Noul-головы: политика p = sigmoid(linear(features)). Внутри
группы берутся G выборок Y ~ Bernoulli(p) (GRPO-стиль), reward считается по
*взятому* действию: -|s - o|, преимущества нормализуются GRPO-стилем,
policy-gradient обновление через лог-вероятность выборок.

Награда обязана зависеть от сэмпла `s`, а не от вероятности политики: иначе
`E_s[∇ log π(s)] = 0` и ожидаемый градиент строго нулевой — обучение идёт
только конечновыборочным шумом.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from training.rlcd.reward import group_advantages, reward_calibration


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
            rewards = reward_calibration(samples, o.unsqueeze(0))
            advantages = group_advantages(rewards)
            loss = -torch.stack(log_probs).mul(advantages.detach()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            history.append(float(rewards.mean().detach().item()))
        return history


def calibrate_noul_heads(
    heads,
    features: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    steps: int = 200,
    lr: float = 1e-2,
    g: int = 8,
    gate_features: dict[str, Tensor] | None = None,
    gate_targets: dict[str, Tensor] | None = None,
) -> dict[str, float]:
    """RLCD-калибровка noul-голов: reward = -|s - o| (s — взятый сэмпл), веса копируются в головы.

    Для каждой noul-головы обучается отдельная политика на её (и только её)
    признаках. Калибровка применяется только если она *улучшает* среднюю
    абсолютную ошибку вероятности, иначе веса головы откатываются (RLCD не
    должен ухудшать уже откалиброванную голову). Возвращает итоговую среднюю
    абсолютную ошибку по каждой noul-голове.

    Гейт «не навреди» считается на ``gate_features``/``gate_targets``, если они
    заданы (отложенный набор — честная нерегрессия), иначе на обучающих признаках.
    """
    from src.sampler.questions import NoulSpec

    history: dict[str, float] = {}
    for name, spec in heads.questions.items():
        if not isinstance(spec, NoulSpec):
            continue
        head = heads.heads[name]
        x = features[name]
        outcomes = targets[name].float()
        if gate_features is not None and gate_targets is not None:
            gate_x = gate_features[name]
            gate_out = gate_targets[name].float()
        else:
            gate_x, gate_out = x, outcomes

        with torch.no_grad():
            before = float((torch.sigmoid(head.logits(gate_x)) - gate_out).abs().mean())

        trainer = RLCDNoulTrainer(head.linear.in_features, lr=lr, g=g)
        trainer.train(x, outcomes, steps=steps)
        with torch.no_grad():
            after = float(
                (torch.sigmoid(trainer.policy(gate_x).squeeze(-1)) - gate_out)
                .abs()
                .mean()
            )

        if after < before:
            with torch.no_grad():
                head.linear.weight.copy_(trainer.policy.weight)
                head.linear.bias.copy_(trainer.policy.bias)
            history[name] = after
        else:
            history[name] = before
    return history