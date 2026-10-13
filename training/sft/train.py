"""SFT: supervised fine-tuning голов решений на типизированных данных.

Признаки можно подавать двумя способами:

* ``dict[str, Tensor]`` — свои промпты на каждый вопрос (так работает рантайм:
  ``build_prompt(state, spec)`` на вопрос, затем голова вопроса на своих строках);
* ``Tensor`` — общая матрица признаков на все головы (допустимо только для
  одного вопроса; для нескольких вопросов берите dict, иначе цели смешаются).
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.decision_head.heads import DecisionHeads


def head_loss(head_logits: Tensor, target: Tensor) -> Tensor:
    """Потеря одной головы: BCE для noul (1 логит), CE для choice/score."""
    if head_logits.ndim == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            head_logits, target
        )
    return torch.nn.functional.cross_entropy(head_logits, target.long())


def sft_loss(
    head_logits: dict[str, Tensor],
    targets: dict[str, Tensor],
    heads: DecisionHeads,
) -> Tensor:
    total = torch.zeros((), requires_grad=True)
    for name in heads.heads:
        total = total + head_loss(head_logits[name], targets[name])
    return total


class SFTTrainer:
    def __init__(self, heads: DecisionHeads, lr: float = 1e-2):
        self.heads = heads
        self.optimizer = torch.optim.Adam(heads.parameters(), lr=lr)

    def fit(
        self,
        features: Tensor | dict[str, Tensor],
        targets: dict[str, Tensor],
        steps: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        if isinstance(features, dict):
            return self._fit_per_question(features, targets, steps, batch_size)
        if len(self.heads.heads) > 1:
            raise ValueError(
                "features is a single tensor but there are "
                f"{len(self.heads.heads)} questions; pass a dict keyed by "
                "question name so each head trains on its own prompts"
            )
        return self._fit_shared(features, targets, steps, batch_size)

    def _fit_shared(
        self, features: Tensor, targets: dict[str, Tensor], steps: int, batch_size: int
    ) -> list[float]:
        n = features.shape[0]
        losses: list[float] = []
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),))
            batch_targets = {k: v[idx] for k, v in targets.items()}
            self.heads.train()
            loss = sft_loss(self.heads.logits(features[idx]), batch_targets, self.heads)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        return losses

    def _fit_per_question(
        self,
        features: dict[str, Tensor],
        targets: dict[str, Tensor],
        steps: int,
        batch_size: int,
    ) -> list[float]:
        missing = set(self.heads.heads) - set(features)
        if missing:
            raise ValueError(f"no features provided for questions {sorted(missing)}")
        unknown = set(features) - set(self.heads.heads)
        if unknown:
            raise ValueError(f"features for unknown questions {sorted(unknown)}")
        missing_targets = set(self.heads.heads) - set(targets)
        if missing_targets:
            raise ValueError(
                f"no targets provided for questions {sorted(missing_targets)}"
            )

        self.heads.train()
        losses: list[float] = []
        for _ in range(steps):
            total = torch.zeros((), requires_grad=True)
            for name, head in self.heads.heads.items():
                f = features[name]
                n = f.shape[0]
                idx = torch.randint(0, n, (min(batch_size, n),))
                total = total + head_loss(head.logits(f[idx]), targets[name][idx])
            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            losses.append(float(total.detach().item()))
        return losses
