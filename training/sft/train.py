"""SFT: supervised fine-tuning голов решений на типизированных данных."""
from __future__ import annotations

import torch
from torch import Tensor

from src.decision_head.heads import DecisionHeads


def sft_loss(
    head_logits: dict[str, Tensor],
    targets: dict[str, Tensor],
    heads: DecisionHeads,
) -> Tensor:
    total = torch.tensor(0.0, requires_grad=True)
    for name in heads.heads:
        logits = head_logits[name]
        target = targets[name]
        if logits.ndim == 1:
            total = total + torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target
            )
        else:
            total = total + torch.nn.functional.cross_entropy(logits, target.long())
    return total


class SFTTrainer:
    def __init__(self, heads: DecisionHeads, lr: float = 1e-2):
        self.heads = heads
        self.optimizer = torch.optim.Adam(heads.parameters(), lr=lr)

    def fit(
        self,
        features: Tensor,
        targets: dict[str, Tensor],
        steps: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        n = features.shape[0]
        losses: list[float] = []
        for _ in range(steps):
            idx = torch.randint(0, n, (batch_size,))
            batch = features[idx]
            batch_targets = {k: v[idx] for k, v in targets.items()}
            self.heads.train()
            loss = sft_loss(self.heads.logits(batch), batch_targets, self.heads)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        return losses