"""Дообучение энкодера состояния под задачу решений.

Замороженное состояние энкодера даёт внутривопросный сигнал риска, но на новый
вопрос не переносится (см. ``docs/risk-aware-cascade.md``, «Сигнал из состояния
энкодера»). Здесь — открытый из того же раздела вариант: дообучить сам энкодер.

Две фазы, каждая включается числом шагов:

1. **Задача** (:meth:`FinetuneTrainer.fit`): энкодер и головы решений обучаются
   совместно на промптах — градиент течёт в веса энкодера, поэтому
   представление адаптируется под задачу, а не остаётся «тем, что было».
2. **Ошибка** (:meth:`FinetuneTrainer.fit_error`): головы замораживаются, рядом
   обучается общая вспомогательная голова ошибки, а энкодер продолжает
   обучаться. Цель — чтобы состояние несло сигнал «ошибочно ли конкретное
   решение», а не только «какой это вопрос» (именно этого не хватает
   замороженному состоянию). Голова ошибки — тренировочный костыль: наружу
   отдаётся только энкодер.

В отличие от ``SFTTrainer`` (учится на готовых признаках), здесь признаки
пересчитываются на каждом шаге, иначе градиент в энкодер не течёт.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.decision_head.heads import DecisionHeads
from src.encoder.base import TrainableStateEncoder
from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    build_prompt,
)
from training.sft.train import head_loss


def _label_indices(spec: QuestionSpec, labels: list[Any]) -> list[int]:
    """Индексы меток: у ``score`` значение уже индекс, у ``choice`` — имя опции."""
    if isinstance(spec, ScoreSpec):
        return [int(value) for value in labels]
    if isinstance(spec, ChoiceSpec):
        index = {option: position for position, option in enumerate(spec.options)}
        return [index[option] for option in labels]
    raise TypeError(f"label indices are undefined for {type(spec).__name__}")


def targets_for(spec: QuestionSpec, labels: list[Any]) -> Tensor:
    """Цели головы: float для ``noul``, индекс класса — для ``choice``/``score``.

    Совпадает с построителем целей бенчмарка (``eval.benchmark.core``): цели
    дообучения и цели бенчмарка обязаны совпадать, иначе метрики дообученной
    модели были бы не сравнимы с базовыми.
    """
    if isinstance(spec, NoulSpec):
        return torch.tensor([float(value) for value in labels])
    return torch.tensor(_label_indices(spec, labels), dtype=torch.long)


def build_prompts(
    specs: dict[str, QuestionSpec], examples: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Промпты по вопросу — ровно как в рантайме (``build_prompt`` на вопрос)."""
    states = [example["state"] for example in examples]
    return {
        name: [build_prompt(state, spec) for state in states]
        for name, spec in specs.items()
    }


def error_targets(head_logits: Tensor, labels: Tensor) -> Tensor:
    """``1.0``, если решение головы ошибочно, — цель вспомогательной головы ошибки.

    Ошибка считается по **взятому** действию (argmax / порог 0.5), а не по
    вероятности головы: цель — «предсказать свою ошибку», а не повторить её
    величину. Логиты приходят отсоединёнными: цель не должна тянуть за собой
    замороженную голову.
    """
    detached = head_logits.detach()
    if detached.ndim == 1:
        predicted = (torch.sigmoid(detached) >= 0.5).float()
        return (predicted != (labels >= 0.5).float()).float()
    return (detached.argmax(dim=-1) != labels.long()).float()


def _sample(count: int, batch_size: int) -> Tensor:
    return torch.randint(0, count, (min(batch_size, count),))


def _parameter_device(encoder: TrainableStateEncoder) -> torch.device:
    params = encoder.trainable_parameters()
    return params[0].device if params else torch.device("cpu")


class FinetuneTrainer:
    """Совместное дообучение энкодера и голов решений на примерах.

    Энкодер должен быть загружен (``load()``) и поддерживать градиент
    (``TrainableStateEncoder``). Обе фазы обучают **энкодер**; головы — либо
    вместе с ним (фаза задачи), либо замороженными (фаза ошибки).

    Память: обратный проход делается по каждому вопросу отдельно, поэтому
    одновременно живёт один граф промптов. Держите ``batch_size`` равным
    ``batch_size`` энкодера (``DECISION_ENCODE_BATCH``) — иначе промпты вопроса
    разобьются на несколько чанков и графы будут жить одновременно.
    """

    def __init__(
        self,
        encoder: TrainableStateEncoder,
        heads: DecisionHeads,
        *,
        lr: float = 1e-4,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.encoder = encoder
        self.heads = heads
        self.lr = lr
        self.error_head: nn.Linear | None = None

    def fit(
        self,
        specs: dict[str, QuestionSpec],
        examples: list[dict[str, Any]],
        *,
        steps: int,
        batch_size: int = 32,
        lr: float | None = None,
    ) -> list[float]:
        """Фаза задачи: энкодер и головы обучаются совместно.

        Шаг — по батчу на каждый вопрос (как ``SFTTrainer``), но признаки
        считаются заново, поэтому в оптимизаторе лежат и веса энкодера.
        """
        prompts = build_prompts(specs, examples)
        targets = {
            name: targets_for(spec, [ex["labels"][name] for ex in examples])
            for name, spec in specs.items()
        }
        self.encoder.set_trainable(True)
        device = _parameter_device(self.encoder)
        # головы обязаны жить на том же устройстве, что и признаки энкодера
        self.heads.to(device)
        self.heads.train()
        optimizer = torch.optim.Adam(
            [*self.encoder.trainable_parameters(), *self.heads.parameters()],
            lr=self.lr if lr is None else lr,
        )
        losses: list[float] = []
        for _ in range(steps):
            # обратный проход по вопросу, а не один на все: графы промптов
            # (самое дорогое по памяти) освобождаются сразу, а сумма градиентов
            # от этого не меняется
            optimizer.zero_grad()
            total = 0.0
            for name in specs:
                index = _sample(len(prompts[name]), batch_size)
                features = self.encoder.encode_trainable(
                    [prompts[name][i] for i in index.tolist()]
                )
                logits = self.heads.heads[name].logits(features)
                loss = head_loss(logits, targets[name][index].to(features.device))
                loss.backward()
                total += float(loss.detach())
            optimizer.step()
            losses.append(total)
        return losses

    def fit_error(
        self,
        specs: dict[str, QuestionSpec],
        examples: list[dict[str, Any]],
        *,
        steps: int,
        batch_size: int = 32,
        lr: float | None = None,
    ) -> list[float]:
        """Фаза ошибки: энкодер учится предсказывать ошибку замороженных голов.

        Голова ошибки **общая** на все вопросы: та, что переносилась бы на новый
        вопрос (у нового вопроса своей обученной головы ошибки нет). Головы
        решений замораживаются и переводятся в ``eval()`` — цель берётся с них,
        а не обучается.
        """
        prompts = build_prompts(specs, examples)
        targets = {
            name: targets_for(spec, [ex["labels"][name] for ex in examples])
            for name, spec in specs.items()
        }
        self.encoder.set_trainable(True)
        device = _parameter_device(self.encoder)
        self.heads.to(device)
        self.heads.eval()
        for param in self.heads.parameters():
            param.requires_grad_(False)
        self.error_head = nn.Linear(self.heads.dim, 1).to(device)
        optimizer = torch.optim.Adam(
            [
                *self.encoder.trainable_parameters(),
                *self.error_head.parameters(),
            ],
            lr=self.lr if lr is None else lr,
        )
        losses: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad()
            total = 0.0
            for name in specs:
                index = _sample(len(prompts[name]), batch_size)
                features = self.encoder.encode_trainable(
                    [prompts[name][i] for i in index.tolist()]
                )
                labels = targets[name][index].to(features.device)
                with torch.no_grad():
                    wrong = error_targets(self.heads.heads[name].logits(features), labels)
                logits = self.error_head(features).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, wrong)
                loss.backward()
                total += float(loss.detach())
            optimizer.step()
            losses.append(total)
        return losses
