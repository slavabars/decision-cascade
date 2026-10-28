"""Головы решений: отображение признаков в типизированные вероятности."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    build_option_prompt,
    spec_from_dict,
    spec_to_dict,
)
from src.schemas import (
    ChoiceResult,
    DecisionResult,
    NoulResult,
    ScoreResult,
    confidence_max,
    confidence_noul,
    validate_distribution,
)
from src.versioning import CHECKPOINT_VERSION, migrate_checkpoint


def _require_unique(name: str, field: str, values: tuple[str, ...]) -> None:
    """Опечатки в опциях/уровнях ломают распределение — ловим на входе."""
    if len(set(values)) != len(values):
        raise ValueError(
            f"question {name!r} {field} must be unique; got {list(values)}"
        )


CHOICE_HEADS = ("linear", "option_scoring")


def _choice_result(logits: Tensor, options: tuple[str, ...]) -> ChoiceResult:
    """Собирает ``ChoiceResult`` из логитов одного состояния ``[1, k]``."""
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    probs_dict = {opt: float(p) for opt, p in zip(options, probs.tolist())}
    validate_distribution(probs_dict, options)
    choice = max(options, key=lambda o: probs_dict[o])
    return ChoiceResult(
        choice=choice,
        probabilities=probs_dict,
        confidence=confidence_max(probs_dict),
    )


class _TemperedHead(nn.Module):
    """Температурное масштабирование логитов при инференсе.

    ``temperature`` подбирается на отложенном val (см.
    ``training/recipe/calibrate_temperature.py``) и по умолчанию равна ``1.0``,
    то есть поведение не меняется. Масштаб применяется только в ``forward`` и
    ``scaled_logits``: обучение и функции потерь работают с «сырыми» ``logits``,
    поэтому температура не влияет на градиенты и не путает оптимизатор.

    Значение — обычный ``float``, а не буфер: ``state_dict`` остаётся прежним
    (старые чекпоинты читаются без изменений), а сама температура сохраняется в
    чекпоинте отдельным полем ``temperature``.
    """

    temperature: float = 1.0

    def set_temperature(self, value: float) -> None:
        value = float(value)
        if not value > 0.0:
            raise ValueError(f"temperature must be > 0, got {value}")
        self.temperature = value

    def scaled_logits(self, features: Tensor) -> Tensor:
        """Логиты, делённые на температуру — то, что видит потребитель."""
        return self.logits(features) / self.temperature


class NoulHead(_TemperedHead):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def logits(self, features: Tensor) -> Tensor:
        return self.linear(features).squeeze(-1)

    def forward(self, features: Tensor) -> NoulResult:
        logit = torch.sigmoid(self.scaled_logits(features))
        p = float(logit.item())
        return NoulResult(noul=p, confidence=confidence_noul(p))


class ChoiceHead(_TemperedHead):
    def __init__(self, dim: int, options: tuple[str, ...]):
        super().__init__()
        self.options = tuple(options)
        self.linear = nn.Linear(dim, len(self.options))

    def logits(self, features: Tensor) -> Tensor:
        return self.linear(features)

    def forward(self, features: Tensor) -> ChoiceResult:
        return _choice_result(self.scaled_logits(features), self.options)


class OptionScoringChoiceHead(_TemperedHead):
    """Голова выбора, оценивающая каждый вариант по его текстовому эмбеддингу.

    В отличие от :class:`ChoiceHead` (``Linear(dim, k)`` с отдельным весом на
    каждый вариант), здесь оценка варианта — **билинейная симметричная** функция
    от признаков состояния и эмбеддинга варианта: ``<f(h), g(e_i)>``. Число
    параметров не зависит от числа вариантов, а сами эмбеддинги берутся из энкодера
    (``build_option_prompt``) и **не хранятся** в чекпоинте. Поэтому обученные веса
    можно применить к другому набору вариантов — в том числе другого размера — без
    переобучения.

    Важно: скалярное произведение обязательно. Аддитивная форма
    ``linear([f(h) ; g(e_i)])`` вырождена — при ней вклад состояния одинаков для
    всех вариантов, и голова всегда выбирает один и тот же вариант (точность
    уровня случайной). Билинейная форма позволяет состоянию *модулировать* оценку
    варианта.

    Логиты: ``<f(h), g(e_i)> / sqrt(dim)`` для каждого варианта ``i``.
    """

    def __init__(self, dim: int, options: tuple[str, ...], option_embeddings):
        super().__init__()
        self.options = tuple(options)
        embeddings = torch.as_tensor(option_embeddings, dtype=torch.float32)
        expected = (len(self.options), dim)
        if tuple(embeddings.shape) != expected:
            raise ValueError(
                f"option_embeddings must be {expected}, got {tuple(embeddings.shape)}"
            )
        # Не персистентный буфер: эмбеддинги пересчитываются из энкодера, а не
        # восстанавливаются из чекпоинта — иначе перенос на новые варианты был бы
        # невозможен.
        self.register_buffer(
            "option_embeddings", embeddings.contiguous(), persistent=False
        )
        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.option_proj = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5

    def logits(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError(
                f"features must be [batch, dim]; got {tuple(features.shape)}"
            )
        dim = self.option_embeddings.shape[-1]
        if features.shape[-1] != dim:
            raise ValueError(
                f"features dim {features.shape[-1]} != option embeddings dim {dim}"
            )
        state = self.state_proj(features)
        options = self.option_proj(self.option_embeddings)
        return (state @ options.t()) * self.scale

    def forward(self, features: Tensor) -> ChoiceResult:
        return _choice_result(self.scaled_logits(features), self.options)


def build_option_embeddings(
    questions: dict[str, QuestionSpec], encode: Callable[[list[str]], Tensor]
) -> dict[str, Tensor]:
    """Эмбеддинги вариантов выбора: ``{имя: [k, dim]}``.

    ``encode`` — функция ``list[str] -> [n, dim]`` (обычно ``encoder.encode``).
    Вектор варианта считается без состояния (``build_option_prompt``), поэтому
    его можно переиспользовать для любых состояний.
    """
    result: dict[str, Tensor] = {}
    for name, spec in questions.items():
        if isinstance(spec, ChoiceSpec):
            prompts = [build_option_prompt(spec, opt) for opt in spec.options]
            result[name] = encode(prompts)
    return result


class ScoreHead(_TemperedHead):
    def __init__(self, dim: int, levels: tuple[str, ...]):
        super().__init__()
        self.levels = tuple(levels)
        self.linear = nn.Linear(dim, len(self.levels))

    def logits(self, features: Tensor) -> Tensor:
        return self.linear(features)

    def forward(self, features: Tensor) -> ScoreResult:
        probs_list = torch.softmax(
            self.scaled_logits(features).squeeze(0), dim=-1
        ).tolist()
        probs_dict = {lvl: float(p) for lvl, p in zip(self.levels, probs_list)}
        validate_distribution(probs_dict, self.levels)
        score = float(sum(i * p for i, p in enumerate(probs_list)))
        return ScoreResult(
            score=score,
            probabilities=probs_dict,
            confidence=confidence_max(probs_dict),
            legend=self.levels,
        )


class DecisionHeads(nn.Module):
    def __init__(
        self,
        dim: int,
        questions: dict[str, QuestionSpec],
        seed: int | None = None,
        choice_head: str = "linear",
        option_embeddings: dict[str, Tensor] | None = None,
    ):
        super().__init__()
        mode = str(choice_head).strip().lower()
        if mode not in CHOICE_HEADS:
            raise ValueError(
                f"choice_head must be one of {CHOICE_HEADS}, got {choice_head!r}"
            )
        self.dim = dim
        self.choice_head = mode
        self.questions = dict(questions)
        self.heads: dict[str, nn.Module] = {}
        if seed is not None:
            torch.manual_seed(seed)
        for name, spec in self.questions.items():
            if isinstance(spec, NoulSpec):
                self.heads[name] = NoulHead(dim)
            elif isinstance(spec, ChoiceSpec):
                _require_unique(name, "options", spec.options)
                if mode == "option_scoring":
                    if not option_embeddings or name not in option_embeddings:
                        raise ValueError(
                            "choice_head='option_scoring' requires option embeddings "
                            f"for question {name!r} (build_option_embeddings)"
                        )
                    self.heads[name] = OptionScoringChoiceHead(
                        dim, spec.options, option_embeddings[name]
                    )
                else:
                    self.heads[name] = ChoiceHead(dim, spec.options)
            elif isinstance(spec, ScoreSpec):
                _require_unique(name, "levels", spec.levels)
                self.heads[name] = ScoreHead(dim, spec.levels)
            else:
                raise TypeError(f"unsupported question spec: {type(spec).__name__}")
            self.add_module(name, self.heads[name])

    def logits(self, features: Tensor) -> dict[str, Tensor]:
        if features.dim() != 2:
            raise ValueError(f"features must be [batch, dim]; got {tuple(features.shape)}")
        return {name: head.logits(features) for name, head in self.heads.items()}

    def evaluate(self, features: Tensor) -> dict[str, DecisionResult]:
        if features.dim() != 2 or features.shape[0] != len(self.questions):
            raise ValueError(
                f"features must be [n_questions, dim]; got shape {tuple(features.shape)}"
            )
        results = {}
        for i, (name, head) in enumerate(self.heads.items()):
            results[name] = head(features[i].unsqueeze(0))
        return results

    def temperatures(self) -> dict[str, float]:
        """Температуры логитов по вопросам (``1.0`` — без масштабирования)."""
        return {name: float(head.temperature) for name, head in self.heads.items()}

    def set_temperatures(self, values: dict[str, float]) -> None:
        """Применяет температуры из чекпоинта; неизвестные вопросы — ошибка."""
        unknown = set(values) - set(self.heads)
        if unknown:
            raise ValueError(
                f"temperature for unknown questions: {sorted(unknown)}"
            )
        for name, value in values.items():
            self.heads[name].set_temperature(value)

    def save_checkpoint(self, path: str | Path) -> Path:
        """Сохраняет веса голов вместе со спецификациями вопросов для восстановления."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": CHECKPOINT_VERSION,
                "dim": self.dim,
                "choice_head": self.choice_head,
                "questions": {
                    name: spec_to_dict(spec) for name, spec in self.questions.items()
                },
                "temperature": self.temperatures(),
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        seed: int | None = None,
        choice_head: str | None = None,
        option_embeddings: dict[str, Tensor] | None = None,
    ) -> DecisionHeads:
        """Восстанавливает головы (структуру и веса) из чекпоинта.

        Legacy-чекпоинты без ``schema_version`` читаются как версия ``1``; более
        новый major отклоняется, чтобы старый код не читал неизвестный формат.

        Режим ``choice_head`` берётся из чекпоинта; для ``option_scoring`` нужны
        ``option_embeddings`` — они не хранятся в файле и пересчитываются из
        энкодера (чтобы набор вариантов можно было менять без переобучения).
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        payload = migrate_checkpoint(payload)
        questions = {
            name: spec_from_dict(data) for name, data in payload["questions"].items()
        }
        mode = choice_head or payload.get("choice_head", "linear")
        heads = cls(
            payload["dim"],
            questions,
            seed=seed,
            choice_head=mode,
            option_embeddings=option_embeddings,
        )
        heads.load_state_dict(payload["state_dict"])
        heads.set_temperatures(payload.get("temperature") or {})
        return heads