"""Self-hosted «живой» конкурент для бенчмарка (протокол Jev/SystemOne).

Поднимает ``POST /v1/systemone`` поверх любой causal-LM из ``transformers`` и
отвечает на типизированные вопросы без генерации: каждый вариант-кандидат
(``option``/``level`` или пара ``yes``/``no``) оценивается по правдоподобию своих
токенов при данном состоянии (sequence log-likelihood), затем кандидаты
нормируются softmax. Так получаются вероятности, пригодные для честного
сравнения (accuracy/Brier/ECE), а не mock-числа.

Настоящие (не mock) числа конкурента целиком локально:

    # 1) поднять сервис (веса скачаются в кэш HF при первом запуске)
    python -m eval.competitors.serve --model Qwen/Qwen2.5-0.5B-Instruct \\
        --port 8090 --device cuda

    # 2) в другом терминале прогнать бенчмарк против него
    python -m eval.competitors.run --specs public --limit 200 \\
        --remote live=http://127.0.0.1:8090 --out data/competitors_live.json

Тот же протокол понимают развёртывания Clef/Clef-Flash/vLLM, поэтому ``--remote``
подключает и их; этот сервис нужен для воспроизводимого локального прогона без
внешних API.
"""
from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from fastapi import FastAPI

from api.server.systemone import (
    SystemOneChoiceAnswer,
    SystemOneNoulAnswer,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreAnswer,
    systemone_to_specs,
)
from src.device import check_dtype_device, resolve_device, resolve_dtype
from src.sampler.questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    build_prompt,
)

# Пары-кандидаты для noul без критериев: первый — «истина» (probability = P(первый)).
NOUL_YES = "yes"
NOUL_NO = "no"


def softmax(scores: list[float]) -> list[float]:
    """Численно устойчивый softmax над списком логитов."""
    if not scores:
        return []
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return [int(token) for token in encoded["input_ids"]]


def _pad_id(tokenizer: Any) -> int:
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


@torch.inference_mode()
def score_candidates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[str],
    *,
    device: str = "cpu",
    max_length: int = 1024,
    normalize: bool = True,
) -> list[float]:
    """Средний log-prob каждого кандидата как продолжения ``prompt``.

    Кандидат токенизируется с ведущим пробелом и приписывается к префиксу
    (подсказке); неуверенность в длине снимается нормировкой на число токенов.
    """
    candidate_ids = [_token_ids(tokenizer, " " + candidate) for candidate in candidates]
    longest = max((len(ids) for ids in candidate_ids), default=1) or 1
    prefix = _token_ids(tokenizer, prompt)
    # Оставляем место под самый длинный кандидат и один позиционный сдвиг.
    keep = max(1, max_length - longest - 1)
    if len(prefix) > keep:
        prefix = prefix[-keep:]

    sequences = [prefix + ids for ids in candidate_ids]
    length = max(len(seq) for seq in sequences)
    pad = _pad_id(tokenizer)
    input_ids = torch.full((len(sequences), length), pad, dtype=torch.long)
    attention = torch.zeros((len(sequences), length), dtype=torch.long)
    for row, seq in enumerate(sequences):
        input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention[row, : len(seq)] = 1

    input_ids = input_ids.to(device)
    attention = attention.to(device)
    logits = model(input_ids, attention_mask=attention).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    scores: list[float] = []
    for row, seq in enumerate(sequences):
        start = len(prefix)
        total = 0.0
        count = 0
        for position in range(start, len(seq)):
            total += float(logprobs[row, position - 1, seq[position]])
            count += 1
        if normalize and count:
            total /= count
        scores.append(total)
    return scores


class LiveEngine:
    """Causal-LM, отвечающая на типизированные вопросы по протоколу SystemOne."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str = "live",
        device: str = "cpu",
        max_length: int = 1024,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.last_usage: dict[str, Any] = {"tokens": 0, "questions": 0}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_length: int = 1024,
    ) -> LiveEngine:
        """Загружает веса HF и переносит модель на устройство (eval-режим)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:  # transformers>=5 зовёт аргумент dtype, ранние версии — torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device).eval()
        return cls(
            model,
            tokenizer,
            model_id=model_id,
            device=device,
            max_length=max_length,
        )

    def _probabilities(self, state: str, spec: QuestionSpec, candidates: list[str]):
        prompt = build_prompt(state, spec) + "\nAnswer:"
        scores = score_candidates(
            self._model,
            self._tokenizer,
            prompt,
            candidates,
            device=self.device,
            max_length=self.max_length,
        )
        return softmax(scores)

    def answer(
        self, state: str, specs: dict[str, QuestionSpec]
    ) -> dict[str, Any]:
        """Состояние + спецификации -> ответы протокола и расход токенов."""
        answers: dict[str, Any] = {}
        tokens = 0
        for name, spec in specs.items():
            if isinstance(spec, NoulSpec):
                # build_prompt трактует criteria как (true_desc, false_desc).
                if spec.criteria:
                    positive, negative = spec.criteria
                else:
                    positive, negative = NOUL_YES, NOUL_NO
                probabilities = self._probabilities(state, spec, [positive, negative])
                probability = float(probabilities[0])
                answers[name] = SystemOneNoulAnswer(
                    probability=min(max(probability, 0.0), 1.0),
                    confidence=abs(2.0 * probability - 1.0),
                )
            elif isinstance(spec, ChoiceSpec):
                options = list(spec.options)
                probabilities = self._probabilities(state, spec, options)
                best = max(range(len(options)), key=lambda i: probabilities[i])
                answers[name] = SystemOneChoiceAnswer(
                    choice=options[best],
                    confidence=float(probabilities[best]),
                    probabilities={
                        option: float(probability)
                        for option, probability in zip(options, probabilities)
                    },
                )
            elif isinstance(spec, ScoreSpec):
                levels = list(spec.levels)
                probabilities = self._probabilities(state, spec, levels)
                expected = sum(i * p for i, p in enumerate(probabilities))
                answers[name] = SystemOneScoreAnswer(
                    score=float(expected),
                    confidence=float(max(probabilities)),
                    probabilities={
                        level: float(probability)
                        for level, probability in zip(levels, probabilities)
                    },
                    legend=levels,
                )
            else:  # pragma: no cover - QuestionSpec закрыт тремя типами
                raise TypeError(f"unsupported question: {type(spec).__name__}")
            tokens += sum(
                1 for _ in _token_ids(self._tokenizer, build_prompt(state, spec))
            )
        self.last_usage = {"tokens": tokens, "questions": len(specs)}
        return answers


def create_app(engine: LiveEngine) -> FastAPI:
    """FastAPI-приложение, отдающее ``POST /v1/systemone`` поверх модели."""
    app = FastAPI(title="decision-cascade live competitor", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": engine.model_id, "device": engine.device}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> SystemOneResponse:
        specs = systemone_to_specs(request.questions)
        answers = engine.answer(request.state, specs)
        return SystemOneResponse(
            model=request.model or engine.model_id,
            answers=answers,
            usage=engine.last_usage,
        )

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Self-hosted SystemOne competitor")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    check_dtype_device(dtype, device)

    print(f"loading {args.model} on {device} ({args.dtype}) ...")
    engine = LiveEngine.from_pretrained(
        args.model, device=device, dtype=dtype, max_length=args.max_length
    )
    print(f"serving http://{args.host}:{args.port}/v1/systemone")
    import uvicorn

    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
