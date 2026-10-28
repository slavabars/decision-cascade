"""Self-hosted адаптеры релизов конкурентов (Laya, Clef-Flash) под SystemOne.

Реальные релизы Clef-Flash и Laya не отдают ``POST /v1/systemone`` «из коробки»
через HTTP: у Clef это функция ``systemone(...)`` в ``joint_schema_model.py``, у
Laya — ``Router.predict(...)``. Этот модуль поднимает единый HTTP-фасад
протокола, поэтому ``eval.competitors.run --remote`` меряет их теми же метриками,
что и наши головы:

    # Clef-Flash (9.4B; на 16 ГБ VRAM целиком не влезает — CPU bf16)
    python -m eval.competitors.release --model clef-flash --device cpu \
        --dtype bfloat16 --port 8091

    # Clef-Flash из GGUF в llama-server (он сам отдаёт /v1/systemone)
    llama-server -m clef-flash-Q4_K_M.gguf -ngl 99 -c 16384 --port 8093
    python -m eval.competitors.release --model clef-flash \
        --base-url http://127.0.0.1:8093 --port 8091

    # Laya (421M, ModernBERT-large)
    python -m eval.competitors.release --model laya --device cuda --port 8092

    # бенчмарк против обоих
    python -m eval.competitors.run --specs support --limit 200 \
        --remote clef-flash=http://127.0.0.1:8091 \
        --remote laya=http://127.0.0.1:8092 --out data/competitors_real.json

Протокол и семантика ответов переводятся в наш вид: ``options``/``levels``
уходят в ``criteria`` (так их ждут обе модели), а ``score``-ответы получают
ключи вероятностей по названиям уровней, а не по индексам.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

CLEF_MODEL_ID = "Cloudflare/clef-flash"
LAYA_MODEL_ID = "convaiinnovations/laya"


def _criteria(question: dict[str, Any]) -> dict[str, Any]:
    """Вопрос протокола -> вопрос в формате Clef/Laya (опции живут в ``criteria``)."""
    kind = question["type"]
    instructions = question.get("instructions") or ""
    if kind == "noul":
        criteria = question.get("criteria")
        result: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            result["criteria"] = {"true": criteria[0], "false": criteria[1]}
        return result
    if kind == "choice":
        options = list(question.get("options") or [])
        criteria_map = question.get("criteria")
        if not criteria_map:
            criteria_map = {option: option for option in options}
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": dict(criteria_map),
        }
    levels = list(question.get("levels") or question.get("criteria") or [])
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _score_answer(answer: dict[str, Any], levels: list[str]) -> dict[str, Any]:
    """``score`` с индексами -> ``score`` с ключами-уровнями (наш протокол)."""
    probabilities = answer["probabilities"]
    return {
        "type": "score",
        "score": float(answer["score"]),
        "confidence": float(answer["confidence"]),
        "probabilities": {
            level: float(probabilities[str(index)])
            for index, level in enumerate(levels)
        },
        "legend": list(levels),
    }


def _noul_answer(answer: dict[str, Any]) -> dict[str, Any]:
    probability = float(answer.get("probability", answer.get("noul")))
    return {
        "type": "noul",
        "probability": probability,
        "confidence": float(answer.get("confidence") or abs(2.0 * probability - 1.0)),
    }


def _to_protocol(
    answers: dict[str, Any], questions: dict[str, Any]
) -> dict[str, Any]:
    """Ответы релиза -> ответы протокола (noul/probability, score по уровням)."""
    converted: dict[str, Any] = {}
    for name, answer in answers.items():
        kind = questions[name]["type"]
        if kind == "noul":
            converted[name] = _noul_answer(answer)
        elif kind == "choice":
            converted[name] = {
                "type": "choice",
                "choice": answer["choice"],
                "confidence": float(answer["confidence"]),
                "probabilities": {
                    key: float(value)
                    for key, value in answer["probabilities"].items()
                },
            }
        else:
            levels = list(questions[name].get("levels") or questions[name]["criteria"])
            converted[name] = _score_answer(answer, levels)
    return converted


class LlamaCppEngine:
    """Clef-Flash в llama.cpp: ``llama-server`` уже отдаёт SystemOne, но ждёт criteria.

    Квантованные GGUF-релизы (``bartowski/Cloudflare_clef-flash-GGUF``) поднимаются
    ``llama-server`` и отвечают на ``POST /v1/systemone`` нативно. Отличие от нашего
    протокола одно: ``options``/``levels`` у него называются ``criteria``, а
    ``score``-вероятности приходят с индексами уровней — это и правит фасад.
    """

    def __init__(self, base_url: str, model_id: str = "clef-flash"):
        import httpx

        self.model_id = model_id
        self.device = base_url
        self._client = httpx.Client(base_url=base_url, timeout=600.0)

    def answer(
        self, state: str, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.model_id,
            "state": state,
            "questions": {
                name: _criteria(question) for name, question in questions.items()
            },
        }
        response = self._client.post("/v1/systemone", json=payload)
        response.raise_for_status()
        body = response.json()
        return _to_protocol(body["answers"], questions), dict(body.get("usage") or {})


class LayaEngine:
    """Laya: ``Router.predict`` уже возвращает SystemOne-подобный payload."""

    def __init__(self, device: str, dtype: str):
        from laya import Router

        self.model_id = LAYA_MODEL_ID
        self.device = device
        self._router = Router(device=device, preload=True)

    def answer(
        self, state: str, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self._router.predict(
            state, {name: _criteria(question) for name, question in questions.items()}
        )
        answers = _to_protocol(payload["answers"], questions)
        usage = dict(payload.get("usage") or {})
        routing = payload.get("routing") or {}
        if routing.get("model"):
            usage["routing_model"] = 1
        return answers, usage


class ClefEngine:
    """Clef-Flash: ``joint_schema_model.systemone`` поверх Qwen3.5 + joint head."""

    def __init__(self, device: str, dtype: str, model_id: str = CLEF_MODEL_ID):
        import torch
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id))
        sys.path.insert(0, str(path))
        from joint_schema_model import load_release_model, systemone

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model_id = model_id
        self.device = device
        self._systemone = systemone
        self._model, self._processor = load_release_model(
            path, device=device, dtype=torch_dtype
        )

    def answer(
        self, state: str, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = {
            "model": self.model_id,
            "state": state,
            "questions": {
                name: _criteria(question) for name, question in questions.items()
            },
        }
        payload = self._systemone(self._model, self._processor, request)
        answers = _to_protocol(payload["answers"], questions)
        return answers, dict(payload.get("usage") or {})


def create_app(engine: Any) -> Any:
    """FastAPI-фасад ``POST /v1/systemone`` над движком релиза."""
    from fastapi import FastAPI

    app = FastAPI(title="decision-cascade release competitor", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": engine.model_id, "device": engine.device}

    @app.post("/v1/systemone")
    def systemone(request: dict[str, Any]) -> dict[str, Any]:
        answers, usage = engine.answer(request["state"], request["questions"])
        return {
            "model": request.get("model") or engine.model_id,
            "answers": answers,
            "usage": usage,
        }

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SystemOne-фасад релиза конкурента")
    parser.add_argument("--model", default="laya", choices=["laya", "clef-flash"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--base-url",
        default="",
        help="llama-server с GGUF-релизом Clef (тогда --model только имя)",
    )
    args = parser.parse_args(argv)

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    if args.base_url:
        engine: Any = LlamaCppEngine(args.base_url, model_id=args.model)
    elif args.model == "laya":
        engine = LayaEngine(args.device, args.dtype)
    else:
        engine = ClefEngine(args.device, args.dtype)
    print(f"serving {engine.model_id} on {args.device} at http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
