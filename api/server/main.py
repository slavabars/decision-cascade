"""decision-cascade API.

Принимает state + вопросы, возвращает типизированные решения
с калиброванными вероятностями и schema_version.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from api.server import metrics
from api.server.deprecation import deprecation_for, validate_registry, warn_once
from api.server.observability import (
    REQUEST_ID_HEADER,
    access_record,
    bind_request_id,
    configure_logging,
    new_request_id,
    reset_request_id,
)
from api.server.service import (
    DEFAULT_CHECKPOINT_DIR,
    UNTRAINED_POLICIES,
    DecisionService,
)
from api.server.systemone import (
    SystemOneRequest,
    SystemOneResponse,
    result_to_answer_model,
    systemone_to_specs,
)
from src import __version__
from src.backends.base import (
    Backend,
    backend_readiness,
    missing_modalities,
    requested_modalities,
)
from src.backends.cascade import CascadeBackend
from src.backends.http import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    HttpSystemOneBackend,
    parse_modalities,
)
from src.backends.local import LocalHeadsBackend
from src.device import check_dtype_device, resolve_device, resolve_dtype
from src.encoder.vision import VisionEncoder
from src.errors import RemoteBackendError, UntrainedDecisionError
from src.sampler.questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec
from src.schemas import SCHEMA_VERSION, DecisionResult

logger = logging.getLogger("decision.cascade")
access_logger = logging.getLogger("decision.access")

# JSON-логи включаются окружением; по умолчанию поведение не меняется.
configure_logging()


def create_vision_encoder() -> VisionEncoder | None:
    """Vision-энкодер по окружению: DECISION_VISION_MODEL_ID включает CLIP."""
    model_id = os.environ.get("DECISION_VISION_MODEL_ID")
    if not model_id:
        return None
    from src.encoder.vision import ClipVisionEncoder

    media_root = os.environ.get("DECISION_MEDIA_ROOT")
    encoder = ClipVisionEncoder(
        model_id=model_id,
        dim=parse_dim(
            "DECISION_VISION_DIM",
            os.environ.get("DECISION_VISION_DIM"),
            default=512,
            lo=1,
            hi=8192,
        ),
        device=resolve_device(os.environ.get("DECISION_DEVICE")),
        media_root=Path(media_root) if media_root else None,
    )
    encoder.load()
    return encoder


def parse_dim(
    name: str, raw: str | None, *, default: int, lo: int, hi: int
) -> int:
    """Разбирает размерность из окружения и держит её в разумных границах."""
    value = default if raw is None else raw
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{name} must be an int in [{lo}, {hi}], got {value!r}"
        ) from exc
    if not lo <= parsed <= hi:
        raise RuntimeError(f"{name} must be in [{lo}, {hi}], got {parsed}")
    return parsed


def parse_max_length(raw: str) -> int | None:
    """Разбирает DECISION_MAX_LENGTH: положительный int или None (без обрезки)."""
    value = raw.strip().lower()
    if value in {"", "none", "0"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"DECISION_MAX_LENGTH must be a positive int or 'none', got {raw!r}"
        ) from exc
    return parsed if parsed >= 1 else None


def parse_positive_float(name: str, raw: str | None, *, default: float) -> float:
    """Разбирает положительное число из окружения (таймаут внешнего бэкенда)."""
    value = default if raw is None else raw
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive number, got {value!r}") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be > 0, got {parsed}")
    return parsed


def parse_retries(raw: str | None, *, default: int) -> int:
    """Разбирает число retry внешнего бэкенда (0 — без повторов)."""
    value = default if raw is None else raw
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"DECISION_BACKEND_RETRIES must be a non-negative int, got {value!r}"
        ) from exc
    if parsed < 0:
        raise RuntimeError(
            f"DECISION_BACKEND_RETRIES must be >= 0, got {parsed}"
        )
    return parsed


def http_backend_options() -> dict[str, float | int]:
    """Таймаут и retry внешних вызовов из окружения (общие для http и cascade)."""
    return {
        "timeout": parse_positive_float(
            "DECISION_BACKEND_TIMEOUT",
            os.environ.get("DECISION_BACKEND_TIMEOUT"),
            default=DEFAULT_TIMEOUT,
        ),
        "retries": parse_retries(
            os.environ.get("DECISION_BACKEND_RETRIES"), default=DEFAULT_RETRIES
        ),
    }


def warmup_enabled() -> bool:
    """Прогрев чекпоинтов при старте; ``DECISION_WARMUP=0`` отключает его."""
    return os.environ.get("DECISION_WARMUP", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def warmup_service(service: DecisionService) -> DecisionService:
    """Прогревает головы, чтобы ``/ready`` отражал реальную готовность."""
    if not warmup_enabled():
        return service
    loaded = service.warmup()
    if loaded:
        logger.info(
            "warmup: loaded %d checkpoint(s) from %s",
            loaded,
            service.checkpoint_dir,
        )
    return service


def resolve_untrained_policy() -> str:
    """Разбирает ``DECISION_UNTRAINED``: ``allow``/``refuse``/``fallback``/``auto``.

    ``auto`` (по умолчанию) — строгий режим, если каталог чекпоинтов задан явно
    (``DECISION_CHECKPOINT_DIR``): тогда обученные головы ожидаются, и отдавать
    случайные нельзя. Без этой переменной это осознанный демо-режим — ``allow``.
    Решение не зависит от файловой системы, поэтому поведение воспроизводимо.
    """
    value = os.environ.get("DECISION_UNTRAINED", "auto").strip().lower()
    if value == "auto":
        return "refuse" if "DECISION_CHECKPOINT_DIR" in os.environ else "allow"
    if value not in UNTRAINED_POLICIES:
        raise RuntimeError(
            f"DECISION_UNTRAINED must be one of {(*UNTRAINED_POLICIES, 'auto')}, "
            f"got {value!r}"
        )
    return value


def create_service() -> DecisionService:
    """Сервис по окружению: DECISION_MODEL_ID включает реальный state-энкодер."""
    checkpoint_dir = os.environ.get("DECISION_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    untrained_policy = resolve_untrained_policy()
    choice_head = os.environ.get("DECISION_CHOICE_HEAD", "linear")
    vision_encoder = create_vision_encoder()
    model_id = os.environ.get("DECISION_MODEL_ID")
    if not model_id:
        return warmup_service(
            DecisionService(
                checkpoint_dir=checkpoint_dir,
                vision_encoder=vision_encoder,
                untrained_policy=untrained_policy,
                choice_head=choice_head,
            )
        )
    from src.encoder.factory import create_state_encoder

    feature_dim = parse_dim(
        "DECISION_FEATURE_DIM",
        os.environ.get("DECISION_FEATURE_DIM"),
        default=768,
        lo=1,
        hi=8192,
    )
    max_length = parse_max_length(os.environ.get("DECISION_MAX_LENGTH", "256"))
    batch_size = parse_dim(
        "DECISION_ENCODE_BATCH",
        os.environ.get("DECISION_ENCODE_BATCH"),
        default=16,
        lo=1,
        hi=4096,
    )
    device = resolve_device(os.environ.get("DECISION_DEVICE"))
    dtype = resolve_dtype(os.environ.get("DECISION_DTYPE"))
    check_dtype_device(dtype, device)
    encoder = create_state_encoder(
        model_id,
        feature_dim,
        dtype=dtype,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    encoder.load()
    return warmup_service(
        DecisionService(
            encoder=encoder,
            dim=feature_dim,
            checkpoint_dir=checkpoint_dir,
            vision_encoder=vision_encoder,
            untrained_policy=untrained_policy,
            choice_head=choice_head,
        )
    )


def create_backend(service: DecisionService | None = None) -> Backend:
    """Выбирает бэкенд по окружению: локальные головы или внешний SystemOne-сервис."""
    kind = os.environ.get("DECISION_BACKEND", "local").strip().lower()
    if kind == "http":
        url = os.environ.get("DECISION_BACKEND_URL")
        if not url:
            raise RuntimeError("DECISION_BACKEND=http requires DECISION_BACKEND_URL")
        model = os.environ.get("DECISION_BACKEND_MODEL", "remote")
        modalities = parse_modalities(
            os.environ.get("DECISION_BACKEND_MODALITIES", "text")
        )
        return HttpSystemOneBackend(
            url, model=model, modalities=modalities, **http_backend_options()
        )
    if kind == "cascade":
        return create_cascade_backend(service)
    return LocalHeadsBackend(service or create_service())


def create_cascade_backend(service: DecisionService | None = None) -> CascadeBackend:
    """Локальные головы как primary, внешний SystemOne как fallback.

    По умолчанию эскалация — по порогу уверенности (``DECISION_CASCADE_THRESHOLD``).
    Если задан ``DECISION_CASCADE_RISK_BUDGET``, используется risk-aware каскад:
    обученный оценщик ``P(ошибка)`` (файл ``DECISION_RISK_MODEL_PATH``, по
    умолчанию ``<DECISION_CHECKPOINT_DIR>/risk.json``) и бюджет ошибок вместо
    порога на сыром confidence. Вердикт сертификата пересчитывается с допуском
    ``DECISION_RISK_GATE_DELTA`` (> 0 ослабляет гейт, < 0 ужесточает), а
    ``DECISION_RISK_FEW_SHOT_N`` (по умолчанию ``RISK_FEW_SHOT_MIN_N``, значение
    < 2 отключает) задаёт порог few-shot переобучения под новый вопрос.

    ``DECISION_CASCADE_POLICY=benefit`` включает benefit-aware каскад: вопрос
    эскалируется, только если ожидаемая выгода внешней модели превышает цену
    вызова. Точность внешней модели ``q`` берётся из
    ``DECISION_CASCADE_REMOTE_ACCURACY`` (число или JSON, снятый
    ``python -m eval.competitors.fit_remote_accuracy``), цена эскалации — из
    ``DECISION_CASCADE_ESCALATION_COST`` (0.05), цена ошибки — из
    ``DECISION_CASCADE_ERROR_COST`` (1.0). Обратная связь (``POST /v1/feedback``)
    уточняет ``q`` по реальным ответам внешней модели: вес априорной оценки в
    псевдонаблюдениях — ``DECISION_CASCADE_FEEDBACK_SHRINKAGE`` (50). Политика
    benefit важнее risk-бюджета.
    """
    url = os.environ.get("DECISION_CASCADE_FALLBACK_URL")
    if not url:
        raise RuntimeError(
            "DECISION_BACKEND=cascade requires DECISION_CASCADE_FALLBACK_URL"
        )
    primary = LocalHeadsBackend(service or create_service())
    fallback = HttpSystemOneBackend(
        url,
        model=os.environ.get("DECISION_CASCADE_FALLBACK_MODEL", "remote"),
        modalities=parse_modalities(
            os.environ.get("DECISION_CASCADE_FALLBACK_MODALITIES", "text")
        ),
        **http_backend_options(),
    )
    threshold = float(os.environ.get("DECISION_CASCADE_THRESHOLD", "0.6"))
    policy = os.environ.get("DECISION_CASCADE_POLICY", "").strip().lower()
    if policy in {"benefit", "benefit-aware"}:
        raw_accuracy = os.environ.get("DECISION_CASCADE_REMOTE_ACCURACY")
        if not raw_accuracy:
            raise RuntimeError(
                "DECISION_CASCADE_POLICY=benefit requires "
                "DECISION_CASCADE_REMOTE_ACCURACY (см. "
                "python -m eval.competitors.fit_remote_accuracy)"
            )
        from src.backends.benefit import (
            BenefitAwareCascadeBackend,
            load_remote_accuracy,
        )

        return BenefitAwareCascadeBackend(
            primary,
            fallback,
            load_remote_accuracy(
                raw_accuracy,
                conservative=os.environ.get(
                    "DECISION_CASCADE_REMOTE_ACCURACY_BOUND", "point"
                ).strip().lower()
                == "lower",
            ),
            escalation_cost=float(
                os.environ.get("DECISION_CASCADE_ESCALATION_COST", "0.05")
            ),
            error_cost=float(os.environ.get("DECISION_CASCADE_ERROR_COST", "1.0")),
            feedback_shrinkage=float(
                os.environ.get("DECISION_CASCADE_FEEDBACK_SHRINKAGE", "50")
            ),
        )
    risk_budget = os.environ.get("DECISION_CASCADE_RISK_BUDGET")
    if risk_budget is not None:
        from src.backends.cascade import RiskAwareCascadeBackend
        from src.risk import RISK_FEW_SHOT_MIN_N, ErrorRiskModel, gate_verdict

        model_path = os.environ.get("DECISION_RISK_MODEL_PATH") or str(
            Path(os.environ.get("DECISION_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
            / "risk.json"
        )
        if not Path(model_path).exists():
            raise RuntimeError(
                "DECISION_CASCADE_RISK_BUDGET requires a fitted risk model at "
                f"{model_path} (set DECISION_RISK_MODEL_PATH or train it via "
                "python -m eval.risk.run)"
            )
        risk_model = ErrorRiskModel.load(model_path)
        gate_delta = float(os.environ.get("DECISION_RISK_GATE_DELTA", "0.0"))
        gate_accepted, gate_worse = gate_verdict(risk_model.gate, gate_delta)
        if not gate_accepted:
            # hard gate отклонил риск-вариант: риск-каскад не применяем
            logger.warning(
                "hard gate отклонил risk-модель (%s); риск-каскад заменён "
                "порогом уверенности %.2f",
                ", ".join(gate_worse) or "метрики не хуже",
                threshold,
            )
        raw_few_shot = os.environ.get("DECISION_RISK_FEW_SHOT_N")
        if raw_few_shot in (None, ""):
            few_shot_n = RISK_FEW_SHOT_MIN_N
        else:
            few_shot_n = int(raw_few_shot)
        return RiskAwareCascadeBackend(
            primary,
            fallback,
            risk_model,
            risk_budget=float(risk_budget),
            fallback_threshold=threshold,
            gate_delta=gate_delta,
            # значение < 2 (в частности 0) отключает few-shot переобучение
            few_shot_n=few_shot_n if few_shot_n >= 2 else None,
        )
    return CascadeBackend(primary, fallback, threshold=threshold)


app = FastAPI(title="decision-cascade API", version=__version__)

# Политика депрекации: реестр проверяется на старте, чтобы объявленный срок
# жизни поверхности не мог быть короче двух минорных версий.
validate_registry()


@app.middleware("http")
async def deprecation_headers(request: Request, call_next):
    """Добавляет ``Deprecation``/``Sunset``/``Link`` для устаревших путей.

    Реестр пуст, пока устаревших поверхностей нет, — middleware тогда ничего не
    меняет. Предупреждение в лог пишется один раз за жизнь процесса.
    """
    response = await call_next(request)
    deprecation = deprecation_for(request.url.path)
    if deprecation is not None:
        warn_once(request.url.path)
        response.headers.update(deprecation.headers())
    return response


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Request id, метрики и одна строка доступа на каждый запрос.

    ``X-Request-ID`` из запроса сохраняется, если он безопасен, иначе
    генерируется новый; заголовок возвращается клиенту, поэтому по нему можно
    найти и лог сервиса, и лог вызывающей стороны.
    """
    request_id = new_request_id(request.headers.get(REQUEST_ID_HEADER))
    token = bind_request_id(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - started
        # маршрут известен только после routing — шаблон пути берём здесь
        path = metrics.route_template(request)
        metrics.observe_request(request.method, path, status, elapsed)
        access_record(
            access_logger,
            method=request.method,
            path=path,
            status=status,
            duration_ms=elapsed * 1000.0,
            request_id=request_id,
        )
        reset_request_id(token)


# Единый исполнитель решений для всех эндпоинтов: локальные головы, внешний
# SystemOne-сервис или каскад (по DECISION_BACKEND). Локальный сервис создаётся
# лениво — только если выбранный бэкенд его действительно использует.
backend: Backend = create_backend()


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str
    criteria: list[str] | None = Field(None, min_length=2, max_length=2)


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str
    options: list[str] = Field(..., min_length=2, max_length=255)
    criteria: dict[str, str] | None = Field(None, max_length=255)

    @field_validator("options")
    @classmethod
    def _unique_options(cls, options: list[str]) -> list[str]:
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        return options


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str
    levels: list[str] = Field(..., min_length=2, max_length=255)
    criteria: list[str] | None = Field(None, max_length=255)

    @field_validator("levels")
    @classmethod
    def _unique_levels(cls, levels: list[str]) -> list[str]:
        if len(set(levels)) != len(levels):
            raise ValueError("levels must be unique")
        return levels

    @model_validator(mode="after")
    def _criteria_match_levels(self) -> ScoreQuestion:
        if self.criteria is not None and len(self.criteria) != len(self.levels):
            raise ValueError("criteria length must match levels length")
        return self


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class DecisionRequest(BaseModel):
    state: str = Field(..., min_length=1)
    questions: dict[str, Question] = Field(..., min_length=1)


class DecisionResponse(BaseModel):
    schema_version: str
    decisions: dict[str, Annotated[DecisionResult, Field(discriminator="type")]]
    trained: bool | None = None


class BatchDecisionRequest(BaseModel):
    states: list[str] = Field(..., min_length=1, max_length=256)
    questions: dict[str, Question] = Field(..., min_length=1)

    @field_validator("states")
    @classmethod
    def _non_empty_states(cls, states: list[str]) -> list[str]:
        if any(not state.strip() for state in states):
            raise ValueError("states must be non-empty strings")
        return states


class BatchDecisionResponse(BaseModel):
    schema_version: str
    results: list[dict[str, Annotated[DecisionResult, Field(discriminator="type")]]]
    trained: bool | None = None


class FeedbackExample(BaseModel):
    """Размеченный пример: решение, которое отдал сервис, и его корректность."""

    question: str = Field(..., min_length=1)
    decision: Annotated[DecisionResult, Field(discriminator="type")]
    correct: bool

    @model_validator(mode="after")
    def _decision_is_well_formed(self) -> FeedbackExample:
        """Решение должно быть таким, какое ядро вправе выдать.

        Примеры попадают в копилку риск-модели, поэтому негодное значение
        (вероятность вне [0, 1], распределение, не суммирующееся в единицу,
        исход вне списка вариантов) отклоняем сразу, а не учим на нём.
        """
        decision = self.decision
        confidence = decision.confidence
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence}")
        if decision.type == "noul" and not 0.0 <= decision.noul <= 1.0:
            raise ValueError(f"noul must be in [0.0, 1.0], got {decision.noul}")
        probabilities = getattr(decision, "probabilities", None)
        if probabilities:
            if any(value < 0.0 for value in probabilities.values()):
                raise ValueError("probabilities must be non-negative")
            total = sum(probabilities.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"probabilities must sum to 1.0, got {total}")
            if decision.type == "choice" and decision.choice not in probabilities:
                raise ValueError(
                    f"choice must be one of the probability keys, got {decision.choice!r}"
                )
        return self


class FeedbackRequest(BaseModel):
    examples: list[FeedbackExample] = Field(..., min_length=1, max_length=1024)


class FeedbackQuestionReport(BaseModel):
    question: str
    #: примеров принято в этом запросе
    accepted: int
    #: всего примеров накоплено по вопросу (после этого запроса)
    examples: int
    #: чужих строк (примеров других вопросов), на которых дообучена модель;
    #: 0 — обучение только на своих, см. docs/risk-aware-cascade.md
    reference: int
    #: порог few-shot достигнут и риск-модель по вопросу переобучена
    retrained: bool
    #: обновлённая точность внешней модели по вопросу (benefit-каскад);
    #: ``None``, если бэкенд не ведёт такую оценку
    remote_accuracy: float | None = None


class FeedbackResponse(BaseModel):
    accepted: int
    questions: list[FeedbackQuestionReport]


def _batch_trained(usages: list[dict[str, Any]]) -> bool | None:
    """Сводный статус батча: ``False`` — если хоть где-то не обучены, иначе ``True``/``None``."""
    values = [usage.get("trained") for usage in usages]
    if any(value is False for value in values):
        return False
    if values and all(value is True for value in values):
        return True
    return None


def _untrained_response(exc: UntrainedDecisionError) -> JSONResponse:
    """409 на запрос схемы, под которую нет обученных голов (политика refuse)."""
    metrics.observe_refusal()
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "code": "UNTRAINED_HEADS",
                "message": str(exc),
                "signature": exc.signature,
                "checkpoint_dir": exc.checkpoint_dir,
            }
        },
    )


def _backend_unavailable(exc: RemoteBackendError) -> JSONResponse:
    """502: внешний SystemOne-сервис недоступен после исчерпания таймаута/retry.

    Наружу отдаём только имя бэкенда и HTTP-статус (если был): тело запроса
    (state/images) в ответ и в лог не попадает.
    """
    logger.warning(
        "backend %s unavailable: %s (status=%s)",
        exc.backend,
        exc,
        exc.status_code,
    )
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "code": "BACKEND_UNAVAILABLE",
                "message": f"backend {exc.backend!r} is unavailable",
                "backend": exc.backend,
                "status_code": exc.status_code,
            }
        },
    )


def _feedback_unsupported() -> JSONResponse:
    """409: бэкенд не принимает обратную связь — few-shot выключен или риск-модели нет."""
    return JSONResponse(
        status_code=409,
        content={
            "error": {
                "code": "FEEDBACK_UNSUPPORTED",
                "message": (
                    "backend does not accept labelled examples; few-shot risk "
                    "retraining requires a risk-aware cascade "
                    "(DECISION_BACKEND=cascade with DECISION_CASCADE_RISK_BUDGET "
                    "and DECISION_RISK_FEW_SHOT_N >= 2)"
                ),
            }
        },
    )


def _engine_info(backend: Backend) -> dict[str, Any] | None:
    """Лучшее, что можно сказать о движке: локальный сервис или каскад."""
    engine = getattr(backend, "engine", None)
    describe = getattr(engine, "info", None)
    if callable(describe):
        return describe()
    primary = getattr(backend, "primary", None)
    fallback = getattr(backend, "fallback", None)
    if primary is not None or fallback is not None:
        info: dict[str, Any] = {
            "primary": _engine_info(primary) if primary is not None else None,
            "fallback": _engine_info(fallback) if fallback is not None else None,
        }
        risk_budget = getattr(backend, "risk_budget", None)
        if risk_budget is not None:
            info["risk_budget"] = risk_budget
            # видно ли в /info, что hard gate отклонил риск-вариант
            info["risk_gate"] = (
                "rejected" if getattr(backend, "gate_rejected", False) else "ok"
            )
            info["risk_gate_delta"] = getattr(backend, "gate_delta", 0.0)
            few_shot_n = getattr(backend, "few_shot_n", None)
            if few_shot_n is not None:
                info["risk_few_shot_n"] = few_shot_n
                info["risk_few_shot"] = getattr(
                    backend, "few_shot_questions", []
                )
        return info
    return None


def _to_specs(questions: dict[str, Question]) -> dict[str, QuestionSpec]:
    specs = {}
    for name, q in questions.items():
        if q.type == "noul":
            criteria = tuple(q.criteria) if q.criteria else None
            specs[name] = NoulSpec(instructions=q.instructions, criteria=criteria)
        elif q.type == "choice":
            criteria = dict(q.criteria) if q.criteria else None
            specs[name] = ChoiceSpec(
                instructions=q.instructions, options=tuple(q.options), criteria=criteria
            )
        elif q.type == "score":
            criteria = tuple(q.criteria) if q.criteria else None
            specs[name] = ScoreSpec(
                instructions=q.instructions, levels=tuple(q.levels), criteria=criteria
            )
    return specs


# Все эндпоинты — синхронные (``def``): FastAPI выполняет их в threadpool, поэтому
# тяжёлый CPU-инференс не блокирует event loop и остальные соединения.
@app.post("/decisions", response_model=DecisionResponse)
def create_decisions(req: DecisionRequest) -> DecisionResponse | JSONResponse:
    """Оценивает все вопросы параллельно одним прогоном энкодера + голов."""
    try:
        result = backend.decide(req.state, _to_specs(req.questions))
    except UntrainedDecisionError as exc:
        return _untrained_response(exc)
    except RemoteBackendError as exc:
        return _backend_unavailable(exc)
    metrics.observe_decisions(result.decisions, result.usage.get("trained"))
    return DecisionResponse(
        schema_version=SCHEMA_VERSION,
        decisions=result.decisions,
        trained=result.usage.get("trained"),
    )


@app.post("/decisions/batch", response_model=BatchDecisionResponse)
def create_decisions_batch(req: BatchDecisionRequest) -> BatchDecisionResponse | JSONResponse:
    """Батч: все состояния и вопросы за один вызов энкодера (внутри — батчами)."""
    try:
        backend_results = backend.decide_many(req.states, _to_specs(req.questions))
    except UntrainedDecisionError as exc:
        return _untrained_response(exc)
    except RemoteBackendError as exc:
        return _backend_unavailable(exc)
    for result in backend_results:
        metrics.observe_decisions(result.decisions, result.usage.get("trained"))
    return BatchDecisionResponse(
        schema_version=SCHEMA_VERSION,
        results=[result.decisions for result in backend_results],
        trained=_batch_trained([result.usage for result in backend_results]),
    )


@app.post(
    "/v1/feedback", response_model=FeedbackResponse, response_model_exclude_none=True
)
def submit_feedback(req: FeedbackRequest) -> FeedbackResponse | JSONResponse:
    """Размеченные примеры решений -> few-shot дообучение риск-модели под вопрос.

    Транспорт примеров для риск-модели (``add_examples`` риск-каскада): клиент
    повторяет решение, которое отдал сервис, и говорит, было ли оно верным.
    Пример относится к вопросу по его имени — тому же, что в запросе решений.
    Достигнув ``DECISION_RISK_FEW_SHOT_N`` примеров, каскад переобучает
    риск-модель вопроса (схема {чужие строки + свои}) и применяет её к нему же;
    hard gate при этом не ослабляется.

    В ответе по каждому вопросу: ``examples`` — сколько примеров накоплено после
    запроса, ``reference`` — сколько чужих строк (примеров других вопросов)
    ушло в обучение. ``reference = 0`` — вырожденный режим «только свои»: он
    измерен и хуже, пока своих примеров мало (см. docs/risk-aware-cascade.md),
    поэтому копить примеры лучше по нескольким вопросам.

    Benefit-каскад (``DECISION_CASCADE_POLICY=benefit``) использует тот же
    транспорт иначе: он не переобучает модель, а уточняет точность внешней
    модели по вопросу ``q`` — только для решений, которые действительно ушли
    наружу (ответы локальных голов ``q`` не меняют). По вопросу в ответе
    приходит обновлённая ``remote_accuracy``.

    Отвечает ``409 FEEDBACK_UNSUPPORTED``, если бэкенд не принимает обратную
    связь: каскад не выбран, риск-вариант с few-shot выключен
    (``DECISION_RISK_FEW_SHOT_N`` < 2) или это обычный порог уверенности.
    """
    if not backend.capabilities.accepts_feedback:
        return _feedback_unsupported()
    add_examples = getattr(backend, "add_examples", None)
    if not callable(add_examples):
        return _feedback_unsupported()
    grouped: dict[str, list[FeedbackExample]] = {}
    for example in req.examples:
        grouped.setdefault(example.question, []).append(example)
    # ``example_counts`` — снимок до этого запроса: из него и получаем «после»
    counts = getattr(backend, "example_counts", {})
    reference_rows = getattr(backend, "reference_rows", None)
    remote_accuracy_for = getattr(backend, "remote_accuracy_for", None)
    reports: list[FeedbackQuestionReport] = []
    for question, examples in grouped.items():
        # чужие строки считаем до приёма примеров: ровно они уйдут в обучение
        n_reference = 0
        if callable(reference_rows):
            features, _labels = reference_rows(question)
            n_reference = 0 if features is None else len(features)
        retrained = add_examples(
            question,
            [example.decision for example in examples],
            [example.correct for example in examples],
        )
        # Копилку читаем после приёма: benefit-каскад считает не все присланные
        # примеры, а только те, что относятся к ответам внешней модели.
        after_counts = getattr(backend, "example_counts", {})
        reports.append(
            FeedbackQuestionReport(
                question=question,
                accepted=len(examples),
                examples=int(
                    after_counts.get(question, int(counts.get(question, 0)) + len(examples))
                ),
                reference=n_reference,
                retrained=retrained is not None,
                remote_accuracy=(
                    float(remote_accuracy_for(question))
                    if callable(remote_accuracy_for)
                    else None
                ),
            )
        )
    return FeedbackResponse(
        accepted=sum(report.accepted for report in reports), questions=reports
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: процесс жив и event loop отвечает. Зависимости не проверяются."""
    return {"status": "ok", "version": __version__}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Метрики Prometheus: запросы, латентность, решения, готовность.

    Эндпоинт не учитывается в собственных метриках (иначе скрейпер раздувал бы
    счётчик запросов). Формат — Prometheus text 0.0.4.
    """
    return Response(
        content=metrics.render(),
        media_type=metrics.CONTENT_TYPE_LATEST,
    )


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: бэкенд действительно может обслуживать решения.

    Для локального бэкенда это загруженный энкодер и читаемые головы (при
    ``DECISION_UNTRAINED=refuse`` — хотя бы один пак); для каскада — готовность
    primary, а недоступный fallback отмечается как ``degraded``. Ответ 503, пока
    сервис не готов, — балансировщик не должен слать на него трафик.
    """
    report = backend_readiness(backend)
    report.setdefault("backend", backend.name)
    report["version"] = __version__
    metrics.set_ready(bool(report.get("ready")))
    if "heads_loaded" in report:
        metrics.set_heads_loaded(int(report["heads_loaded"]))
    return JSONResponse(
        status_code=200 if report.get("ready") else 503,
        content=report,
    )


@app.get("/info")
def get_info() -> dict[str, Any]:
    """Конфигурация сервиса: бэкенд, модальности и статус локальных голов."""
    return {
        "version": __version__,
        "backend": backend.name,
        "modalities": sorted(backend.capabilities.modalities),
        # принимает ли бэкенд обратную связь (few-shot риск-модели, /v1/feedback)
        "accepts_feedback": backend.capabilities.accepts_feedback,
        "engine": _engine_info(backend),
    }


@app.post("/v1/systemone", response_model=SystemOneResponse)
def create_systemone_decisions(
    req: SystemOneRequest,
) -> SystemOneResponse | JSONResponse:
    """Стандартный протокол Jev/SystemOne: state + типизированные вопросы -> ответы.

    Неподдерживаемая бэкендом модальность (например, изображения на текстовом
    локальном бэкенде) отклоняется типизированной ошибкой до исполнения.
    """
    requested = requested_modalities(images=req.images, videos=req.videos)
    missing = missing_modalities(requested, backend.capabilities.modalities)
    if missing:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "MODALITY_NOT_SUPPORTED",
                    "message": (
                        f"backend {backend.name!r} does not support "
                        f"{sorted(missing)}"
                    ),
                    "supported": sorted(backend.capabilities.modalities),
                    "requested": sorted(requested),
                }
            },
        )
    try:
        specs = systemone_to_specs(req.questions)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_QUESTION", "message": str(exc)}},
        )
    try:
        result = backend.decide(
            req.state,
            specs,
            images=req.images,
            videos=req.videos,
            media_kwargs=req.media_kwargs,
        )
    except UntrainedDecisionError as exc:
        return _untrained_response(exc)
    except RemoteBackendError as exc:
        return _backend_unavailable(exc)
    answers = {
        name: result_to_answer_model(decision)
        for name, decision in result.decisions.items()
    }
    metrics.observe_decisions(result.decisions, result.usage.get("trained"))
    return SystemOneResponse(
        model=backend.name, answers=answers, usage=result.usage
    )