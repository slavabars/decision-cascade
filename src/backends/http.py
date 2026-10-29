"""Бэкенд внешнего сервиса, совместимого с протоколом Jev/SystemOne.

Годится для Cloudflare Clef / Clef-Flash (Workers AI, SGLang, vLLM), Perplexity
Decider и любого другого сервиса на ``POST /v1/systemone``. Модальности
(text/image/video) объявляются вызывающей стороной и пробрасываются как есть.

Таймауты и retry. ``timeout`` — бюджет **всего** вызова ``decide``, а не
отдельной операции: попытки делят его между собой, поэтому зависший внешний
сервис не держит запрос дольше заданного времени. Повторяем только то, что
имеет смысл повторять: сетевые ошибки, таймауты и ответы 429/5xx. Ответ 4xx —
ошибка протокола, повтор её не исправит. Тело запроса (state/images) не
логируется и не попадает в текст исключения.
"""
from __future__ import annotations

import time
from typing import Any, Self

import httpx

from src.backends.base import (
    IMAGE,
    TEXT,
    VIDEO,
    BackendResult,
    Capabilities,
    decide_many_states,
)
from src.backends.protocol import answer_to_result, spec_to_request
from src.errors import RemoteBackendError
from src.sampler.questions import QuestionSpec

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 1
#: Пауза между попытками (секунды); не выходит за остаток бюджета.
RETRY_BACKOFF = 0.2
#: Статусы, которые имеет смысл повторить: перегрузка и временные отказы.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: Таймаут проверки готовности: проба не должна задерживать балансировщик.
READY_PROBE_TIMEOUT = 1.0
#: Таймаут установки соединения внутри общего бюджета.
CONNECT_TIMEOUT = 10.0


class HttpSystemOneBackend:
    def __init__(
        self,
        base_url: str,
        model: str = "remote",
        modalities: tuple[str, ...] | list[str] = (TEXT,),
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ):
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}")
        self.name = model
        self.capabilities = Capabilities(modalities=frozenset(modalities))
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._base_url = base_url
        self._client = client or httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport
        )

    def decide(
        self,
        state: str,
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> BackendResult:
        payload: dict[str, Any] = {
            "model": self.name,
            "state": state,
            "questions": {
                name: spec_to_request(spec) for name, spec in questions.items()
            },
        }
        if images is not None:
            payload["images"] = images
        if videos is not None:
            payload["videos"] = videos
        if media_kwargs is not None:
            payload["media_kwargs"] = media_kwargs
        resp = self._post(payload)
        try:
            body = resp.json()
        except ValueError as exc:
            raise RemoteBackendError(
                f"{self.name}: invalid JSON in /v1/systemone response",
                backend=self.name,
                status_code=resp.status_code,
            ) from exc
        decisions = {
            name: answer_to_result(answer)
            for name, answer in body.get("answers", {}).items()
        }
        return BackendResult(decisions=decisions, usage=dict(body.get("usage", {})))

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """POST с бюджетом ``timeout`` на все попытки и одним retry по умолчанию."""
        deadline = time.monotonic() + self.timeout
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                resp = self._client.post(
                    "/v1/systemone",
                    json=payload,
                    timeout=self._timeout_for(remaining),
                )
            except httpx.HTTPError as exc:
                last = exc
                if attempt < self.retries:
                    self._backoff(deadline)
                    continue
                raise RemoteBackendError(
                    f"{self.name}: {type(exc).__name__}",
                    backend=self.name,
                ) from exc
            if resp.status_code in RETRY_STATUS and attempt < self.retries:
                last = RemoteBackendError(
                    f"{self.name}: HTTP {resp.status_code}",
                    backend=self.name,
                    status_code=resp.status_code,
                )
                self._backoff(deadline)
                continue
            if resp.status_code >= 400:
                raise RemoteBackendError(
                    f"{self.name}: HTTP {resp.status_code}",
                    backend=self.name,
                    status_code=resp.status_code,
                )
            return resp
        if isinstance(last, RemoteBackendError):
            raise last
        raise RemoteBackendError(
            f"{self.name}: timeout budget {self.timeout:g}s exhausted",
            backend=self.name,
        )

    @staticmethod
    def _timeout_for(remaining: float) -> httpx.Timeout:
        connect = min(CONNECT_TIMEOUT, remaining)
        return httpx.Timeout(remaining, connect=connect)

    @staticmethod
    def _backoff(deadline: float) -> None:
        pause = min(RETRY_BACKOFF, deadline - time.monotonic())
        if pause > 0:
            time.sleep(pause)

    def readiness(self) -> dict[str, Any]:
        """Готовность: клиент открыт и внешний сервис отвечает на ``/health``.

        Для протокола SystemOne ``/health`` не обязателен, поэтому любой ответ
        ниже 500 (включая 404) считается признаком живого сервиса. Проба
        ограничена ``READY_PROBE_TIMEOUT`` и не повторяется.
        """
        report: dict[str, Any] = {
            "backend": self.name,
            "remote": self._base_url,
            "timeout": self.timeout,
        }
        if self._client.is_closed:
            report.update(ready=False, reason="client closed")
            return report
        try:
            resp = self._client.get(
                "/health", timeout=min(READY_PROBE_TIMEOUT, self.timeout)
            )
        except httpx.HTTPError as exc:
            report.update(ready=False, reason=type(exc).__name__)
            return report
        report.update(ready=resp.status_code < 500, remote_status=resp.status_code)
        return report

    def decide_many(
        self,
        states: list[str],
        questions: dict[str, QuestionSpec],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        media_kwargs: dict[str, Any] | None = None,
    ) -> list[BackendResult]:
        """Внешний сервис вызывается по одному состоянию (у протокола нет батча)."""
        return decide_many_states(
            self,
            states,
            questions,
            images=images,
            videos=videos,
            media_kwargs=media_kwargs,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parse_modalities(value: str) -> tuple[str, ...]:
    """Разбирает строку вида "text,image,video" в кортеж модальностей."""
    parts = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {TEXT, IMAGE, VIDEO}
    unknown = set(parts) - allowed
    if unknown:
        raise ValueError(f"unknown modalities: {sorted(unknown)}")
    return parts or (TEXT,)
